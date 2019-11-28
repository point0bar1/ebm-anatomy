#######################################
# ## TRAIN EBM USING IMAGE DATASET ## #
#######################################

import torch as t
import torchvision.transforms as tr
import torchvision.datasets as datasets
import json
import os
from nets import VanillaNet, NonlocalNet
from utils import download_flowers_data, plot_ims, plot_diagnostics

# directory for experiment results
EXP_DIR = './out_data/flowers_convergent_1/'
# json file with experiment config
CONFIG_FILE = './config_locker/flowers_convergent.json'


#######################
# ## INITIAL SETUP ## #
#######################

# load experiment config
with open(CONFIG_FILE) as file:
    config = json.load(file)

# make directory for saving results
if os.path.exists(EXP_DIR):
    # prevents overwriting old experiment folders by accident
    raise RuntimeError('Folder "{}" already exists. Please use a different "EXP_DIR".'.format(EXP_DIR))
else:
    os.makedirs(EXP_DIR)
    for folder in ['checkpoints', 'shortrun', 'longrun', 'plots', 'code']:
        os.mkdir(EXP_DIR + folder)

# save copy of code in the experiment folder
def save_code():
    def save_file(file_name):
        file_in = open('./' + file_name, 'r')
        file_out = open(EXP_DIR + 'code/' + os.path.basename(file_name), 'w')
        for line in file_in:
            file_out.write(line)
    for file in ['train_data.py', 'nets.py', 'utils.py', CONFIG_FILE]:
        save_file(file)
save_code()

# set seed for cpu and CUDA, get device
t.manual_seed(config['seed'])
if t.cuda.is_available():
    t.cuda.manual_seed_all(config['seed'])
device = t.device('cuda' if t.cuda.is_available() else 'cpu')


########################
# ## TRAINING SETUP # ##
########################

print('Setting up network and optimizer...')
# set up network
net_bank = {'vanilla': VanillaNet, 'nonlocal': NonlocalNet}
f = net_bank[config['net_type']](n_c=config['im_ch']).to(device)
# set up optimizer
optim_bank = {'adam': t.optim.Adam, 'sgd': t.optim.SGD}
if config['optimizer_type'] == 'sgd' and config['epsilon'] > 0:
    # scale learning rate according to langevin noise for invariant tuning
    config['lr_init'] *= (config['epsilon'] ** 2) / 2
    config['lr_min'] *= (config['epsilon'] ** 2) / 2
optim = optim_bank[config['optimizer_type']](f.parameters(), lr=config['lr_init'])

print('Processing data...')
# make tensor of training data
if config['data'] == 'flowers':
    download_flowers_data()
data = {'cifar10': lambda path, func: datasets.CIFAR10(root=path, transform=func, download=True),
        'mnist': lambda path, func: datasets.MNIST(root=path, transform=func, download=True),
        'flowers': lambda path, func: datasets.ImageFolder(root=path, transform=func)}
transform = tr.Compose([tr.Resize(config['im_sz']),
                        tr.CenterCrop(config['im_sz']),
                        tr.ToTensor(),
                        tr.Normalize(tuple(0.5*t.ones(config['im_ch'])), tuple(0.5*t.ones(config['im_ch'])))])
q = t.stack([x[0] for x in data[config['data']]('./data/' + config['data'], transform)]).to(device)

# initialize persistent images from noise (one persistent image for each data image)
# s_t_0 is used when init_type == 'persistent' in sample_s_t()
s_t_0 = 2 * t.rand_like(q) - 1


################################
# ## FUNCTIONS FOR SAMPLING ## #
################################

# sample batch from given array of images
def sample_image_set(image_set):
    rand_inds = t.randperm(image_set.shape[0])[0:config['batch_size']]
    return image_set[rand_inds], rand_inds

# sample positive images from dataset distribution q (add noise to ensure min sd is at least langevin noise sd)
def sample_q():
    x_q = sample_image_set(q)[0]
    return x_q + config['data_epsilon'] * t.randn_like(x_q)

# initialize and update images with langevin dynamics to obtain samples from finite-step MCMC distribution s_t
def sample_s_t(L, init_type, update_s_t_0=True):
    # get initial mcmc states for langevin updates ("persistent", "data", "uniform", or "gaussian")
    def sample_s_t_0():
        if init_type == 'persistent':
            return sample_image_set(s_t_0)
        elif init_type == 'data':
            return sample_q(), None
        elif init_type == 'uniform':
            noise_image = 2 * t.rand([config['batch_size'], config['im_ch'], config['im_sz'], config['im_sz']]) - 1
            return noise_image.to(device), None
        elif init_type == 'gaussian':
            noise_image = t.randn([config['batch_size'], config['im_ch'], config['im_sz'], config['im_sz']])
            return noise_image.to(device), None
        else:
            raise RuntimeError('Invalid method for "init_type" (use "persistent", "data", "uniform", or "gaussian")')

    # initialize MCMC samples
    x_s_t_0, s_t_0_inds = sample_s_t_0()

    # iterative langevin updates of MCMC samples
    x_s_t = t.autograd.Variable(x_s_t_0.clone(), requires_grad=True)
    r_s_t = t.zeros(1).to(device)  # variable r_s_t (Section 3.2) to record average gradient magnitude
    for ell in range(L):
        f_prime = t.autograd.grad(f(x_s_t).sum(), [x_s_t])[0]
        x_s_t.data += - f_prime + config['epsilon'] * t.randn_like(x_s_t)
        r_s_t += f_prime.view(f_prime.shape[0], -1).norm(dim=1).mean()

    if init_type == 'persistent' and update_s_t_0:
        # update persistent image bank
        s_t_0.data[s_t_0_inds] = x_s_t.detach().data.clone()

    return x_s_t.detach(), r_s_t.squeeze() / L


#######################
# ## TRAINING LOOP ## #
#######################

# containers for diagnostic records (see Section 3)
d_s_t_record = t.zeros(config['num_train_iters']).to(device)  # energy difference between positive and negative samples
r_s_t_record = t.zeros(config['num_train_iters']).to(device)  # average image gradient magnitude along Langevin path

print('Training has started.')
for i in range(config['num_train_iters']):
    # obtain positive and negative samples
    x_q = sample_q()
    x_s_t, r_s_t = sample_s_t(L=config['num_shortrun_steps'], init_type=config['shortrun_init'])

    # calculate ML computational loss d_s_t (Section 3) for data and shortrun samples
    d_s_t = f(x_q).mean() - f(x_s_t).mean()
    if config['epsilon'] > 0:
        # scale loss with the langevin implementation
        d_s_t *= 2 / (config['epsilon'] ** 2)
    # stochastic gradient ML update for model weights
    optim.zero_grad()
    d_s_t.backward()
    optim.step()

    # record diagnostics
    d_s_t_record[i] = d_s_t.detach().data
    r_s_t_record[i] = r_s_t

    # anneal learning rate
    for lr_gp in optim.param_groups:
        lr_gp['lr'] = max(config['lr_min'], lr_gp['lr'] * config['lr_decay'])

    # print and save learning info
    if (i + 1) == 1 or (i + 1) % config['log_freq'] == 0:
        print('{:>6d}   d_s_t={:>14.9f}   r_s_t={:>14.9f}'.format(i+1, d_s_t.detach().data, r_s_t))
        # visualize synthesized images
        plot_ims(EXP_DIR + 'shortrun/' + 'x_s_t_{:>06d}.png'.format(i+1), x_s_t)
        if config['shortrun_init'] == 'persistent':
            plot_ims(EXP_DIR + 'shortrun/' + 'x_s_t_0_{:>06d}.png'.format(i+1), s_t_0[0:config['batch_size']])
        # save network weights
        t.save(f.state_dict(), EXP_DIR + 'checkpoints/' + 'net_{:>06d}.pth'.format(i+1))
        # plot diagnostics for energy difference d_s_t and gradient magnitude r_t
        if (i + 1) > 1:
            plot_diagnostics(i, d_s_t_record, r_s_t_record, EXP_DIR + 'plots/')

    # sample longrun chains to diagnose model steady-state
    if config['log_longrun'] and (i+1) % config['log_longrun_freq'] == 0:
        print('{:>6d}   Generating long-run samples. (L={:>6d} MCMC steps)'.format(i+1, config['num_longrun_steps']))
        x_p_theta = sample_s_t(L=config['num_longrun_steps'], init_type=config['longrun_init'], update_s_t_0=False)[0]
        plot_ims(EXP_DIR + 'longrun/' + 'longrun_{:>06d}.png'.format(i+1), x_p_theta)
        print('{:>6d}   Long-run samples saved.'.format(i+1))
