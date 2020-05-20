##############################
# ## EVALUATE TRAINED EBM ## #
##############################

import torch as t
import torchvision.transforms as tr
import torchvision.datasets as datasets

import matplotlib.pyplot as plt
import json
import os

from nets import VanillaNet, NonlocalNet
from utils import download_flowers_data, plot_ims

# directory for experiment results
EXP_DIR = './out_eval/flowers_convergent_eval_1/'
# json file with experiment config
CONFIG_FILE = './config_locker/eval_flowers_convergent.json'


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
    for folder in ['code']:
        os.mkdir(EXP_DIR + folder)

# save copy of code in the experiment folder
def save_code():
    def save_file(file_name):
        file_in = open('./' + file_name, 'r')
        file_out = open(EXP_DIR + 'code/' + os.path.basename(file_name), 'w')
        for line in file_in:
            file_out.write(line)
    for file in ['eval.py', 'nets.py', 'utils.py', CONFIG_FILE]:
        save_file(file)
save_code()

# set seed for cpu and CUDA, get device
t.manual_seed(config['seed'])
if t.cuda.is_available():
    t.cuda.manual_seed_all(config['seed'])
device = t.device('cuda' if t.cuda.is_available() else 'cpu')


####################
# ## EVAL SETUP # ##
####################

print('Setting up network...')
# set up network
net_bank = {'vanilla': VanillaNet, 'nonlocal': NonlocalNet}
f = net_bank[config['net_type']](n_c=config['im_ch'])
# load saved weights
f.load_state_dict(t.load(config['net_weight_path'], map_location=lambda storage, loc: storage.cpu()))
# put net on device
f.to(device)
# temperature from training
if config['train_epsilon'] > 0:
    temp = config['temp_factor'] * (config['train_epsilon'] ** 2) / 2
else:
    temp = config['temp_factor']

print('Processing initial MCMC states...')
if config['mcmc_init'] == 'uniform':
    q = 2 * t.rand([config['batch_size'], config['im_ch'], config['im_sz'], config['im_sz']]).to(device) - 1
elif config['mcmc_init'] == 'gaussian':
    q = t.randn([config['batch_size'], config['im_ch'], config['im_sz'], config['im_sz']]).to(device)
else:
    # make tensor of training data
    if config['mcmc_init'] == 'flowers':
        download_flowers_data()
    data = {'cifar10': lambda path, func: datasets.CIFAR10(root=path, transform=func, download=True),
            'mnist': lambda path, func: datasets.MNIST(root=path, transform=func, download=True),
            'flowers': lambda path, func: datasets.ImageFolder(root=path, transform=func)}
    transform = tr.Compose([tr.Resize(config['im_sz']),
                            tr.CenterCrop(config['im_sz']),
                            tr.ToTensor(),
                            tr.Normalize(tuple(0.5*t.ones(config['im_ch'])), tuple(0.5*t.ones(config['im_ch'])))])
    q = t.stack([x[0] for x in data[config['mcmc_init']]('./data/' + config['mcmc_init'], transform)]).to(device)
# get a random sample of initial states from image bank
x_s_t_0 = q[t.randperm(q.shape[0])[0:config['batch_size']]]


################################
# ## FUNCTIONS FOR SAMPLING ## #
################################

# langevin equation without MH adjustment
def langevin_grad():
    x_s_t = t.autograd.Variable(x_s_t_0.clone(), requires_grad=True)

    # sampling records
    grads = t.zeros(config['num_mcmc_steps'], config['batch_size'])
    ens = t.zeros(config['num_mcmc_steps'], config['batch_size'])

    # iterative langevin updates of MCMC samples
    for ell in range(config['num_mcmc_steps']):
        en = f(x_s_t) / temp
        ens[ell] = en.detach().cpu()
        grad = t.autograd.grad(en.sum(), [x_s_t])[0]
        if config['epsilon'] > 0:
            x_s_t.data += - ((config['epsilon']**2)/2) * grad + config['epsilon'] * t.randn_like(x_s_t)
            grads[ell] = ((config['epsilon']**2)/2) * grad.view(grad.shape[0], -1).norm(dim=1).cpu()
        else:
            x_s_t.data += - grad
            grads[ell] = grad.view(grad.shape[0], -1).norm(dim=1).cpu()
        if ell == 0 or (ell + 1) % config['log_freq'] == 0 or (ell + 1) == config['num_mcmc_steps']:
            print('Step {} of {}.   Ave. En={:>14.9f}   Ave. Grad={:>14.9f}'.
                  format(ell+1, config['num_mcmc_steps'], ens[ell].mean(), grads[ell].mean()))
    return x_s_t.detach(), ens, grads

# langevin equation with MH adjustment
def langevin_mh():
    x_s_t = t.autograd.Variable(x_s_t_0.clone(), requires_grad=True)

    # sampling records
    ens = t.zeros(config['num_mcmc_steps'], config['batch_size'])
    grads = t.zeros(config['num_mcmc_steps'], config['batch_size'])
    accepts = t.zeros(config['num_mcmc_steps'])

    # iterative langevin updates of MCMC samples
    for ell in range(config['num_mcmc_steps']):
        # get energy and gradient of current states
        en = f(x_s_t) / temp
        ens[ell] = en.detach().cpu()
        grad = t.autograd.grad(en.sum(), [x_s_t])[0]
        grads[ell] = ((config['epsilon'] ** 2)/2) * grad.view(grad.shape[0], -1).norm(dim=1).cpu()

        # get initial gaussian momenta
        p = t.randn_like(x_s_t)

        # get proposal states
        x_prop = x_s_t - ((config['epsilon'] ** 2)/2) * grad + config['epsilon'] * p
        # update momentum
        en_prop = f(x_prop) / temp
        grad_prop = t.autograd.grad(en_prop.sum(), [x_prop])[0]
        p_prop = p - (config['epsilon'] / 2) * (grad + grad_prop)

        # joint energy of states and auxiliary momentum variables
        joint_en_orig = en + 0.5 * t.sum((p ** 2).view(x_s_t.shape[0], -1), 1)
        joint_en_prop = en_prop + 0.5 * t.sum((p_prop ** 2).view(x_s_t.shape[0], -1), 1)

        # accept or reject states_prop using MH acceptance ratio
        accepted_proposals = t.rand_like(en) < t.exp(joint_en_orig - joint_en_prop)

        # update only states with accepted proposals
        x_s_t.data[accepted_proposals] = x_prop.data[accepted_proposals]
        accepts[ell] = float(accepted_proposals.sum().cpu()) / float(config['batch_size'])

        if ell == 0 or (ell + 1) % config['log_freq'] == 0 or (ell + 1) == config['num_mcmc_steps']:
            print('Step {} of {}.   Ave. En={:>14.9f}   Ave. Grad={:>14.9f}   Accept Rate={:>14.9f}'.
                  format(ell+1, config['num_mcmc_steps'], ens[ell].mean(), grads[ell].mean(), accepts[ell]))

    return x_s_t.detach(), ens, grads, accepts


###################################
# ## SAMPLE FROM LEARNED MODEL ## #
###################################

print('Sampling for {} Langevin steps.'.format(config['num_mcmc_steps']))
if config['use_mh_langevin']:
    x_s_t, en_record, grad_record, accept_record = langevin_mh()
    plt.plot(accept_record.numpy())
    plt.savefig(EXP_DIR + 'accept.png')
    plt.close()
else:
    x_s_t, en_record, grad_record = langevin_grad()

# visualize initial and synthesized images
plot_ims(EXP_DIR + 'initial_states.png', x_s_t_0)
plot_ims(EXP_DIR + 'sample_states.png', x_s_t)

# plot diagnostics
plt.plot(en_record.numpy())
plt.title('Energy over sampling path')
plt.xlabel('Langevin step')
plt.ylabel('energy')
plt.savefig(EXP_DIR + 'en.png')
plt.close()
plt.plot(grad_record.numpy())
plt.title('Gradient magnitude over sampling path')
plt.xlabel('Langevin step')
plt.ylabel('Gradient magnitude')
plt.savefig(EXP_DIR + 'grad.png')
plt.close()
