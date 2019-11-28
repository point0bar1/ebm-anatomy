#############################################
# ## TRAIN EBM USING 2D TOY DISTRIBUTION ## #
#############################################

import torch as t
import json
import os
from nets import ToyNet
from utils import plot_diagnostics, ToyDataset

# directory for experiment results
EXP_DIR = './out_toy/rings_convergent_1/'
# json file with experiment config
CONFIG_FILE = './config_locker/rings_convergent.json'


#######################
# ## INITIAL SETUP ## #
#######################

# load experiment config
with open(CONFIG_FILE) as file:
    config = json.load(file)

# make directory for saving results
if os.path.exists(EXP_DIR):
    # prevents overwriting old experiment folders by accident
    raise RuntimeError('Experiment folder "{}" already exists. Please use a different "EXP_DIR".'.format(EXP_DIR))
else:
    os.makedirs(EXP_DIR)
    for folder in ['checkpoints', 'landscape', 'plots', 'code']:
        os.mkdir(EXP_DIR + folder)

# save copy of code in the experiment folder
def save_code():
    def save_file(file_name):
        file_in = open('./' + file_name, 'r')
        file_out = open(EXP_DIR + 'code/' + os.path.basename(file_name), 'w')
        for line in file_in:
            file_out.write(line)
    for file in ['train_toy.py', 'nets.py', 'utils.py', CONFIG_FILE]:
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
net_bank = {'toy': ToyNet}
f = net_bank[config['net_type']]().to(device)
# set up optimizer
optim_bank = {'adam': t.optim.Adam, 'sgd': t.optim.SGD}
if config['optimizer_type'] == 'sgd' and config['epsilon'] > 0:
    # scale learning rate according to langevin noise for invariant tuning
    config['lr_init'] *= (config['epsilon'] ** 2) / 2
    config['lr_min'] *= (config['epsilon'] ** 2) / 2
optim = optim_bank[config['optimizer_type']](f.parameters(), lr=config['lr_init'])

print('Processing data...')
# toy dataset for which true samples can be obtained
q = ToyDataset(config['toy_type'], config['toy_groups'], config['toy_sd'],
               config['toy_radius'], config['viz_res'], config['kde_bw'])

# initialize persistent states from noise 
# s_t_0 is used when init_type == 'persistent' in sample_s_t()
s_t_0 = 2 * t.rand([config['s_t_0_size'], 2, 1, 1]).to(device) - 1


################################
# ## FUNCTIONS FOR SAMPLING ## #
################################

# sample batch from given array of states
def sample_state_set(state_set, batch_size=config['batch_size']):
    rand_inds = t.randperm(state_set.shape[0])[0:batch_size]
    return state_set[rand_inds], rand_inds

# sample positive states from toy 2d distribution q
def sample_q(batch_size=config['batch_size']): return t.Tensor(q.sample_toy_data(batch_size)).to(device)

# initialize and update states with langevin dynamics to obtain samples from finite-step MCMC distribution s_t
def sample_s_t(batch_size, L=config['num_mcmc_steps'], init_type=config['init_type'], update_s_t_0=True):
    # get initial mcmc states for langevin updates ("persistent", "data", "uniform", or "gaussian")
    def sample_s_t_0():
        if init_type == 'persistent':
            return sample_state_set(s_t_0, batch_size)
        elif init_type == 'data':
            return sample_q(batch_size), None
        elif init_type == 'uniform':
            return config['noise_init_factor'] * (2 * t.rand([batch_size, 2, 1, 1]) - 1).to(device), None
        elif init_type == 'gaussian':
            return config['noise_init_factor'] * t.randn([batch_size, 2, 1, 1]).to(device), None
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
        # update persistent state bank
        s_t_0.data[s_t_0_inds] = x_s_t.detach().data.clone()

    return x_s_t.detach(), r_s_t.squeeze() / L


#######################
# ## TRAINING LOOP ## #
#######################

# containers for diagnostic records (see Section 3)
d_s_t_record = t.zeros(config['num_train_iters']).to(device)  # energy difference between positive and negative samples
r_s_t_record = t.zeros(config['num_train_iters']).to(device)  # average state gradient magnitude along Langevin path

print('Training has started.')
for i in range(config['num_train_iters']):
    # obtain positive and negative samples
    x_q = sample_q()
    x_s_t, r_s_t = sample_s_t(batch_size=config['batch_size'])

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
    if (i + 1) == 1 or (i + 1) % config['log_info_freq'] == 0:
        print('{:>6d}   d_s_t={:>14.9f}   r_s_t={:>14.9f}'.format(i+1, d_s_t.detach().data, r_s_t))
        # save network weights
        t.save(f.state_dict(), EXP_DIR + 'checkpoints/' + 'net_{:>06d}.pth'.format(i+1))
        # plot diagnostics for energy difference d_s_t and gradient magnitude r_t
        if (i + 1) > 1:
            plot_diagnostics(i, d_s_t_record, r_s_t_record, EXP_DIR + 'plots/')

    # visualize density and log-density for groundtruth, learned energy, and short-run distributions
    if (i + 1) % config['log_viz_freq'] == 0:
        print('{:>6}   Visualizing true density, learned density, and short-run KDE.'.format(i+1))
        x_kde = sample_s_t(batch_size=config['batch_size_kde'], update_s_t_0=False)[0]
        q.plot_toy_density(True, f, config['epsilon'], x_kde, EXP_DIR+'landscape/'+'toy_viz_{:>06d}.pdf'.format(i+1))
        print('{:>6}   Visualizations saved.'.format(i + 1))
