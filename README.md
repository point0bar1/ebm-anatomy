# Code for **On the Anatomy of MCMC-Based Maximum Likelihood Learning of Energy-Based Models**

This repository will reproduce the main results from our paper:

**On the Anatomy of MCMC-Based Maximum Likelihood Learning of Energy-Based Models**<br/>Erik Nijkamp\*, Mitch Hill\*, Tian Han, Song-Chun Zhu, and Ying Nian Wu (*\*equal contributions*)<br/>https://arxiv.org/abs/1903.12370<br/>AAAI 2020.

The files ```train_data.py``` and ```train_toy.py``` are PyTorch-based implementations of Algorithm 1 for image datasets and toy 2D distributions respectively. Both files will measure and plot the diagnostic values $d_{s_t}$ and $r_t$ described in Section 3 during training. The file ```eval.py``` will sample from a saved checkpoint using either unadjusted Langevin dynamics or Metropolis-Hastings adjusted Langevin dynamics. We provide an appendix ```ebm-anatomy-appendix.pdf``` that contains further practical considerations and empirical observations.

## Config Files

The folder ```config_locker``` has several JSON files that reproduce different convergent and non-convergent learning outcomes for image datasets and toy distributions. Config files for evaluation of pre-trained networks are also included. The files ```data_config.json```, ```toy_config.json```, and ```eval_config.json``` fully explain the parameters for ```train_data.py```, ```train_toy.py```, and ```eval.py``` respectively.

## Executable Files

To run an experiment with ```train_data.py```, ```train_toy.py```, or ```eval.py```, just specify a name for the experiment folder and the location of the JSON config file:

```python
# directory for experiment results
EXP_DIR = './name_of/new_folder/'
# json file with experiment config
CONFIG_FILE = './path_to/config.json'
```

before execution.

## Other Files

Network structures are located in ```nets.py```. A download function for Oxford Flowers 102 data, plotting functions, and a toy dataset class can be found in ```utils.py```.

## Diagnostics

**Energy Difference and Langevin Gradient Magnitude:** Both image and toy experiments will plot $d_{s_t}$ and $r_t$ (see Section 3) over training along with correlation plots as in Figure 4 (with ACF rather than PACF).

**Landscape Plots:** Toy experiments will plot the density and log-density (negative energy) for ground-truth, learned energy, and short-run models. Kernel density estimation is used to obtain the short-run density.

**Short-Run MCMC Samples**: Image data experiments will periodically visualize the short-run MCMC samples. A batch of persistent MCMC samples will also be saved for implementations that use persistent initialization for short-run sampling.

**Long-Run MCMC Samples**: Image data experiments have the option to obtain long-run MCMC samples during training. When ```log_longrun``` is set to ```true``` in a data config file, the training implementation will generate long-run MCMC samples at a frequency determined by ```log_longrun_freq```. The appearance of long-run MCMC samples indicates whether the energy function assigns probability mass in realistic regions of the image space.

## Pre-trained Networks

A convergent pre-trained network and non-convergent pre-trained network for the Oxford Flowers 102 dataset are available in the Releases section of the repository. The config files ```eval_flowers_convergent.json``` and ```eval_flowers_convergent_mh.json``` are set up to evaluate ```flowers_convergent_net.pth```. The config file ```eval_flowers_nonconvergent.json``` is set up to evaluate ```flowers_nonconvergent_net.pth```.

## Contact

Please contact Mitch Hill (mkhill@ucla.edu) or Erik Nijkamp (enijkamp@ucla.edu) for any questions.
