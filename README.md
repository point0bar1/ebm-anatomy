# Code for **On the Anatomy of MCMC-Based Maximum Likelihood Learning of Energy-Based Models**

This repository will reproduce the main results from our paper:

**On the Anatomy of MCMC-Based Maximum Likelihood Learning of Energy-Based Models**<br/>*Erik Nijkamp, Mitch Hill, Tian Han, Song-Chun Zhu, and Ying Nian Wu*<br/>https://arxiv.org/abs/1903.12370.

The files ```train_data.py``` and ```train_toy.py``` are Python 3.6 PyTorch-based implementations of Algorithm 1 for image datasets and toy 2D distributions respectively. Both files will measure and plot the diagnostic values $d_{s_t}$ and $r_t$ described in Section 3 during training. We provide an appendix ```ebm-anatomy-appendix.pdf``` that contains further practical considerations and empirical observations.

## Config Files

The folder ```config_locker``` has several JSON files that reproduce different convergent and non-convergent learning outcomes for image datasets and toy distributions. The files ```data_config.json``` and ```toy_config.json``` fully explain the parameters for ```train_data.py``` and ```train_toy.py``` respectively.

## Executable Files

To run an experiment with either ```train_data.py``` or ```train_toy.py```, just specify a name for the experiment folder and the location of the JSON config file:

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

## Contact

Please contact Mitch Hill (mkhill@ucla.edu) or Erik Nijkamp (enijkamp@ucla.edu) for any questions.
