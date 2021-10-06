# SMLM Package for Voronoi density analysis of SMLM data

## System Requirements
- The package is developed on Ubuntu 20.04 LTS Kernel 5.13.0-7614-generic
- A working Python 3 installation is required
- A conda environment is recommended
- Dependencies of the package can be found in the `requirements.txt` file

## Installation Guide for Linux
1. open a Terminal 
2. create a conda environment
   - `conda create -n voronoi_analysis python=3.9`
3. activate the environment
   - `conda activate voronoi_analysis`
4. install the package via pip
   - `pip install git+https://github.com/npeschke/smlm`

The typical install time is less than 10 min on a "normal" desktop computer

## Demo
There is a demo repository with some sample data provided at
https://github.com/npeschke/smlm_demo

1. clone the repository to your machine
   - `git clone https://github.com/npeschke/smlm_demo.git`
   1. open the repo
   - `cd smlm_demo`
2. run the demo
   - `python demo.py`

This will read the localization files from the `data/orte` directory,
run the analysis and plot figures similar to the ones in Figure 5
in the paper that are then located in the `results` directory.
The demo should run in under 10 minutes on a "normal" desktop
computer.

## Instructions for use
The demo repository also contains the notebooks used to create
the plots in the paper. Due to filesize restrictions in github the
data needed to reproduce them is not included but available upon
reasonable request.
