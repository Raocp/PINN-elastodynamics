# PINN-elastodynamics
physics-informed neural network for solving elastodynamics (elasticity) problem

# Reference paper
This repo includes the implementation of physics-informed neural networks in paper: 

[Chengping Rao, Hao Sun and Yang Liu. Physics informed deep learning for computational elastodynamics without labeled data.](https://arxiv.org/abs/2006.08472)

# Description for each folder
- **PlateHoleQuarter**: Training script and dataset for plate with a hole (stress concentration) problem;
- **ElasticWaveInfinite**: Training script and dataset for elastic wave propagation in infinite domain (to be uploaded);


# Results overview

![](https://github.com/Raocp/PINN-elastodynamics/blob/master/PlateHoleQuarter/results/GIF_stress.gif)

> Defected plate under cyclic load (top: PINN; bottom: FEM.)

![](https://github.com/Raocp/PINN-elastodynamics/blob/master/ElasticWaveInfinite/results/GIF_uv.gif)
![](https://github.com/Raocp/PINN-elastodynamics/blob/master/ElasticWaveInfinite/results/color_map_uv.png)
> Elastic wave propagation in infinite (unbounded) domain (top: PINN; bottom: FEM.)


# Note
- These implementations were developed and tested on the GPU version of TensorFlow 1.10.0. 
