# PINN-elastodynamics
physics-informed neural network for solving elastodynamics (elasticity) problem

# Reference paper
This repo includes the implementation of physics-informed neural networks in paper: 

[Chengping Rao, Hao Sun and Yang Liu. Physics informed deep learning for computational elastodynamics without labeled data.](https://arxiv.org/abs/2006.08472)

# Description for each folder
- **PlateHoleQuarter**: Training script and dataset for plate with a hole (stress concentration) problem in Sec 3.1;
- **ElasticWaveInfinite**: Training script and dataset for elastic wave propagation in infinite domain in Sec 3.2; 
- **ElasticWaveConfined**: Training script and dataset for elastic wave propagation in confined (4 edges fixed) domain in Sec 3.2. 
- **ElasticWaveSemiInfinite**: Training script and dataset for elastic wave propagation in semi-infinite (top is traction-free) domain in Sec 3.2. 


# Results overview

<!-- ![](https://github.com/Raocp/PINN-elastodynamics/blob/master/PlateHoleQuarter/results/GIF_stress.gif) -->
<img src="https://github.com/Raocp/PINN-elastodynamics/blob/master/PlateHoleQuarter/results/GIF_stress.gif" width="500" />

> Defected plate under cyclic load (top: PINN; bottom: FEM.)


<!--![](https://github.com/Raocp/PINN-elastodynamics/blob/master/ElasticWaveInfinite/results/GIF_uv.gif) -->
<img src="https://github.com/Raocp/PINN-elastodynamics/blob/master/ElasticWaveInfinite/results/GIF_uv.gif" width="500" />
<!-- <img src="https://github.com/Raocp/PINN-elastodynamics/blob/master/ElasticWaveInfinite/results/color_map_uv.png" width="200" class="center"> -->

> Elastic wave propagation in infinite (unbounded) domain (top: PINN; bottom: FEM.)

<img src="https://github.com/Raocp/PINN-elastodynamics/blob/master/ElasticWaveConfined/GIF_uv.gif" width="500" />

> Elastic wave propagation in confined domain (top: PINN; bottom: FEM.)

<img src="https://github.com/Raocp/PINN-elastodynamics/blob/master/ElasticWaveSemiInfinite/Gif_uv.gif" width="500" />

> Elastic wave propagation in semi-infinite (half-bounded) domain (top: PINN; bottom: FEM.)


# Note
- These implementations were developed and tested on the GPU version of TensorFlow 1.10.0. 
