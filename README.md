# PINN-elastodynamics
physics-informed neural network for solving elastodynamics (elasticity) problem

# Reference paper
This repo includes the implementation of physics-informed neural networks in paper: 

[Chengping Rao, Hao Sun and Yang Liu. Physics informed deep learning for computational elastodynamics without labeled data.](https://arxiv.org/abs/2006.08472)

# Description for each folder
- **FluentReferenceMu002**: Reference solution from Ansys Fluent for steady flow;
<!--- - **FluentReferenceUnsteady**: Reference solution from Ansys Fluent for unsteady flow; --->
- **PINN_steady**: Implementation for steady flow with PINN;
- **PINN_unsteady**: Implementation for unsteady flow with PINN;

# Results overview

![](https://github.com/Raocp/PINN-laminar-flow/blob/master/PINN_steady/uvp.png)

> Defected plate under cyclic load (top: ; bottom: )



# Note
- These implementations were developed and tested on the GPU version of TensorFlow 1.10.0. 
