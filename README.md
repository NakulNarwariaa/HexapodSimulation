# Hexapod Simulation
---

## Abstract
---

An alternative approach is presented to the problem of real time forward kinematics calculations of the Stewart Platform with the use of inverse kinematics equations. Due to the complexity of nonlinear and polynomial equations associated with the geometry of machines, these cannot be computed at real time. Consequently, most mechanical contraptions have not been used to their full potential. To overcome this limitation, this approach suggests the use of neural networks to train according to datasets and calculate forward kinematic values at real time. 

The proposed method is applied to a hexapod, also known as the Gough- Stewart platform. To compute the forward kinematics values, inverse kinematics equations would be used to generate data for a specific range of input parameters, generated data would then be clustered into different classes/clusters using k-means, and for each cluster a neural network would be trained to compute forward kinematics at real time. Also, various changes in the parameters are introduced to increase the accuracy of the neural networks. At the end, performance for orientation and position is computed separately by using mean squared error to observe the deviation of calculated values from expected values for forward kinematics. 

Detailed description is added in the complete report of the simulation along with the conclusions. 

## Terminology
---

Any freely-suspended object placed on the platform is allowed to move in six degree of freedoms, which are the three
plausible positions along with three possible corresponding rotations. Our positions are denoted
by P(Px, Py, Pz) and the rotations by ϴ(α, β, γ).

