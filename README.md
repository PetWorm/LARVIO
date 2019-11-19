# light-msckf
A lightweight, accurate and robust monocular visual inertial odometry based on Multi-State Constraint Kalman Filter.

To be open sourced soon.


## Introduction
Loop closure was not applied in our algorithm. It is capable of online imu-cam extrinsic calibration, online timestamp error calibration and online imu instrinsic calibration, and can be automatically initialized in either static or dynamic scenerios.


## Results
RMSE on EuRoC dataset are listed. 
Evaluations below are done using [PetWorm/sim3_evaluate_tool](https://github.com/PetWorm/sim3_evaluate_tool).

In the newest update, online imu-cam extrinsic and timestamp error calibration in VINS-MONO are turned on to explore its extreme ability. While in this setup, the V102 sequence would somehow fail. The result of VINS-MONO in V102 below is of setup without online calibration.

Results of our algorithm are repeatible in every run of every computer I tested so far.

![comparison](https://github.com/PetWorm/light-msckf/blob/master/results/comparison.jpg)


## Cross Platform Performance
This package has been successfully deployed on ARM (Jetson Nano and Jetson TX2, no GPU refinement yet) recently. The performances are comparable to the results on PCs.
