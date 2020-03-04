# LARVIO
Lightweight, Accurate and Robust monocular Visual Inertial Odometry is based on hybrid EKF VIO. It is featured by augmenting features with long track length into the filter state by 1D IDP of MSCKF to provide accurate positioning results.


## Acknowledgement
This repo is for academic use only.

LARVIO is originally developed based on [MSCKF_VIO](https://github.com/KumarRobotics/msckf_vio). Tremendous changes has been made, including the interface, visualization, visual front-end and filter details. 

LARVIO also benefits from [VINS-MONO](https://github.com/HKUST-Aerial-Robotics/VINS-Mono) and [ORB_SLAM2](https://github.com/raulmur/ORB_SLAM2).

We would like to thank the authors of repo above for their great contribution. We kept the copyright announcement of these repos.


## Introduction
LARVIO is a EKF-based monocular VIO. Loop closure was not applied in our algorithm. 
#### 1) Hybrid EKF with 1D IDP
A hybrid EKF architecture is utilized, which based on the work of [Mingyang Li](http://roboticsproceedings.org/rss08/p31.pdf). It augments features with long track length into the filter state. In LARVIO, One-Dimensional Inverse Depth Parametrization is utilized to parametrize the augmented feature state, which is different from the original 3d solution by Li. This novelty improves the computational efficiency compare to the 3d solution. The positioning precision is also improved thanks to the utilization of complete constraints of features with long track length.
#### 2) Online calibration
It is capable of online imu-cam extrinsic calibration, online timestamp error calibration and online imu instrinsic calibration. 
#### 3) Automatic initialization
LARVIO can be automatically initialized in either static or dynamic scenerios.
#### 4) Robust visual front-end
We applied a ORB-descriptor assisted optical flow tracking visual front-end to improve the feature tracking performances.
#### 5) Closed-form ZUPT
A closed-form ZUPT measurement update is proposed to cope with the static scene.


## Feasibility
LARVIO is a feasible software. 

Users can change the settings in config file to set the VIO as MSCKF-only, 3d hybrid or 1d hybrid solutions. And the online calibration.
All the online calibration functions can be turned on or off in each solution by the config file.


## Dependencies
LARVIO depends on `OpenCV` (4.1.2 on OSX and 3.4.6 on Ubuntu 16.04), `Eigen`, `Boost`, `Suitesparse`, `Ceres` and `Pangolin`.

The software has been tested on OSX 10.15 and Ubuntu 16.04. It can also be ingetrated into ROS easily with some modification of the interface.


## Usage
LARVIO is a CMake based software. After install the dependencies, try commands below to compile the software:
```
cd LARVIO
mkdir build
cd build
cmake ..
make
```
An example is given in `LARVIO/run.sh` to show how to run LARVIO.


## Results
This is the results of an earlier version of LARVIO. Due to the changes, the current repo might not reproduce the exact results as below. 

RMSE on EuRoC dataset are listed. 

Evaluations below are done using [PetWorm/sim3_evaluate_tool](https://github.com/PetWorm/sim3_evaluate_tool).

In the newest update, online imu-cam extrinsic and timestamp error calibration in VINS-MONO are turned on to explore its extreme ability. While in this setup, the V102 sequence would somehow fail. The result of VINS-MONO in V102 below is of setup without online calibration.

Results of our algorithm are repeatible in every run of every computer I tested so far.

![comparison](https://github.com/PetWorm/LARVIO_test/blob/master/results/comparison.jpg)


## Related Works
A related journal paper has been initially accepted by 'Chinese Journal of Aeronautics'.

Another earlier work illustrating some parts of LARVIO is as below:
```txt
@article{qiu2019monocular,
  title={Monocular Visual-Inertial Odometry with an Unbiased Linear System Model and Robust Feature Tracking Front-End},
  author={Qiu, Xiaochen and Zhang, Hai and Fu, Wenxing and Zhao, Chenxu and Jin, Yanqiong},
  journal={Sensors},
  volume={19},
  number={8},
  pages={1941},
  year={2019},
  publisher={Multidisciplinary Digital Publishing Institute}
}
```


## Cross Platform Performance
This package has been successfully deployed on ARM (Jetson Nano and Jetson TX2, realtime without GPU refinement). The performances are comparable to the results on PCs.