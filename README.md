# LARVIO
LARVIO is short for Lightweight, Accurate and Robust monocular Visual Inertial Odometry, which is based on hybrid EKF VIO. It is featured by augmenting features with long track length into the filter state of MSCKF by 1D IDP to provide accurate positioning results.

The core algorithm of LARVIO depends on `Eigen`, `Boost`, `Suitesparse`, `Ceres` and `OpenCV`, making the algorithm of good portability. 

A single-thread toyish example as well as a ROS nodelet package for LARVIO is provided in this repo.

Notice that Hamilton quaternion is utilized in LARVIO, which is a little bit different from the JPL quaternion used in traditional MSCKF community. The filter formulation is thus derivated from scratch. Please check our [Senors2019](https://www.mdpi.com/1424-8220/19/8/1941/htm) and CJA2020 (coming soon) papers for details.


## Results
#### 1) Demo on EuRoC
![LARVIO on EuRoC](https://github.com/PetWorm/LARVIO/blob/master/results/euroc_x8.gif)
#### 2) Trajectories RMSEs
This is the results of an earlier version of LARVIO. Due to the changes, the current repo might not reproduce the exact results as below. 

RMSE on EuRoC dataset are listed. 

Evaluations below are done using [PetWorm/sim3_evaluate_tool](https://github.com/PetWorm/sim3_evaluate_tool).

In the newest update, online imu-cam extrinsic and timestamp error calibration in VINS-MONO are turned on to explore its extreme ability. While in this setup, the V102 sequence would somehow fail. The result of VINS-MONO in V102 below is of setup without online calibration.

Results of our algorithm are repeatible in every run of every computer I tested so far.

![comparison](https://github.com/PetWorm/LARVIO/blob/master/results/comparison.jpg)


## Cross Platform Performance
This package has been successfully deployed on ARM (Jetson Nano and Jetson TX2, realtime without GPU refinement). The performances are comparable to the results on PCs.

Below is the exper1ment result in our office. A TX2-based multi-thread CPU-only implementation without ROS was developed here. We used [MYNT-EYE-D](https://github.com/slightech/MYNT-EYE-D-SDK) camera SDK to collect monocular images and IMU data, and estimate the camera poses in realtime. We walk around out the office to the corridor or the neighbor room, and return to the start point (in white circle) for a couple of times.

![TX2 implementation](https://github.com/PetWorm/LARVIO/blob/master/results/TX2_result.png)


## Acknowledgement
This repo is for academic use only.

LARVIO is originally developed based on [MSCKF_VIO](https://github.com/KumarRobotics/msckf_vio). Tremendous changes has been made, including the interface, visualization, visual front-end and filter details. 

LARVIO also benefits from [VINS-MONO](https://github.com/HKUST-Aerial-Robotics/VINS-Mono) and [ORB_SLAM2](https://github.com/raulmur/ORB_SLAM2).

We would like to thank the authors of repos above for their great contribution. We kept the copyright announcement of these repos.


## Introduction
LARVIO is an EKF-based monocular VIO. Loop closure was not applied in our algorithm. 
#### 1) Hybrid EKF with 1D IDP
A hybrid EKF architecture is utilized, which is based on the work of [Mingyang Li](http://roboticsproceedings.org/rss08/p31.pdf). It augments features with long track length into the filter state. In LARVIO, One-Dimensional Inverse Depth Parametrization is utilized to parametrize the augmented feature state, which is different from the original 3d solution by Li. This novelty improves the computational efficiency compare to the 3d solution. The positioning precision is also improved thanks to the utilization of complete constraints of features with long track length.
#### 2) Online calibration
It is capable of online imu-cam extrinsic calibration, online timestamp error calibration and online imu intrinsic calibration. 
#### 3) Automatic initialization
LARVIO can be automatically initialized in either static or dynamic scenerios.
#### 4) Robust visual front-end
We applied a ORB-descriptor assisted optical flow tracking visual front-end to improve the feature tracking performances.
#### 5) Closed-form ZUPT
A closed-form ZUPT measurement update is proposed to cope with the static scene.


## Feasibility
LARVIO is a feasible software. 

Users can change the settings in config file to set the VIO as MSCKF-only, 3d hybrid or 1d hybrid solutions. And all the online calibration functions can be turned on or off in each solution by the config file.


## Dependencies
LARVIO depends on `Eigen`, `Boost`, `Suitesparse`, `Ceres` and `OpenCV` for the core algorithm.
#### Toyish example
The toyish example depends on `OpenCV` (4.1.2 on OSX and 3.4.6 on Ubuntu 16.04/18.04), `Pangolin` is needed for visualization. Notice that extra `gcc 7` installation is needed for Ubuntu 16.04.
#### ROS nodelet
The ROS nodelet package has been tested on `Kinetic` and `Melodic` for Ubuntu 16.04/18.04. Following ROS packages are needed: `tf`, `cv_bridge`, `message_filters` and `image_transport`.


## Usage
This part show how to play LARVIO with [EuRoC dataset](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets).
#### Toyish example
The toyish LARVIO example is a `CMake` based software. After install the dependencies, try commands below to compile the software:
```
cd LARVIO
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=Release ..
make
```
An example is given in `LARVIO/run.sh` to show how to run the example.
#### ROS nodelet
A ROS nodelet package is provided in `LARVIO/ros_wrapper`. It has been tested on `Kinetic` and `Melodic`. Use commands below to compile the nodelet: 
```
cd YOUR_PATH/LARVIO/ros_wrapper
catkin_make
```
After building it, launch the LARVIO by:
```
. YOUR_PATH/LARVIO/ros_wrapper/devel/setup.bash
roslaunch larvio larvio_euroc.launch
```
Open a new terminal, and launch the `rviz` for visualization by (optional): 
```
. YOUR_PATH/LARVIO/ros_wrapper/devel/setup.bash
roslaunch larvio larvio_rviz.launch
```
Open a new terminal to play the dataset:
```
rosbag play MH_01_easy.bag
```


## Docker
A `Dockerfile` is provided in `LARVIO/docker`. After building it, you need to load dateset and modify the `run.sh` in container to run toyish example, or use 'roslaunch' to run the ROS package. Also, GUI is needed in the host to display the Pangolin and rviz view.

There is another VNC docker image which is convinent for monitoring the rviz view. Click [petworm/vnc-larvio-playground](https://hub.docker.com/r/petworm/vnc-larvio-playground) to directly pull this image, or build it from source with [PetWorm/docker-larvio-playground](https://github.com/PetWorm/docker-larvio-playground).


## Related Works
A related journal paper has been initially accepted by `Chinese Journal of Aeronautics`.

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