/*
 * COPYRIGHT AND PERMISSION NOTICE
 * Penn Software MSCKF_VIO
 * Copyright (C) 2017 The Trustees of the University of Pennsylvania
 * All rights reserved.
 */

// The original file belongs to MSCKF_VIO (https://github.com/KumarRobotics/msckf_vio/)
// Some changes have been made to use it in LARVIO


#ifndef IMU_STATE_H
#define IMU_STATE_H

#include <map>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#define GRAVITY_ACCELERATION 9.81

namespace larvio {

typedef long long int FeatureIDType;

/*
 * @brief IMUState State for IMU
 */
struct IMUState {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  typedef long long int StateIDType;

  // An unique identifier for the IMU state.
  StateIDType id;

  // id for next IMU state
  static StateIDType next_id;

  // Time when the state is recorded
  double time;

  // Time interval to the nearest image
  double dt;

  // Orientation
  // Take a quaternion from the world frame to
  // the IMU (body) frame.
  Eigen::Vector4d orientation;

  // Position of the IMU (body) frame
  // in the world frame.
  Eigen::Vector3d position;

  // Velocity of the IMU (body) frame
  // in the world frame.
  Eigen::Vector3d velocity;

  // Bias for measured angular velocity
  // and acceleration.
  Eigen::Vector3d gyro_bias;
  Eigen::Vector3d acc_bias;

  // Transformation between the IMU and the
  // left camera (cam0)
  Eigen::Matrix3d R_imu_cam0;
  Eigen::Vector3d t_cam0_imu;

  // Gravity vector in the world frame
  static Eigen::Vector3d gravity;

  // Transformation offset from the IMU frame to
  // the body frame. The transformation takes a
  // vector from the IMU frame to the body frame.
  // The z axis of the body frame should point upwards.
  // Normally, this transform should be identity.
  static Eigen::Isometry3d T_imu_body;

  IMUState(): id(0), time(0),
    orientation(Eigen::Vector4d(0, 0, 0, 1)),
    position(Eigen::Vector3d::Zero()),
    velocity(Eigen::Vector3d::Zero()),
    gyro_bias(Eigen::Vector3d::Zero()),
    acc_bias(Eigen::Vector3d::Zero()) {}

  IMUState(const StateIDType& new_id): id(new_id), time(0),
    orientation(Eigen::Vector4d(0, 0, 0, 1)),
    position(Eigen::Vector3d::Zero()),
    velocity(Eigen::Vector3d::Zero()),
    gyro_bias(Eigen::Vector3d::Zero()),
    acc_bias(Eigen::Vector3d::Zero()) {}

};

typedef IMUState::StateIDType StateIDType;

/*
 * @brief IMUState_aug Augmented state for IMU, including only orientation and position
 */
struct IMUState_Aug {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  // An unique identifier for the IMU state.
  StateIDType id;

  // Time when the state is recorded
  double time;

  // Time interval to the nearest image
  double dt;

  // Orientation
  // Take a quaternion from the world frame to
  // the IMU (body) frame.
  Eigen::Vector4d orientation;

  // Position of the IMU (body) frame
  // in the world frame.
  Eigen::Vector3d position;

  // First estimated Jacobian of position;
  Eigen::Vector3d position_FEJ;

  // Transformation between the IMU and the
  // left camera (cam0)
  Eigen::Matrix3d R_imu_cam0;
  Eigen::Vector3d t_cam0_imu;

  // Corresponding camera pose
  Eigen::Vector4d orientation_cam;
  Eigen::Vector3d position_cam;

  // Store the feature id whose anchor frame is 
  // corresponding to this imu state.
  std::vector<FeatureIDType> FeatureIDs;

  IMUState_Aug(): id(0), time(0),
    orientation(Eigen::Vector4d(0, 0, 0, 1)),
    position(Eigen::Vector3d::Zero()) {}

  IMUState_Aug(const StateIDType& new_id): id(new_id), time(0),
    orientation(Eigen::Vector4d(0, 0, 0, 1)),
    position(Eigen::Vector3d::Zero()) {}
};

// for augmented imu states, added by QXC
typedef std::map<StateIDType, IMUState_Aug, std::less<int>,
        Eigen::aligned_allocator<
        std::pair<const StateIDType, IMUState_Aug> > > IMUStateServer;

} // namespace larvio

#endif // IMU_STATE_H
