/*
 * COPYRIGHT AND PERMISSION NOTICE
 * Penn Software MSCKF_VIO
 * Copyright (C) 2017 The Trustees of the University of Pennsylvania
 * All rights reserved.
 */

// The original file belongs to MSCKF_VIO (https://github.com/KumarRobotics/msckf_vio/)
// Several changes have been made to use it in LARVIO


#ifndef FEATURE_HPP
#define FEATURE_HPP

#include <iostream>
#include <map>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/StdVector>

#include "math_utils.hpp"
#include "imu_state.h"

namespace larvio {

/*
 * @brief Feature Salient part of an image. Please refer
 *    to the Appendix of "A Multi-State Constraint Kalman
 *    Filter for Vision-aided Inertial Navigation" for how
 *    the 3d position of a feature is initialized.
 */
struct Feature {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /*
   * @brief OptimizationConfig Configuration parameters
   *    for 3d feature position optimization.
   */
  struct OptimizationConfig {
    double translation_threshold;
    double huber_epsilon;
    double estimation_precision;
    double initial_damping;
    int outer_loop_max_iteration;
    int inner_loop_max_iteration;

    OptimizationConfig():
      translation_threshold(0.2),
      huber_epsilon(0.01),
      estimation_precision(5e-7),
      initial_damping(1e-3),
      outer_loop_max_iteration(10),
      inner_loop_max_iteration(10) {
      return;
    }
  };

  // Constructors for the struct.
  Feature(): id(0), position(Eigen::Vector3d::Zero()),
    is_initialized(false), best_cost(99999.9),
    id_anchor(-1), invParam(Eigen::Vector3d::Zero()), 
    in_state(false), totalObsNum(0), ekf_feature(false) {}

  Feature(const FeatureIDType& new_id): id(new_id),
    position(Eigen::Vector3d::Zero()),
    is_initialized(false), best_cost(99999.9),
    id_anchor(-1), invParam(Eigen::Vector3d::Zero()), 
    in_state(false), totalObsNum(0), ekf_feature(false) {}

  /*
   * @brief cost Compute the cost of the camera observations
   * @param T_c0_c1 A rigid body transformation takes
   *    a vector in c0 frame to ci frame.
   * @param x The current estimation.
   * @param z The ith measurement of the feature j in ci frame.
   * @return e The cost of this observation.
   */
  inline void cost(const Eigen::Isometry3d& T_c0_ci,
      const Eigen::Vector3d& x, const Eigen::Vector2d& z,
      double& e) const;

  /*
   * @brief jacobian Compute the Jacobian of the camera observation
   * @param T_c0_c1 A rigid body transformation takes
   *    a vector in c0 frame to ci frame.
   * @param x The current estimation.
   * @param z The actual measurement of the feature in ci frame.
   * @return J The computed Jacobian.
   * @return r The computed residual.
   * @return w Weight induced by huber kernel.
   */
  inline void jacobian(const Eigen::Isometry3d& T_c0_ci,
      const Eigen::Vector3d& x, const Eigen::Vector2d& z,
      Eigen::Matrix<double, 2, 3>& J, Eigen::Vector2d& r,
      double& w) const;

  /*
   * @brief generateInitialGuess Compute the initial guess of
   *    the feature's 3d position using only two views.
   * @param T_c1_c2: A rigid body transformation taking
   *    a vector from c2 frame to c1 frame.
   * @param z1: feature observation in c1 frame.
   * @param z2: feature observation in c2 frame.
   * @return p: Computed feature position in c1 frame.
   */
  inline void generateInitialGuess(
      const Eigen::Isometry3d& T_c1_c2, const Eigen::Vector2d& z1,
      const Eigen::Vector2d& z2, Eigen::Vector3d& p) const;

  /*
   * @brief checkMotion Check the input camera poses to ensure
   *    there is enough translation to triangulate the feature
   *    positon.
   * @param imu_states : input to aquire camera poses.
   × @param if_tracked : if feature be tracked now.
   * @return True if the translation between the input camera
   *    poses is sufficient.
   */
  inline bool checkMotion(
      const IMUStateServer& imu_states, bool if_tracked) const;

  /*
   * @brief InitializePosition Intialize the feature position
   *    based on all current available measurements.
   * @param imu_states: A map containing the camera poses with its
   *    ID as the associated key value.
   * @param curr_id: current camera id.
   * @return The computed 3d position is used to set the position
   *    member variable. Note the resulted position is in world
   *    frame.
   * @return True if the estimated 3d position of the feature
   *    is valid.
   */
  inline bool initializePosition(
      const IMUStateServer& imu_states, const StateIDType& curr_id);

  /*
   * @brief initializePosition_AssignAnchor Intialize the feature position
   *    based on all current available measurements. Current camera would
   *    be the anchor
   * @param imu_states: A map containing the camera poses with its
   *    ID as the associated key value.
   * @param anchor_id: Assigned anchor camera id.
   * @return The computed 3d position is used to set the position
   *    member variable. Note the resulted position is in world
   *    frame.
   * @return True if the estimated 3d position of the feature
   *    is valid.
   */
  inline bool initializePosition_AssignAnchor(
      const IMUStateServer& imu_states);

  /*
   * @brief initializeInvParamPosition Intialize the feature position
   *    based on all current available measurements. Result in 
   *    inverse depth parameterization in anchor frame.
   * @param imu_states: A map containing the camera poses with its
   *    ID as the associated key value.
   * @param curr_id: current camera id.
   * @return The computed 3d position is used to set the position
   *    member variable. Note the resulted position is in world
   *    frame.
   * @return The computed inverse depth parameterizations is used to 
   *    set the invParam variable. Note the resulted inverse depth
   *    parameterization is in the anchor frame, which is the newest 
   *    frame observing this feature except for current frame.
   * @return True if the estimated 3d position of the feature
   *    is valid.
   */
  inline bool initializeInvParamPosition(
      const IMUStateServer& imu_states, const StateIDType& curr_id);


  // An unique identifier for the feature.
  // In case of long time running, the variable
  // type of id is set to FeatureIDType in order
  // to avoid duplication.
  FeatureIDType id;

  // id for next feature
  static FeatureIDType next_id;

  // Store the observations of the features in the
  // state_id(key)-image_coordinates(value) manner.
  std::map<StateIDType, Eigen::Vector2d, std::less<StateIDType>,
    Eigen::aligned_allocator<
      std::pair<const StateIDType, Eigen::Vector2d> > > observations;

  // Store the observations of the features velocity in the
  // state_id(key)-features_velocity(value) manner.
  std::map<StateIDType, Eigen::Vector2d, std::less<StateIDType>,
    Eigen::aligned_allocator<
      std::pair<const StateIDType, Eigen::Vector2d> > > observations_vel;

  // 3d postion of the feature in the world frame.
  Eigen::Vector3d position;

  // First estimate 3d position in world frame
  Eigen::Vector3d position_FEJ;

  // Best normalized cost, added by QXC
  double best_cost;

  // First observed camera pose, added by QXC
  Eigen::Isometry3d firstCamPose;

  // If failed because of nagtive depth of big reprojection error
  bool failed_by_neg_dpth;
  bool failed_by_big_proj;

  // [x/z, y/z] under first observed camera coordinate, added by QXC
  Eigen::Vector2d solutionInFirstCam; 

  // A indicator to show if the 3d postion of the feature
  // has been initialized or not.
  bool is_initialized;

  // Optimization configuration for solving the 3d position.
  static OptimizationConfig optimization_config;

  // QXC debug log
  inline double getMaxObsDiff();
  inline double getTotalObsChange();

  // Anchor camera id
  StateIDType id_anchor;

  // 3d inverse depth parameter position in anchor camera frame.
  Eigen::Vector3d invParam;

  // 1d inverse depth parameter position in anchor camera frame.
  double invDepth;
  // Corrected observation in anchor camera frame.
  Eigen::Vector3d obs_anchor;

  // If this feature is in filter state
  bool in_state;

  // Record total observation number
  int totalObsNum;

  // If this is a potential ekf feature
  bool ekf_feature;
};

typedef std::map<FeatureIDType, Feature, std::less<int>,
        Eigen::aligned_allocator<
        std::pair<const FeatureIDType, Feature> > > MapServer;

void Feature::cost(const Eigen::Isometry3d& T_c0_ci,
    const Eigen::Vector3d& x, const Eigen::Vector2d& z,
    double& e) const {
  // Compute hi1, hi2, and hi3 as Equation (37).
  const double& alpha = x(0);
  const double& beta = x(1);
  const double& rho = x(2);

  Eigen::Vector3d h = T_c0_ci.linear()*
    Eigen::Vector3d(alpha, beta, 1.0) + rho*T_c0_ci.translation();
  double& h1 = h(0);
  double& h2 = h(1);
  double& h3 = h(2);

  // Predict the feature observation in ci frame.
  Eigen::Vector2d z_hat(h1/h3, h2/h3);

  // Compute the residual.
  e = (z_hat-z).squaredNorm();
  return;
}

void Feature::jacobian(const Eigen::Isometry3d& T_c0_ci,
    const Eigen::Vector3d& x, const Eigen::Vector2d& z,
    Eigen::Matrix<double, 2, 3>& J, Eigen::Vector2d& r,
    double& w) const {

  // Compute hi1, hi2, and hi3 as Equation (37). 
  const double& alpha = x(0);
  const double& beta = x(1);
  const double& rho = x(2);

  Eigen::Vector3d h = T_c0_ci.linear()*
    Eigen::Vector3d(alpha, beta, 1.0) + rho*T_c0_ci.translation();
  double& h1 = h(0);
  double& h2 = h(1);
  double& h3 = h(2);

  // Compute the Jacobian.
  Eigen::Matrix3d W;
  W.leftCols<2>() = T_c0_ci.linear().leftCols<2>();
  W.rightCols<1>() = T_c0_ci.translation();

  J.row(0) = 1/h3*W.row(0) - h1/(h3*h3)*W.row(2);
  J.row(1) = 1/h3*W.row(1) - h2/(h3*h3)*W.row(2);

  // Compute the residual.
  Eigen::Vector2d z_hat(h1/h3, h2/h3);
  r = z_hat - z;

  // Compute the weight based on the residual.
  double e = r.norm();
  if (e <= optimization_config.huber_epsilon)
    w = 1.0;
  else
    w = std::sqrt(2.0*optimization_config.huber_epsilon / e);

  return;
}

void Feature::generateInitialGuess(
    const Eigen::Isometry3d& T_c1_c2, const Eigen::Vector2d& z1,
    const Eigen::Vector2d& z2, Eigen::Vector3d& p) const {
  // Construct a least square problem to solve the depth.
  Eigen::Vector3d m = T_c1_c2.linear() * Eigen::Vector3d(z1(0), z1(1), 1.0);

  Eigen::Vector2d A(0.0, 0.0);
  A(0) = m(0) - z2(0)*m(2);
  A(1) = m(1) - z2(1)*m(2);

  Eigen::Vector2d b(0.0, 0.0);
  b(0) = z2(0)*T_c1_c2.translation()(2) - T_c1_c2.translation()(0);
  b(1) = z2(1)*T_c1_c2.translation()(2) - T_c1_c2.translation()(1);

  // Solve for the depth.
  double depth = (A.transpose() * A).inverse() * A.transpose() * b;
  p(0) = z1(0) * depth;
  p(1) = z1(1) * depth;
  p(2) = depth;
  return;
}

bool Feature::checkMotion(
    const IMUStateServer& imu_states, bool if_tracked) const {

  StateIDType first_cam_id = observations.begin()->first;
  StateIDType last_cam_id;
  if (if_tracked)
    last_cam_id = (--(--observations.end()))->first;
  else
    last_cam_id = (--observations.end())->first;

  Eigen::Isometry3d first_cam_pose;      
  const Eigen::Vector4d& first_cam_orientation = imu_states.find(first_cam_id)->second.orientation_cam;
  first_cam_pose.linear() = Eigen::Quaterniond(
      first_cam_orientation(3),first_cam_orientation(0),first_cam_orientation(1),first_cam_orientation(2)).toRotationMatrix();
  first_cam_pose.translation() =
      imu_states.find(first_cam_id)->second.position_cam;

  Eigen::Isometry3d last_cam_pose;       
  const Eigen::Vector4d& last_cam_orientation = imu_states.find(last_cam_id)->second.orientation_cam;
  last_cam_pose.linear() = Eigen::Quaterniond(
      last_cam_orientation(3),last_cam_orientation(0),last_cam_orientation(1),last_cam_orientation(2)).toRotationMatrix();
  last_cam_pose.translation() =
      imu_states.find(last_cam_id)->second.position_cam;

  // Get the direction of the feature when it is first observed.
  // This direction is represented in the world frame.
  Eigen::Vector3d feature_direction(
      observations.begin()->second(0),
      observations.begin()->second(1), 1.0);
  feature_direction = feature_direction / feature_direction.norm();
  feature_direction = first_cam_pose.linear()*feature_direction;       

  // Compute the translation between the first frame
  // and the last frame. We assume the first frame and
  // the last frame will provide the largest motion to
  // speed up the checking process.
  Eigen::Vector3d translation = last_cam_pose.translation() -
    first_cam_pose.translation();
  double parallel_translation =
    translation.transpose()*feature_direction;    
  Eigen::Vector3d orthogonal_translation = translation -
    parallel_translation*feature_direction;       

  if (orthogonal_translation.norm() >
      optimization_config.translation_threshold)  
    return true;
  else return false;
}

bool Feature::initializePosition(
    const IMUStateServer& imu_states, const StateIDType& curr_id) {
  // Organize camera poses and feature observations properly.
  std::vector<Eigen::Isometry3d,
    Eigen::aligned_allocator<Eigen::Isometry3d> > cam_poses(0); 
  std::vector<Eigen::Vector2d,
    Eigen::aligned_allocator<Eigen::Vector2d> > measurements(0);

  std::vector<StateIDType> cam_ids(0);
  for (auto& m : observations) {                                    
    // TODO: This should be handled properly. Normally, the
    //    required camera states should all be available in
    //    the input imu_states buffer.
    auto state_iter = imu_states.find(m.first);
    if (state_iter == imu_states.end()) continue;

    if (curr_id == state_iter->first) continue;

    // Add the measurement.
    measurements.push_back(m.second.head<2>());

    // This camera pose will take a vector from this camera frame
    // to the world frame.
    Eigen::Isometry3d cam_pose;
    const Eigen::Vector4d& cam_qua = state_iter->second.orientation_cam;
    cam_pose.linear() = Eigen::Quaterniond(
        cam_qua(3),cam_qua(0),cam_qua(1),cam_qua(2)).toRotationMatrix();
    cam_pose.translation() = state_iter->second.position_cam;

    cam_poses.push_back(cam_pose);

    cam_ids.push_back(state_iter->first);
  }

  // All camera poses should be modified such that it takes a
  // vector from the last camera frame in the buffer to this
  // camera frame.
  Eigen::Isometry3d T_c_w_last = cam_poses[cam_poses.size()-1];
  for (auto& pose : cam_poses)
    pose = pose.inverse() * T_c_w_last;     

  // Generate initial guess
  Eigen::Vector3d initial_position(0.0, 0.0, 0.0);
  if (!is_initialized)
    generateInitialGuess(cam_poses[0], measurements[cam_poses.size()-1],
        measurements[0], initial_position);       
  else
    initial_position = T_c_w_last.inverse()*position;
  Eigen::Vector3d solution(
      initial_position(0)/initial_position(2),
      initial_position(1)/initial_position(2),
      1.0/initial_position(2));     

  // Apply Levenberg-Marquart method to solve for the 3d position.
  double lambda = optimization_config.initial_damping;
  int inner_loop_cntr = 0;
  int outer_loop_cntr = 0;
  bool is_cost_reduced = false;
  double delta_norm = 0;

  // Compute the initial cost.      
  double total_cost = 0.0;
  for (int i = 0; i < cam_poses.size(); ++i) {
    double this_cost = 0.0;
    cost(cam_poses[i], solution, measurements[i], this_cost);
    total_cost += this_cost;
  }

  // Outer loop.
  do {
    Eigen::Matrix3d A = Eigen::Matrix3d::Zero();
    Eigen::Vector3d b = Eigen::Vector3d::Zero();

    for (int i = 0; i < cam_poses.size(); ++i) {
      Eigen::Matrix<double, 2, 3> J;
      Eigen::Vector2d r;
      double w;

      jacobian(cam_poses[i], solution, measurements[i], J, r, w);

      if (w == 1) {
        A += J.transpose() * J;
        b += J.transpose() * r;
      } else {
        double w_square = w * w;
        A += w_square * J.transpose() * J;
        b += w_square * J.transpose() * r;
      }
    }

    // Inner loop.
    // Solve for the delta that can reduce the total cost.
    do {
      Eigen::Matrix3d damper = lambda * Eigen::Matrix3d::Identity();
      Eigen::Vector3d delta = (A+damper).ldlt().solve(b);  
      Eigen::Vector3d new_solution = solution - delta;
      delta_norm = delta.norm();

      double new_cost = 0.0;
      for (int i = 0; i < cam_poses.size(); ++i) {
        double this_cost = 0.0;
        cost(cam_poses[i], new_solution, measurements[i], this_cost);
        new_cost += this_cost;
      }

      if (new_cost < total_cost) {
        is_cost_reduced = true;
        solution = new_solution;
        total_cost = new_cost;
        lambda = lambda/10 > 1e-10 ? lambda/10 : 1e-10;     
      } else {
        is_cost_reduced = false;
        lambda = lambda*10 < 1e12 ? lambda*10 : 1e12;  
      }

    } while (inner_loop_cntr++ <
        optimization_config.inner_loop_max_iteration && !is_cost_reduced);

    inner_loop_cntr = 0;

  } while (outer_loop_cntr++ <
      optimization_config.outer_loop_max_iteration &&
      delta_norm > optimization_config.estimation_precision);

  // Covert the feature position from inverse depth
  // representation to its 3d coordinate.
  Eigen::Vector3d final_position(solution(0)/solution(2),
      solution(1)/solution(2), 1.0/solution(2));

  // Check if the solution is valid. Make sure the feature
  // is in front of every camera frame observing it.
  bool is_valid_solution = true;
  failed_by_neg_dpth = false;
  for (const auto& pose : cam_poses) {
    Eigen::Vector3d pos =
      pose.linear()*final_position + pose.translation();
    if (pos(2) <= 0) {
      is_valid_solution = false;
      failed_by_neg_dpth = true;
      break;
    }
  }

  // added by QXC: check total_cost
  double normalized_cost =
          total_cost / (2 * cam_poses.size() * cam_poses.size());
  double uv_cost =
          sqrt(total_cost/cam_poses.size());
  failed_by_big_proj = false;
  if (normalized_cost > 4.7673e-04) { 
    is_valid_solution = false;
    failed_by_big_proj = true;
  }

  if (is_valid_solution) {
    if (!is_initialized)
      position_FEJ = position;
    is_initialized = true;
    position = T_c_w_last.linear()*final_position + T_c_w_last.translation();
    invParam = solution;
    id_anchor = cam_ids[cam_ids.size()-1];
    invDepth = 1/final_position(2);
    obs_anchor = Eigen::Vector3d(final_position(0)*invDepth,     // correct observation
        final_position(1)*invDepth, 1);
    // obs_anchor = Eigen::Vector3d(measurements[cam_ids.size()-1](0),     // do not correct observation
    //     measurements[cam_ids.size()-1](1), 1);
  }

  return is_valid_solution;
}

bool Feature::initializePosition_AssignAnchor(
    const IMUStateServer& imu_states) {
  // Organize camera poses and feature observations properly.
  std::vector<Eigen::Isometry3d,
    Eigen::aligned_allocator<Eigen::Isometry3d> > cam_poses(0);     
  std::vector<Eigen::Vector2d,
    Eigen::aligned_allocator<Eigen::Vector2d> > measurements(0);

  std::vector<StateIDType> cam_ids(0);
  for (auto& m : observations) {                                    
    // TODO: This should be handled properly. Normally, the
    //    required camera states should all be available in
    //    the input imu_states buffer.
    auto state_iter = imu_states.find(m.first);
    if (state_iter == imu_states.end()) continue;

    // Add the measurement.
    measurements.push_back(m.second.head<2>());

    // This camera pose will take a vector from this camera frame
    // to the world frame.
    Eigen::Isometry3d cam_pose;
    const Eigen::Vector4d& cam_qua = state_iter->second.orientation_cam;
    cam_pose.linear() = Eigen::Quaterniond(
        cam_qua(3),cam_qua(0),cam_qua(1),cam_qua(2)).toRotationMatrix();
    cam_pose.translation() = state_iter->second.position_cam;

    cam_poses.push_back(cam_pose);

    cam_ids.push_back(state_iter->first);
  }

  // All camera poses should be modified such that it takes a
  // vector from the last camera frame in the buffer to this
  // camera frame.
  Eigen::Isometry3d T_c_w_last = cam_poses[cam_poses.size()-1];
  for (auto& pose : cam_poses)
    pose = pose.inverse() * T_c_w_last;    

  // Generate initial guess
  Eigen::Vector3d initial_position(0.0, 0.0, 0.0);
  if (!is_initialized)
    generateInitialGuess(cam_poses[0], measurements[cam_poses.size()-1],
        measurements[0], initial_position);       
  else
    initial_position = T_c_w_last.inverse()*position;
  Eigen::Vector3d solution(
      initial_position(0)/initial_position(2),
      initial_position(1)/initial_position(2),
      1.0/initial_position(2));     

  // Apply Levenberg-Marquart method to solve for the 3d position.
  double lambda = optimization_config.initial_damping;
  int inner_loop_cntr = 0;
  int outer_loop_cntr = 0;
  bool is_cost_reduced = false;
  double delta_norm = 0;

  // Compute the initial cost.      
  double total_cost = 0.0;
  for (int i = 0; i < cam_poses.size(); ++i) {
    double this_cost = 0.0;
    cost(cam_poses[i], solution, measurements[i], this_cost);
    total_cost += this_cost;
  }

  // Outer loop.
  do {
    Eigen::Matrix3d A = Eigen::Matrix3d::Zero();
    Eigen::Vector3d b = Eigen::Vector3d::Zero();

    for (int i = 0; i < cam_poses.size(); ++i) {
      Eigen::Matrix<double, 2, 3> J;
      Eigen::Vector2d r;
      double w;

      jacobian(cam_poses[i], solution, measurements[i], J, r, w);  

      if (w == 1) {
        A += J.transpose() * J;    
        b += J.transpose() * r;
      } else {
        double w_square = w * w;
        A += w_square * J.transpose() * J;
        b += w_square * J.transpose() * r;
      }
    }

    // Inner loop.
    // Solve for the delta that can reduce the total cost.
    do {
      Eigen::Matrix3d damper = lambda * Eigen::Matrix3d::Identity();
      Eigen::Vector3d delta = (A+damper).ldlt().solve(b);      
      Eigen::Vector3d new_solution = solution - delta;
      delta_norm = delta.norm();

      double new_cost = 0.0;
      for (int i = 0; i < cam_poses.size(); ++i) {
        double this_cost = 0.0;
        cost(cam_poses[i], new_solution, measurements[i], this_cost);
        new_cost += this_cost;
      }

      if (new_cost < total_cost) {
        is_cost_reduced = true;
        solution = new_solution;
        total_cost = new_cost;
        lambda = lambda/10 > 1e-10 ? lambda/10 : 1e-10;     
      } else {
        is_cost_reduced = false;
        lambda = lambda*10 < 1e12 ? lambda*10 : 1e12;      
      }

    } while (inner_loop_cntr++ <
        optimization_config.inner_loop_max_iteration && !is_cost_reduced);

    inner_loop_cntr = 0;

  } while (outer_loop_cntr++ <
      optimization_config.outer_loop_max_iteration &&
      delta_norm > optimization_config.estimation_precision);

  // Covert the feature position from inverse depth
  // representation to its 3d coordinate.
  Eigen::Vector3d final_position(solution(0)/solution(2),
      solution(1)/solution(2), 1.0/solution(2));

  // Check if the solution is valid. Make sure the feature
  // is in front of every camera frame observing it.
  bool is_valid_solution = true;
  failed_by_neg_dpth = false;
  for (const auto& pose : cam_poses) {
    Eigen::Vector3d pos =
      pose.linear()*final_position + pose.translation();
    if (pos(2) <= 0) {
      is_valid_solution = false;
      failed_by_neg_dpth = true;
      break;
    }
  }

  // added by QXC: check total_cost
  double normalized_cost =
          total_cost / (2 * cam_poses.size() * cam_poses.size());
  double uv_cost =
          sqrt(total_cost/cam_poses.size());
  failed_by_big_proj = false;
  if (normalized_cost > 4.7673e-04) {  
    is_valid_solution = false;
    failed_by_big_proj = true;
  }

  if (is_valid_solution) {
    if (!is_initialized)
      position_FEJ = position;
    is_initialized = true;
    position = T_c_w_last.linear()*final_position + T_c_w_last.translation();
    invParam = solution;
    id_anchor = cam_ids[cam_ids.size()-1];
    invDepth = 1/final_position(2);
    obs_anchor = Eigen::Vector3d(final_position(0)*invDepth,     // correct observation
        final_position(1)*invDepth, 1);
    // obs_anchor = Eigen::Vector3d(measurements[cam_ids.size()-1](0),     // do not correct observation
    //     measurements[cam_ids.size()-1](1), 1);
  }

  return is_valid_solution;
}

bool Feature::initializeInvParamPosition(
    const IMUStateServer& imu_states, const StateIDType& curr_id) {
  // Organize camera poses and feature observations properly.
  std::vector<Eigen::Isometry3d,
    Eigen::aligned_allocator<Eigen::Isometry3d> > cam_poses(0);  
  std::vector<Eigen::Vector2d,
    Eigen::aligned_allocator<Eigen::Vector2d> > measurements(0);

  std::vector<StateIDType> cam_ids(0);
  for (auto& m : observations) {                                
    // TODO: This should be handled properly. Normally, the
    //    required camera states should all be available in
    //    the input imu_states buffer.
    auto state_iter = imu_states.find(m.first);
    if (state_iter == imu_states.end()) continue;

    if (curr_id == state_iter->first) continue;

    // Add the measurement.
    measurements.push_back(m.second.head<2>());

    // This camera pose will take a vector from this camera frame
    // to the world frame.
    Eigen::Isometry3d cam_pose;
    const Eigen::Vector4d& cam_qua = state_iter->second.orientation_cam;
    cam_pose.linear() = Eigen::Quaterniond(
        cam_qua(3),cam_qua(0),cam_qua(1),cam_qua(2)).toRotationMatrix();
    cam_pose.translation() = state_iter->second.position_cam;

    cam_poses.push_back(cam_pose);

    cam_ids.push_back(state_iter->first);
  }

  // All camera poses should be modified such that it takes a
  // vector from the last camera frame in the buffer to this
  // camera frame.
  Eigen::Isometry3d T_c_w_last = cam_poses[cam_poses.size()-1];
  for (auto& pose : cam_poses)
    pose = pose.inverse() * T_c_w_last;    

  // Generate initial guess
  Eigen::Vector3d initial_position(0.0, 0.0, 0.0);
  generateInitialGuess(cam_poses[0], measurements[measurements.size()-1],
      measurements[0], initial_position);     
  Eigen::Vector3d solution(
      initial_position(0)/initial_position(2),
      initial_position(1)/initial_position(2),
      1.0/initial_position(2));    

  // Apply Levenberg-Marquart method to solve for the 3d position.
  double lambda = optimization_config.initial_damping;
  int inner_loop_cntr = 0;
  int outer_loop_cntr = 0;
  bool is_cost_reduced = false;
  double delta_norm = 0;

  // Compute the initial cost.     
  double total_cost = 0.0;
  for (int i = 0; i < cam_poses.size(); ++i) {
    double this_cost = 0.0;
    cost(cam_poses[i], solution, measurements[i], this_cost);
    total_cost += this_cost;
  }

  // Outer loop.
  do {
    Eigen::Matrix3d A = Eigen::Matrix3d::Zero();
    Eigen::Vector3d b = Eigen::Vector3d::Zero();

    for (int i = 0; i < cam_poses.size(); ++i) {
      Eigen::Matrix<double, 2, 3> J;
      Eigen::Vector2d r;
      double w;

      jacobian(cam_poses[i], solution, measurements[i], J, r, w);  

      if (w == 1) {
        A += J.transpose() * J;   
        b += J.transpose() * r;
      } else {
        double w_square = w * w;
        A += w_square * J.transpose() * J;
        b += w_square * J.transpose() * r;
      }
    }

    // Inner loop.
    // Solve for the delta that can reduce the total cost.
    do {
      Eigen::Matrix3d damper = lambda * Eigen::Matrix3d::Identity();
      Eigen::Vector3d delta = (A+damper).ldlt().solve(b);      
      Eigen::Vector3d new_solution = solution - delta;
      delta_norm = delta.norm();

      double new_cost = 0.0;
      for (int i = 0; i < cam_poses.size(); ++i) {
        double this_cost = 0.0;
        cost(cam_poses[i], new_solution, measurements[i], this_cost);
        new_cost += this_cost;
      }

      if (new_cost < total_cost) {
        is_cost_reduced = true;
        solution = new_solution;
        total_cost = new_cost;
        lambda = lambda/10 > 1e-10 ? lambda/10 : 1e-10;   
      } else {
        is_cost_reduced = false;
        lambda = lambda*10 < 1e12 ? lambda*10 : 1e12;      
      }

    } while (inner_loop_cntr++ <
        optimization_config.inner_loop_max_iteration && !is_cost_reduced);

    inner_loop_cntr = 0;

  } while (outer_loop_cntr++ <
      optimization_config.outer_loop_max_iteration &&
      delta_norm > optimization_config.estimation_precision);

  // Covert the feature position from inverse depth
  // representation to its 3d coordinate.
  Eigen::Vector3d final_position(solution(0)/solution(2),
      solution(1)/solution(2), 1.0/solution(2));

  // Check if the solution is valid. Make sure the feature
  // is in front of every camera frame observing it.
  bool is_valid_solution = true;
  failed_by_neg_dpth = false;
  for (const auto& pose : cam_poses) {
    Eigen::Vector3d pos =
      pose.linear()*final_position + pose.translation();
    if (pos(2) <= 0) {
      is_valid_solution = false;
      failed_by_neg_dpth = true;
      break;
    }
  }

  // added by QXC: check total_cost
  double normalized_cost =
          total_cost / (2 * cam_poses.size() * cam_poses.size());
  double uv_cost =
          sqrt(total_cost/cam_poses.size());
  failed_by_big_proj = false;
  if (normalized_cost > 4.7673e-04) {  
    is_valid_solution = false;
    failed_by_big_proj = true;
  }

  if (is_valid_solution) {
    if (!is_initialized)
      position_FEJ = position;
    ekf_feature = true;
    is_initialized = true;
    position = T_c_w_last.linear()*final_position + T_c_w_last.translation();
    invParam = solution;
    id_anchor = cam_ids[cam_ids.size()-1];
    invDepth = 1/final_position(2);
    obs_anchor = Eigen::Vector3d(final_position(0)*invDepth,     // correct observation
        final_position(1)*invDepth, 1);
    // obs_anchor = Eigen::Vector3d(measurements[cam_ids.size()-1](0),     // do not correct observation
    //     measurements[cam_ids.size()-1](1), 1);
  }

  return is_valid_solution;
}


double Feature::getMaxObsDiff() {
  Eigen::Vector2d p0(observations.begin()->second(0),
                      observations.begin()->second(1));
  Eigen::Vector2d p1((--observations.end())->second(0),
                      (--observations.end())->second(1));

  return (p0-p1).norm();
}


double Feature::getTotalObsChange() {
  
  std::vector<Eigen::Vector2d,
              Eigen::aligned_allocator<Eigen::Vector2d>> p(0);
  for (const auto& obs : observations)
    p.push_back(Eigen::Vector2d(obs.second(0),obs.second(1)));
  
  double diff = 0.0;
  for (int i = 1; i < p.size(); i++)
  {
    diff += (p[i]-p[i-1]).norm();
  }
  return diff;
}

} // namespace larvio

#endif // FEATURE_HPP
