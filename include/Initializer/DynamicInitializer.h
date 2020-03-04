//
// Created by xiaochen at 19-8-13.
// A vio initializer that utlize dynamic imu and img data to initialize.
// The core method comes from VINS-MONO (https://github.com/HKUST-Aerial-Robotics/VINS-Mono)
//

#ifndef DYNAMIC_INITIALIZER_H
#define DYNAMIC_INITIALIZER_H

#include <map>
#include <list>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/StdVector>

#include <boost/shared_ptr.hpp>

#include <larvio/feature_msg.h>

#include "Initializer/feature_manager.h"
#include "Initializer/initial_alignment.h"
#include "Initializer/initial_sfm.h"
#include "Initializer/solve_5pts.h"

#include "larvio/imu_state.h"

#include <iostream>

#include "sensors/ImuData.hpp"

using namespace std;

namespace larvio {

class DynamicInitializer
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // Constructor.
    DynamicInitializer() = delete;
    DynamicInitializer(const double& td_, const Eigen::Matrix3d& Ma_, 
        const Eigen::Matrix3d& Tg_, const Eigen::Matrix3d& As_, 
        const double& acc_n_, const double& acc_w_, const double& gyr_n_, 
        const double& gyr_w_, const Eigen::Matrix3d& R_c2b, 
        const Eigen::Vector3d& t_bc_b, const double& imu_img_timeTh_) : 
        td(td_), bInit(false), state_time(0.0), curr_time(-1), 
        first_imu(false), frame_count(0), acc_n(acc_n_), acc_w(acc_w_), 
        gyr_n(gyr_n_), gyr_w(gyr_w_), initial_timestamp(0.0),
        RIC(R_c2b), TIC(t_bc_b), imu_img_timeTh(imu_img_timeTh_),
        lower_time_bound(0.0) {

        Ma = Ma_;
        Tg = Tg_;
        As = As_;
        gyro_bias = Eigen::Vector3d::Zero();
        acc_bias = Eigen::Vector3d::Zero();
        position = Eigen::Vector3d::Zero();
        velocity = Eigen::Vector3d::Zero();
        orientation = Eigen::Vector4d(0.0, 0.0, 0.0, 1.0);

        for (int i = 0; i < WINDOW_SIZE + 1; i++)   
        {
            Rs[i].setIdentity();
            Ps[i].setZero();
            Vs[i].setZero();
            Bas[i].setZero();
            Bgs[i].setZero();
            dt_buf[i].clear();
            linear_acceleration_buf[i].clear();
            angular_velocity_buf[i].clear();
        }

        g = Eigen::Vector3d::Zero();

        // Initialize feature manager
        f_manager.clearState();
        // f_manager.init(Rs);
        f_manager.setRic(R_c2b);

        Times.resize(WINDOW_SIZE + 1);
    }

    // Destructor.
    ~DynamicInitializer(){};

    // Interface for trying to initialize.
    bool tryDynInit(const std::vector<ImuData>& imu_msg_buffer,
        MonoCameraMeasurementPtr img_msg);

    // Assign the initial state if initialized successfully.
    void assignInitialState(std::vector<ImuData>& imu_msg_buffer,
        Eigen::Vector3d& m_gyro_old, Eigen::Vector3d& m_acc_old, IMUState& imu_state);

    // If initialized.
    bool ifInitialized() {
        return bInit;
    }

private:

    // Time lower bound for used imu data.
    double lower_time_bound;

    // Threshold for deciding which imu is at the same time with a img frame
    double imu_img_timeTh;

    // Flag indicating if initialized.
    bool bInit;

    // Error bewteen timestamp of imu and img.
    double td;

    // Error between img and its nearest imu msg.
    double ddt;

    // Relative rotation between camera and imu.
    Eigen::Matrix3d RIC;
    Eigen::Vector3d TIC;

    // Imu instrinsic.
    Eigen::Matrix3d Ma;
    Eigen::Matrix3d Tg;
    Eigen::Matrix3d As;

    // Initialize results.
    double state_time;
    Eigen::Vector3d gyro_bias;
    Eigen::Vector3d acc_bias;
    Eigen::Vector4d orientation;
    Eigen::Vector3d position;
    Eigen::Vector3d velocity;

    // Save the last imu data that have been processed.
    Eigen::Vector3d last_acc;
    Eigen::Vector3d last_gyro;

    // Flag for declare the first imu data.
    bool first_imu;

    // Imu data for initialize every imu preintegration base.
    Vector3d acc_0, gyr_0;

    // Frame counter in sliding window.
    int frame_count;

    // Current imu time.
    double curr_time;

    // Imu noise param.
    double acc_n, acc_w;
    double gyr_n, gyr_w;

    // IMU preintegration between keyframes.
    boost::shared_ptr<IntegrationBase> pre_integrations[(WINDOW_SIZE + 1)];

    // Temporal buff for imu preintegration between ordinary frames.
    boost::shared_ptr<IntegrationBase> tmp_pre_integration;

    // Store the information of ordinary frames
    map<double, ImageFrame> all_image_frame;

    // Bias of gyro and accelerometer of imu corresponding to every keyframe.
    Eigen::Vector3d Bas[(WINDOW_SIZE + 1)];
    Eigen::Vector3d Bgs[(WINDOW_SIZE + 1)];

    // Every member of this vector store the dt between every adjacent imu 
    // between two keyframes in sliding window.
    vector<double> dt_buf[(WINDOW_SIZE + 1)];

    // Every member of this two vectors store all imu measurements 
    // between two keyframes in sliding window.
    vector<Eigen::Vector3d> linear_acceleration_buf[(WINDOW_SIZE + 1)];
    vector<Eigen::Vector3d> angular_velocity_buf[(WINDOW_SIZE + 1)];

    // Gravity under reference camera frame.
    Eigen::Vector3d g;

    // Feature manager.
    FeatureManager f_manager;

    // State of body frame under reference frame.
    Eigen::Matrix3d Rs[(WINDOW_SIZE + 1)];
    Eigen::Vector3d Ps[(WINDOW_SIZE + 1)];
    Eigen::Vector3d Vs[(WINDOW_SIZE + 1)];

    // Flags for marginalization.
    enum MarginalizationFlag
    {
        MARGIN_OLD = 0,
        MARGIN_SECOND_NEW = 1
    };
    MarginalizationFlag  marginalization_flag;

    // Timestamps of sliding window.
    vector<double> Times;

    // Initial timestamp
    double initial_timestamp;

    // For solving relative motion.
    MotionEstimator m_estimator;

private:

    // Process every imu frame before the img.
    void processIMU(const ImuData& imu_msg);

    // Process img frame.
    void processImage(MonoCameraMeasurementPtr img_msg);

    // Check if the condition is fit to conduct the vio initialization, and conduct it while suitable.
    bool initialStructure();

    // Try to recover relative pose.
    bool relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l);

    // Align the visual sfm with imu preintegration.
    bool visualInitialAlign();

    // Slide the window.
    void slideWindow();
};

}

#endif //DYNAMIC_INITIALIZER_H
