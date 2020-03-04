//
// Created by xiaochen at 19-8-13.
// A inclinometer-initializer utilizing the static scene.
//

#ifndef STATIC_INITIALIZER_H
#define STATIC_INITIALIZER_H

#include <map>
#include <list>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/StdVector>

#include <larvio/feature_msg.h>

#include "larvio/imu_state.h"

#include "sensors/ImuData.hpp"

using namespace std;

namespace larvio {

class StaticInitializer
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // Constructor
    StaticInitializer() = delete;
    StaticInitializer(const double& max_feature_dis_, const int& static_Num_, const double& td_, 
        const Eigen::Matrix3d& Ma_, const Eigen::Matrix3d& Tg_, const Eigen::Matrix3d& As_) : 
        max_feature_dis(max_feature_dis_), static_Num(static_Num_), td(td_), 
        bInit(false), state_time(0.0), lower_time_bound(0.0) {
        staticImgCounter = 0;
        init_features.clear();
        Ma = Ma_;
        Tg = Tg_;
        As = As_;
        gyro_bias = Eigen::Vector3d::Zero();
        acc_bias = Eigen::Vector3d::Zero();
        position = Eigen::Vector3d::Zero();
        velocity = Eigen::Vector3d::Zero();
        orientation = Eigen::Vector4d(0.0, 0.0, 0.0, 1.0);
    }

    // Destructor
    ~StaticInitializer(){}

    // Interface for trying to initialize
    bool tryIncInit(const std::vector<ImuData>& imu_msg_buffer,
        MonoCameraMeasurementPtr img_msg);

    // Assign the initial state if initialized successfully
    void assignInitialState(std::vector<ImuData>& imu_msg_buffer,
        Eigen::Vector3d& m_gyro_old, Eigen::Vector3d& m_acc_old, IMUState& imu_state);

    // If initialized
    bool ifInitialized() {
        return bInit;
    }

private:

    // Time lower bound for used imu data.
    double lower_time_bound;

    // Maximum feature distance allowed bewteen static images
    double max_feature_dis;

    // Number of consecutive image for trigger static initializer
    unsigned int static_Num;

    // Defined type for initialization
    typedef std::map<FeatureIDType, Eigen::Vector2d, std::less<int>,
        Eigen::aligned_allocator<
        std::pair<const FeatureIDType, Eigen::Vector2d> > > InitFeatures;
    InitFeatures init_features;

    // Counter for static images that will be used in inclinometer-initializer
    unsigned int staticImgCounter;

    // Error bewteen timestamp of imu and img
    double td;

    // Imu instrinsic
    Eigen::Matrix3d Ma;
    Eigen::Matrix3d Tg;
    Eigen::Matrix3d As;

    // Initialize results
    double state_time;
    Eigen::Vector3d gyro_bias;
    Eigen::Vector3d acc_bias;
    Eigen::Vector4d orientation;
    Eigen::Vector3d position;
    Eigen::Vector3d velocity;

    // Flag indicating if initialized
    bool bInit;

    // initialize rotation and gyro bias by static imu datas
    void initializeGravityAndBias(const double& time_bound,
        const std::vector<ImuData>& imu_msg_buffer);
};

} // namespace larvio


#endif //STATIC_INITIALIZER_H
