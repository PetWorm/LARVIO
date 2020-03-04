/*
 * @Descripttion: Types of IMU sensor data.
 * @Author: Xiaochen Qiu
 */


#ifndef IMU_DATA_HPP
#define IMU_DATA_HPP


#include "Eigen/Core"
#include "Eigen/Dense"

namespace larvio {

struct ImuData {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    ImuData (double t, double wx, double wy, double wz, 
            double ax, double ay, double az) {
        timeStampToSec = t;
        angular_velocity[0] = wx;
        angular_velocity[1] = wy;
        angular_velocity[2] = wz;
        linear_acceleration[0] = ax;
        linear_acceleration[1] = ay;
        linear_acceleration[2] = az;
    }

    ImuData (double t, const Eigen::Vector3d& omg, const Eigen::Vector3d& acc) {
        timeStampToSec = t;
        angular_velocity = omg;
        linear_acceleration = acc;
    }

    double timeStampToSec;
    Eigen::Vector3d angular_velocity;
    Eigen::Vector3d linear_acceleration;
};

}


#endif // IMU_DATA_HPP