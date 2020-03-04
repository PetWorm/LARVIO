//
// Created by xiaochen at 19-8-13.
// A flexible initializer that can automatically initialize in case of static or dynamic scene.
//

#include "Initializer/FlexibleInitializer.h"

namespace larvio {

bool FlexibleInitializer::tryIncInit(std::vector<ImuData>& imu_msg_buffer,
        MonoCameraMeasurementPtr img_msg,
        Eigen::Vector3d& m_gyro_old, Eigen::Vector3d& m_acc_old, 
        IMUState& imu_state) {

    if(staticInitPtr->tryIncInit(imu_msg_buffer, img_msg)) {
        staticInitPtr->assignInitialState(imu_msg_buffer, 
            m_gyro_old, m_acc_old, imu_state);
        return true;
    } else if (dynamicInitPtr->tryDynInit(imu_msg_buffer, img_msg)) {
        dynamicInitPtr->assignInitialState(imu_msg_buffer, 
            m_gyro_old, m_acc_old, imu_state);
        return true;
    }

    return false;
}


}