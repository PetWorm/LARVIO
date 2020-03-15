//
// Created by xiaochen at 19-8-21.
// Nodelet for system manager.
//

#include <System_nodelet.h>

namespace larvio {
    
void SystemNodelet::onInit() {
    system_ptr.reset(new System(getPrivateNodeHandle()));
    if (!system_ptr->initialize()) {
        ROS_ERROR("Cannot initialize System Manager...");
        return;
    }
    return;
}

PLUGINLIB_EXPORT_CLASS(larvio::SystemNodelet, nodelet::Nodelet);

} // end namespace larvio
