//
// Added by xiaochen at 19-8-16.
// Type and methods for initial alignment.
// The original file belong to VINS-MONO (https://github.com/HKUST-Aerial-Robotics/VINS-Mono).
//

#ifndef INITIAL_ALIGNMENT_H
#define INITIAL_ALIGNMENT_H

// #pragma once
#include <Eigen/Dense>
#include <iostream>
#include "Initializer/ImuPreintegration.h"
#include "Initializer/feature_manager.h"
#include <map>

#include <larvio/feature_msg.h>

#include <boost/shared_ptr.hpp>

using namespace Eigen;
using namespace std;

namespace larvio {

class ImageFrame
{
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        
        ImageFrame(){};
        ImageFrame(MonoCameraMeasurementPtr _points, const double& td):is_key_frame{false}
        {
            t = _points->timeStampToSec+td;
            for (const auto& pt : _points->features)
            {
                double x = pt.u+pt.u_vel*td;
                double y = pt.v+pt.v_vel*td;
                double z = 1;
                double p_u = pt.u;
                double p_v = pt.v;
                double velocity_x = pt.u_vel;
                double velocity_y = pt.v_vel;
                Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
                xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
                points[pt.id] = xyz_uv_velocity;
            }
        };
        map<int, Eigen::Matrix<double, 7, 1>> points;
        double t;
        Matrix3d R;
        Vector3d T;
        boost::shared_ptr<IntegrationBase> pre_integration;
        bool is_key_frame;
};

bool VisualIMUAlignment(map<double, ImageFrame> &all_image_frame, Vector3d* Bgs, Vector3d &g, VectorXd &x, const Vector3d& TIC);

}


#endif //INITIAL_ALIGNMENT_H
