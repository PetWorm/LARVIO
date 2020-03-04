//
// Created by xiaochen at 19-8-21.
// Managing the image processer and the estimator.
//

#ifndef FEATURE_MSG_H
#define FEATURE_MSG_H

#include <vector>
#include <boost/shared_ptr.hpp>

namespace larvio {

    // Measurement for one features
    class MonoFeatureMeasurement {
    public:
        MonoFeatureMeasurement()
                : id(0)
                , u(0.0)
                , v(0.0)
                , u_init(0.0)
                , v_init(0.0)
                , u_vel(0.0)
                , v_vel(0.0)
                , u_init_vel(0.0)
                , v_init_vel(0.0)  {
        }

        // id
        unsigned long long int id;
        // Normalized feature coordinates (with identity intrinsic matrix)
        double u;   // horizontal coordinate
        double v;   // vertical coordinate
        // Normalized feature coordinates (with identity intrinsic matrix) in initial frame of this feature
        //# (meaningful if this is the first msg of this feature id)
        double u_init;
        double v_init;
        // Velocity of current normalized feature coordinate
        double u_vel;
        double v_vel;
        // Velocity of initial normalized feature coordinate
        double u_init_vel;
        double v_init_vel;
    };

    // Measurements for features in one camera img
    class MonoCameraMeasurement {
    public:
        double timeStampToSec;
        // All features on the current image,
        // including tracked ones and newly detected ones.
        std::vector<MonoFeatureMeasurement> features;
    };

//    typedef boost::shared_ptr<MonoCameraMeasurement> MonoCameraMeasurementPtr;
    typedef MonoCameraMeasurement* MonoCameraMeasurementPtr;

}

#endif  //FEATURE_MSG_H