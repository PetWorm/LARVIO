//
// Added by xiaochen at 19-8-16.
// Solving relative motion.
// The original file belong to VINS-MONO (https://github.com/HKUST-Aerial-Robotics/VINS-Mono).
//

#ifndef SOLVE_5PTS_H
#define SOLVE_5PTS_H

// #pragma once

#include <vector>

using namespace std;

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

using namespace Eigen;


namespace larvio {

class MotionEstimator
{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    bool solveRelativeRT(const vector<pair<Vector3d, Vector3d>> &corres, Matrix3d &R, Vector3d &T);
};

}


#endif //SOLVE_5PTS_H