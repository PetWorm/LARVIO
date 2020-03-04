//
// Added by xiaochen at 19-8-15.
// Feature manager.
// The original file belong to VINS-MONO (https://github.com/HKUST-Aerial-Robotics/VINS-Mono).
//

#ifndef FEATURE_MANAGER_H
#define FEATURE_MANAGER_H

#include <list>
#include <algorithm>
#include <vector>
#include <numeric>
using namespace std;

#include <Eigen/Dense>
using namespace Eigen;

#include <larvio/feature_msg.h>


namespace larvio {

const int WINDOW_SIZE = 10;
const double MIN_PARALLAX = 10/460;

class FeaturePerFrame
{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    FeaturePerFrame(const MonoFeatureMeasurement& _point, double td)
    {
        point.x() = _point.u+_point.u_vel*td;
        point.y() = _point.v+_point.v_vel*td;
        point.z() = 1;
        velocity.x() = _point.u_vel; 
        velocity.y() = _point.v_vel; 
        cur_td = td;
    }
    double cur_td;
    Vector3d point;
    Vector2d velocity;
    double z;
    bool is_used;
    double parallax;
    MatrixXd A;
    VectorXd b;
    double dep_gradient;
};

class FeaturePerId
{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    const int feature_id;
    int start_frame;
    vector<FeaturePerFrame> feature_per_frame;

    int used_num;
    bool is_outlier;
    bool is_margin;
    double estimated_depth;
    int solve_flag; // 0 haven't solve yet; 1 solve succ; 2 solve fail;

    Vector3d gt_p;

    FeaturePerId(int _feature_id, int _start_frame)
        : feature_id(_feature_id), start_frame(_start_frame),
          used_num(0), estimated_depth(-1.0), solve_flag(0)
    {
    }

    int endFrame();
};

class FeatureManager
{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    FeatureManager(){};

    void setRic(const Matrix3d& _ric);

    void clearState();

    int getFeatureCount();

    bool addFeatureCheckParallax(int frame_count, MonoCameraMeasurementPtr image, double td);
    vector<pair<Vector3d, Vector3d>> getCorresponding(int frame_count_l, int frame_count_r);

    //void updateDepth(const VectorXd &x);
    void setDepth(const VectorXd &x);
    void removeFailures();
    void clearDepth(const VectorXd &x);
    VectorXd getDepthVector();
    void removeBack();
    void removeFront(int frame_count);
    void removeOutlier();
    list<FeaturePerId> feature;
    int last_track_num;

  private:
    double compensatedParallax2(const FeaturePerId &it_per_id, int frame_count);
    Matrix3d ric;
};

}

#endif
