/*
 * COPYRIGHT AND PERMISSION NOTICE
 * Penn Software MSCKF_VIO
 * Copyright (C) 2017 The Trustees of the University of Pennsylvania
 * All rights reserved.
 */

// The original file belongs to MSCKF_VIO (https://github.com/KumarRobotics/msckf_vio/)
// Tremendous changes have been done to use it in LARVIO

#include <iostream>
#include <algorithm>
#include <set>
#include <Eigen/Dense>

#include <larvio/image_processor.h>
#include <larvio/math_utils.hpp>

#include <opencv2/core/utility.hpp>

using namespace std;
using namespace cv;
using namespace Eigen;

namespace larvio {


ImageProcessor::ImageProcessor(std::string& config_file_) :
        config_file(config_file_) {
    image_state = FIRST_IMAGE;
    next_feature_id = 0;

    return;
}


ImageProcessor::~ImageProcessor() {
    destroyAllWindows();

    return;
}


bool ImageProcessor::loadParameters() {
    cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
    if (!fsSettings.isOpened()) {
        cout << "config_file error: cannot open " << config_file << endl;
        return false;
    }

    processor_config.fast_threshold = fsSettings["fast_threshold"];
    processor_config.patch_size = fsSettings["patch_size"];
    processor_config.pyramid_levels = fsSettings["pyramid_levels"];
    processor_config.max_iteration = fsSettings["max_iteration"];
    processor_config.track_precision = fsSettings["track_precision"];
    processor_config.ransac_threshold = fsSettings["ransac_threshold"];

    processor_config.max_features_num = fsSettings["max_features_num"];
    processor_config.min_distance = fsSettings["min_distance"];
    processor_config.flag_equalize = (static_cast<int>(fsSettings["flag_equalize"]) ? true : false);

    processor_config.pub_frequency = fsSettings["pub_frequency"];
    processor_config.img_rate = fsSettings["img_rate"];

    // Output files directory
    fsSettings["output_dir"] >> output_dir;

    /*
     * Camera calibration parameters
     */
    // Distortion model
    fsSettings["distortion_model"] >> cam_distortion_model;
    // Resolution of camera
    cam_resolution[0] = fsSettings["resolution_width"];
    cam_resolution[1] = fsSettings["resolution_height"];
    // Camera calibration instrinsics
    cv::FileNode n_instrin = fsSettings["intrinsics"];
    cam_intrinsics[0] = static_cast<double>(n_instrin["fx"]);
    cam_intrinsics[1] = static_cast<double>(n_instrin["fy"]);
    cam_intrinsics[2] = static_cast<double>(n_instrin["cx"]);
    cam_intrinsics[3] = static_cast<double>(n_instrin["cy"]);
    // Distortion coefficient
    cv::FileNode n_distort = fsSettings["distortion_coeffs"];
    cam_distortion_coeffs[0] = static_cast<double>(n_distort["k1"]);
    cam_distortion_coeffs[1] = static_cast<double>(n_distort["k2"]);
    cam_distortion_coeffs[2] = static_cast<double>(n_distort["p1"]);
    cam_distortion_coeffs[3] = static_cast<double>(n_distort["p2"]);
    // Extrinsics between camera and IMU
    cv::Mat T_imu_cam;
    fsSettings["T_cam_imu"] >> T_imu_cam;
    cv::Matx33d R_imu_cam(T_imu_cam(cv::Rect(0,0,3,3)));      
    cv::Vec3d t_imu_cam = T_imu_cam(cv::Rect(3,0,1,3));
    R_cam_imu = R_imu_cam.t();
    t_cam_imu = -R_imu_cam.t() * t_imu_cam;

    // cout << ".fast_threshold = " << processor_config.fast_threshold << endl;
    // cout << ".patch_size = " << processor_config.patch_size << endl;
    // cout << ".pyramid_levels = " << processor_config.pyramid_levels << endl;
    // cout << ".max_iteration = " << processor_config.max_iteration << endl;
    // cout << ".track_precision = " << processor_config.track_precision << endl;
    // cout << ".ransac_threshold = " << processor_config.ransac_threshold << endl;
    // cout << ".max_features_num = " << processor_config.max_features_num << endl;
    // cout << ".min_distance = " << processor_config.min_distance << endl;
    // cout << ".flag_equalize = " << processor_config.flag_equalize << endl;
    // cout << ".pub_frequency = " << processor_config.pub_frequency << endl;
    // cout << "cam_distortion_model = " << cam_distortion_model << endl;
    // cout << "cam_intrinsics = " << cam_intrinsics << endl;
    // cout << "cam_distortion_coeffs = " << cam_distortion_coeffs << endl;
    // cout << "R_cam_imu = " << R_cam_imu << endl;
    // cout << "t_cam_imu = " << t_cam_imu << endl;

    return true;
}


bool ImageProcessor::initialize() {
    if (!loadParameters()) return false;

    // Initialize publish counter
    pub_counter = 0;

    // Initialize flag for first useful img msg
    bFirstImg = false;

    return true;
}


// Process current img msg and return the feature msg.
bool ImageProcessor::processImage(const ImageDataPtr& msg,
        const std::vector<ImuData>& imu_msg_buffer, MonoCameraMeasurementPtr features) {

    // images are not utilized until receiving imu msgs ahead
    if (!bFirstImg) {
        if ((imu_msg_buffer.begin() != imu_msg_buffer.end()) && 
            (imu_msg_buffer.begin()->timeStampToSec-msg->timeStampToSec <= 0.0)) {
            bFirstImg = true;
            printf("Images from now on will be utilized...\n\n");
        }
        else
            return false;
    }

    curr_img_ptr = msg;

    // Build the image pyramids once since they're used at multiple places
    createImagePyramids();

    // Initialize ORBDescriptor pointer
    currORBDescriptor_ptr.reset(new ORBdescriptor(curr_pyramid_[0], 2, processor_config.pyramid_levels));

    // Get current image time
    curr_img_time = curr_img_ptr->timeStampToSec;

    // Flag to return;
    bool haveFeatures = false;

    // Detect features in the first frame.
    if ( FIRST_IMAGE==image_state ) {
        if (initializeFirstFrame())
            image_state = SECOND_IMAGE;
    } else if ( SECOND_IMAGE==image_state ) {
        if ( !initializeFirstFeatures(imu_msg_buffer) ) {
            image_state = FIRST_IMAGE;
        } else {
            // frequency control
            if ( curr_img_time-last_pub_time >= 0.9*(1.0/processor_config.pub_frequency) ) {
                // Find new features to be tracked
                findNewFeaturesToBeTracked();

                // Det processed feature
                getFeatureMsg(features);

                // Publishing msgs
                publish();

                haveFeatures = true;
            }

            image_state = OTHER_IMAGES;
        }
    } else if ( OTHER_IMAGES==image_state ) {
        // Integrate gyro data to get a guess of rotation between current and previous image
        integrateImuData(R_Prev2Curr, imu_msg_buffer);

        // Tracking features
        trackFeatures();

        // Track new features extracted in last image, and add them into the gird
        trackNewFeatures();

        // frequency control
        if ( curr_img_time-last_pub_time >= 0.9*(1.0/processor_config.pub_frequency) ) {
            // Find new features to be tracked
            findNewFeaturesToBeTracked();

            // Det processed feature
            getFeatureMsg(features);

            // Publishing msgs
            publish();

            haveFeatures = true;
        }
    }

    // Update the previous image and previous features.
    prev_img_ptr = curr_img_ptr;
    swap(prev_pyramid_,curr_pyramid_);
    prevORBDescriptor_ptr = currORBDescriptor_ptr;

    // Initialize the current features to empty vectors.
    swap(prev_pts_,curr_pts_);
    vector<Point2f>().swap(curr_pts_);

    prev_img_time = curr_img_time;

    return haveFeatures;
}


void ImageProcessor::integrateImuData(Matx33f& cam_R_p2c,
        const std::vector<ImuData>& imu_msg_buffer) {
    // Find the start and the end limit within the imu msg buffer.
    auto begin_iter = imu_msg_buffer.begin();
    while (begin_iter != imu_msg_buffer.end()) {
    if (begin_iter->timeStampToSec-
            prev_img_ptr->timeStampToSec < -0.0049)
        ++begin_iter;
    else
        break;
    }

    auto end_iter = begin_iter;
    while (end_iter != imu_msg_buffer.end()) {
    if (end_iter->timeStampToSec-
            curr_img_ptr->timeStampToSec < 0.0049)
        ++end_iter;
    else
        break;
    }

    // Compute the mean angular velocity in the IMU frame.
    Vec3f mean_ang_vel(0.0, 0.0, 0.0);
    for (auto iter = begin_iter; iter < end_iter; ++iter)
    mean_ang_vel += Vec3f(iter->angular_velocity[0],
        iter->angular_velocity[1], iter->angular_velocity[2]);

    if (end_iter-begin_iter > 0)
    mean_ang_vel *= 1.0f / (end_iter-begin_iter);

    // Transform the mean angular velocity from the IMU
    // frame to the cam0 and cam1 frames.
    Vec3f cam_mean_ang_vel = R_cam_imu.t() * mean_ang_vel;

    // Compute the relative rotation.
    double dtime = curr_img_ptr->timeStampToSec-
        prev_img_ptr->timeStampToSec;
    Rodrigues(cam_mean_ang_vel*dtime, cam_R_p2c);
    cam_R_p2c = cam_R_p2c.t();

    return;
}


void ImageProcessor::predictFeatureTracking(
    const vector<cv::Point2f>& input_pts,
    const cv::Matx33f& R_p_c,
    const cv::Vec4d& intrinsics,
    vector<cv::Point2f>& compensated_pts) {
    // Return directly if there are no input features.
    if (input_pts.size() == 0) {
        compensated_pts.clear();
        return;
    }
    compensated_pts.resize(input_pts.size());

    // Intrinsic matrix.
    cv::Matx33f K(
        intrinsics[0], 0.0, intrinsics[2],
        0.0, intrinsics[1], intrinsics[3],
        0.0, 0.0, 1.0);
    cv::Matx33f H = K * R_p_c * K.inv();  

    for (int i = 0; i < input_pts.size(); ++i) {
        cv::Vec3f p1(input_pts[i].x, input_pts[i].y, 1.0f);
        cv::Vec3f p2 = H * p1;
        compensated_pts[i].x = p2[0] / p2[2];
        compensated_pts[i].y = p2[1] / p2[2];
    }

    return;
}


void ImageProcessor::rescalePoints(
    vector<Point2f>& pts1, vector<Point2f>& pts2,
    float& scaling_factor) {
    scaling_factor = 0.0f;

    for (int i = 0; i < pts1.size(); ++i) {
        scaling_factor += sqrt(pts1[i].dot(pts1[i]));
        scaling_factor += sqrt(pts2[i].dot(pts2[i]));
    }

    scaling_factor = (pts1.size()+pts2.size()) /
        scaling_factor * sqrt(2.0f);

    for (int i = 0; i < pts1.size(); ++i) {
        pts1[i] *= scaling_factor;
        pts2[i] *= scaling_factor;
    }

    return;
}


void ImageProcessor::createImagePyramids() {
    const Mat& curr_img = curr_img_ptr->image;
    // CLAHE
    cv::Mat img_;
    if (processor_config.flag_equalize) {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        clahe->apply(curr_img, img_);
    }
    else
        img_ = curr_img;
    // Get Pyramid
    buildOpticalFlowPyramid(
        img_, curr_pyramid_,
        Size(processor_config.patch_size, processor_config.patch_size),
        processor_config.pyramid_levels, true, BORDER_REFLECT_101,
        BORDER_CONSTANT, false);     
}


bool ImageProcessor::initializeFirstFrame() {
    // Get current image
    const Mat& img = curr_pyramid_[0];

    // Detect new features on the frist image.
    vector<Point2f>().swap(new_pts_);
    cv::goodFeaturesToTrack(img, new_pts_, processor_config.max_features_num, 0.01, processor_config.min_distance);

    // Initialize last publish time
    last_pub_time = curr_img_ptr->timeStampToSec;

    if (new_pts_.size()>20)
        return true;
    else
        return false;
}


bool ImageProcessor::initializeFirstFeatures(
        const std::vector<ImuData>& imu_msg_buffer) {

    // Integrate gyro data to get a guess of ratation between current and previous image
    integrateImuData(R_Prev2Curr, imu_msg_buffer);

    // Pridict features in current image
    vector<Point2f> curr_pts(0);
    predictFeatureTracking(
        new_pts_, R_Prev2Curr, cam_intrinsics, curr_pts);

    // Using LK optical flow to track feaures
    vector<unsigned char> track_inliers(new_pts_.size());
    calcOpticalFlowPyrLK(
        prev_pyramid_, curr_pyramid_,
        new_pts_, curr_pts,
        track_inliers, noArray(),
        Size(processor_config.patch_size, processor_config.patch_size),
        processor_config.pyramid_levels,
        TermCriteria(TermCriteria::COUNT+TermCriteria::EPS,
            processor_config.max_iteration,
            processor_config.track_precision),
        cv::OPTFLOW_USE_INITIAL_FLOW);

    // Mark those tracked points out of the image region
    // as untracked.
    for (int i = 0; i < curr_pts.size(); ++i) {  
        if (track_inliers[i] == 0) continue;
        if (curr_pts[i].y < 0 ||
            curr_pts[i].y > curr_img_ptr->image.rows-1 ||
            curr_pts[i].x < 0 ||
            curr_pts[i].x > curr_img_ptr->image.cols-1)
            track_inliers[i] = 0;
    }

    // Remove outliers
    vector<Point2f> prev_pts_inImg_(0);
    vector<Point2f> curr_pts_inImg_(0);
    removeUnmarkedElements(    
            new_pts_, track_inliers, prev_pts_inImg_);
    removeUnmarkedElements(
            curr_pts, track_inliers, curr_pts_inImg_);

    // Return if not enough inliers
    if ( prev_pts_inImg_.size()<20 )
        return false;

    // Using reverse LK optical flow tracking to eliminate outliers
    vector<unsigned char> reverse_inliers(curr_pts_inImg_.size());
    vector<Point2f> prev_pts_cpy(prev_pts_inImg_);
    calcOpticalFlowPyrLK(
        curr_pyramid_, prev_pyramid_, 
        curr_pts_inImg_, prev_pts_cpy,
        reverse_inliers, noArray(),
        Size(processor_config.patch_size, processor_config.patch_size),
        processor_config.pyramid_levels,
        TermCriteria(TermCriteria::COUNT+TermCriteria::EPS,
            processor_config.max_iteration,
            processor_config.track_precision),
        cv::OPTFLOW_USE_INITIAL_FLOW);
    // Mark those tracked points out of the image region
    // as untracked.
    for (int i = 0; i < prev_pts_cpy.size(); ++i) {  
        if (reverse_inliers[i] == 0) continue;
        if (prev_pts_cpy[i].y < 0 ||
            prev_pts_cpy[i].y > prev_pyramid_[0].rows-1 ||
            prev_pts_cpy[i].x < 0 ||
            prev_pts_cpy[i].x > prev_pyramid_[0].cols-1) {
            reverse_inliers[i] = 0;
            continue;
        }
        float dis = cv::norm(prev_pts_cpy[i]-prev_pts_inImg_[i]);
        if (dis > 1)    
            reverse_inliers[i] = 0;
    }
    // Remove outliers
    vector<Point2f> prev_pts_inImg(0);
    vector<Point2f> curr_pts_inImg(0);
    removeUnmarkedElements(   
            prev_pts_inImg_, reverse_inliers, prev_pts_inImg);
    removeUnmarkedElements(
            curr_pts_inImg_, reverse_inliers, curr_pts_inImg);
    // Return if not enough inliers
    if ( prev_pts_inImg.size()<20 )
        return false;

    // Mark as outliers if descriptor distance is too large
    vector<int> levels(prev_pts_inImg.size(), 0);
    Mat prevDescriptors, currDescriptors;
    if (!prevORBDescriptor_ptr->computeDescriptors(prev_pts_inImg, levels, prevDescriptors) ||
        !currORBDescriptor_ptr->computeDescriptors(curr_pts_inImg, levels, currDescriptors)) {
        cerr << "error happen while compute descriptors" << endl;
        return false;
    }
    vector<int> vDis;
    for (int j = 0; j < currDescriptors.rows; ++j) {
        int dis = ORBdescriptor::computeDescriptorDistance(
                prevDescriptors.row(j), currDescriptors.row(j));
        vDis.push_back(dis);
    }
    vector<unsigned char> desc_inliers(prev_pts_inImg.size(), 0);
    vector<Mat> desc_first(0);
    for (int i = 0; i < prev_pts_inImg.size(); i++) {
        if (vDis[i]<=58) {  
            desc_inliers[i] = 1;
            desc_first.push_back(prevDescriptors.row(i));
        }
    }

    // Remove outliers
    vector<Point2f> prev_pts_inlier(0);
    vector<Point2f> curr_pts_inlier(0);
    removeUnmarkedElements(   
            prev_pts_inImg, desc_inliers, prev_pts_inlier);
    removeUnmarkedElements(
            curr_pts_inImg, desc_inliers, curr_pts_inlier);

    // Return if not enough inliers
    if ( prev_pts_inlier.size()<20 )
        return false;

    // Undistort inliers
    vector<Point2f> prev_unpts_inlier(prev_pts_inlier.size());
    vector<Point2f> curr_unpts_inlier(curr_pts_inlier.size());
    undistortPoints(
            prev_pts_inlier, cam_intrinsics, cam_distortion_model,
            cam_distortion_coeffs, prev_unpts_inlier, 
            cv::Matx33d::eye(), cam_intrinsics);
    undistortPoints(
            curr_pts_inlier, cam_intrinsics, cam_distortion_model,
            cam_distortion_coeffs, curr_unpts_inlier, 
            cv::Matx33d::eye(), cam_intrinsics);

    vector<unsigned char> ransac_inliers;

    float fx = cam_intrinsics[0];
    float fy = cam_intrinsics[1];
    float cx = cam_intrinsics[2];
    float cy = cam_intrinsics[3];
    Mat K = ( cv::Mat_<double> (3,3) << fx, 0, cx, 0, fy, cy, 0, 0, 1 );
    // findEssentialMat(
    //         prev_unpts_inlier, curr_unpts_inlier,
    //         K, cv::RANSAC, 0.999, 1.0, ransac_inliers);
    findFundamentalMat(
            prev_unpts_inlier, curr_unpts_inlier,
            cv::FM_RANSAC, 1.0, 0.99, ransac_inliers);

    vector<Point2f> prev_pts_matched(0);
    vector<Point2f> curr_pts_matched(0);
    vector<Mat> prev_desc_matched(0);
    removeUnmarkedElements(
            prev_pts_inlier, ransac_inliers, prev_pts_matched);
    removeUnmarkedElements(
            curr_pts_inlier, ransac_inliers, curr_pts_matched);
    removeUnmarkedElements(
            desc_first, ransac_inliers, prev_desc_matched);

    // Features initialized failed if less than 20 inliers are tracked
    if ( curr_pts_matched.size()<20 )   
        return false;

    // Fill initialized features into init_pts_, curr_pts_, 
    // and set their ids and lifetime
    vector<Point2f>().swap(prev_pts_);
    vector<Point2f>().swap(curr_pts_);
    vector<FeatureIDType>().swap(pts_ids_);
    vector<int>().swap(pts_lifetime_);
    vector<Point2f>().swap(init_pts_);
    vector<Mat>().swap(vOrbDescriptors);
    for (int i = 0; i < prev_pts_matched.size(); ++i) {
        prev_pts_.push_back(prev_pts_matched[i]);
        init_pts_.push_back(Point2f(-1,-1));       
        curr_pts_.push_back(curr_pts_matched[i]);
        pts_ids_.push_back(next_feature_id++);
        pts_lifetime_.push_back(2);
        vOrbDescriptors.push_back(prev_desc_matched[i]);
    }

    // Clear new_pts_
    vector<Point2f>().swap(new_pts_);

    return true;
}


void ImageProcessor::trackFeatures() {
    // Number of the features before tracking.
    before_tracking = prev_pts_.size();

    // Abort tracking if there is no features in
    // the previous frame.
    if (0 == before_tracking) {
        printf("No feature in prev img !\n");
        return;
    }

    // Pridict features in current image
    vector<Point2f> curr_points(prev_pts_.size());
    predictFeatureTracking(
        prev_pts_, R_Prev2Curr, cam_intrinsics, curr_points);

    // Using LK optical flow to track feaures
    vector<unsigned char> track_inliers(prev_pts_.size());
    calcOpticalFlowPyrLK(
        prev_pyramid_, curr_pyramid_,
        prev_pts_, curr_points,
        track_inliers, noArray(),
        Size(processor_config.patch_size, processor_config.patch_size),
        processor_config.pyramid_levels,
        TermCriteria(TermCriteria::COUNT+TermCriteria::EPS,
            processor_config.max_iteration,
            processor_config.track_precision),
        cv::OPTFLOW_USE_INITIAL_FLOW);

    // Mark those tracked points out of the image region
    // as untracked.
    for (int i = 0; i < curr_points.size(); ++i) {   
        if (track_inliers[i] == 0) continue;
        if (curr_points[i].y < 0 ||
            curr_points[i].y > curr_img_ptr->image.rows-1 ||
            curr_points[i].x < 0 ||
            curr_points[i].x > curr_img_ptr->image.cols-1)
            track_inliers[i] = 0;
    }  

    // Collect the tracked points.
    vector<FeatureIDType> prev_inImg_ids_(0);
    vector<int> prev_inImg_lifetime_(0);
    vector<Point2f> prev_inImg_points_(0);
    vector<Point2f> curr_inImg_points_(0);
    vector<Point2f> init_inImg_position_(0);
    vector<Mat> prev_imImg_desc_(0);
    removeUnmarkedElements(   
            pts_ids_, track_inliers, prev_inImg_ids_);
    removeUnmarkedElements(
            pts_lifetime_, track_inliers, prev_inImg_lifetime_);
    removeUnmarkedElements(
            prev_pts_, track_inliers, prev_inImg_points_);
    removeUnmarkedElements(
            curr_points, track_inliers, curr_inImg_points_);
    removeUnmarkedElements(
            init_pts_, track_inliers, init_inImg_position_);
    removeUnmarkedElements(
            vOrbDescriptors, track_inliers, prev_imImg_desc_);

    // Number of features left after tracking.
    after_tracking = curr_inImg_points_.size();

    // debug log
    if (0 == after_tracking) {
        printf("No feature is tracked !");
        vector<Point2f>().swap(prev_pts_);
        vector<Point2f>().swap(curr_pts_);
        vector<FeatureIDType>().swap(pts_ids_);
        vector<int>().swap(pts_lifetime_);
        vector<Point2f>().swap(init_pts_);
        vector<Mat>().swap(vOrbDescriptors);
        return;
    }

    // Using reverse LK optical flow tracking to eliminate outliers
    vector<unsigned char> reverse_inliers(curr_inImg_points_.size());
    vector<Point2f> prev_pts_cpy(prev_inImg_points_);
    calcOpticalFlowPyrLK(
        curr_pyramid_, prev_pyramid_, 
        curr_inImg_points_, prev_pts_cpy,
        reverse_inliers, noArray(),
        Size(processor_config.patch_size, processor_config.patch_size),
        processor_config.pyramid_levels,
        TermCriteria(TermCriteria::COUNT+TermCriteria::EPS,
            processor_config.max_iteration,
            processor_config.track_precision),
        cv::OPTFLOW_USE_INITIAL_FLOW);
    // Mark those tracked points out of the image region
    // as untracked.
    for (int i = 0; i < prev_pts_cpy.size(); ++i) {  
        if (reverse_inliers[i] == 0) continue;
        if (prev_pts_cpy[i].y < 0 ||
            prev_pts_cpy[i].y > prev_pyramid_[0].rows-1 ||
            prev_pts_cpy[i].x < 0 ||
            prev_pts_cpy[i].x > prev_pyramid_[0].cols-1) {
            reverse_inliers[i] = 0;
            continue;
        }
        float dis = cv::norm(prev_pts_cpy[i]-prev_inImg_points_[i]);
        if (dis > 1)    
            reverse_inliers[i] = 0;
    }
    // Remove outliers
    vector<FeatureIDType> prev_inImg_ids(0);
    vector<int> prev_inImg_lifetime(0);
    vector<Point2f> prev_inImg_points(0);
    vector<Point2f> curr_inImg_points(0);
    vector<Point2f> init_inImg_position(0);
    vector<Mat> prev_imImg_desc(0);
    removeUnmarkedElements(   
            prev_inImg_ids_, reverse_inliers, prev_inImg_ids);
    removeUnmarkedElements(
            prev_inImg_lifetime_, reverse_inliers, prev_inImg_lifetime);
    removeUnmarkedElements(
            prev_inImg_points_, reverse_inliers, prev_inImg_points);
    removeUnmarkedElements(
            curr_inImg_points_, reverse_inliers, curr_inImg_points);
    removeUnmarkedElements(
            init_inImg_position_, reverse_inliers, init_inImg_position);
    removeUnmarkedElements(
            prev_imImg_desc_, reverse_inliers, prev_imImg_desc);
    // Number of features left after tracking.
    after_tracking = curr_inImg_points.size();
    // debug log
    if (0 == after_tracking) {
        printf("No feature is tracked !");
        vector<Point2f>().swap(prev_pts_);
        vector<Point2f>().swap(curr_pts_);
        vector<FeatureIDType>().swap(pts_ids_);
        vector<int>().swap(pts_lifetime_);
        vector<Point2f>().swap(init_pts_);
        vector<Mat>().swap(vOrbDescriptors);
        return;
    }

    // Mark as outliers if descriptor distance is too large
    vector<int> levels(prev_inImg_points.size(), 0);
    Mat prevDescriptors, currDescriptors;
    if (!currORBDescriptor_ptr->computeDescriptors(curr_inImg_points, levels, currDescriptors)) {
        cerr << "error happen while compute descriptors" << endl;
        vector<Point2f>().swap(prev_pts_);
        vector<Point2f>().swap(curr_pts_);
        vector<FeatureIDType>().swap(pts_ids_);
        vector<int>().swap(pts_lifetime_);
        vector<Point2f>().swap(init_pts_);
        vector<Mat>().swap(vOrbDescriptors);
        return;
    }
    vector<int> vDis;
    for (int j = 0; j < currDescriptors.rows; ++j) {
        int dis = ORBdescriptor::computeDescriptorDistance(
                prev_imImg_desc[j], currDescriptors.row(j));
        vDis.push_back(dis);
    }
    vector<unsigned char> desc_inliers(prev_inImg_points.size(), 0);
    for (int i = 0; i < prev_inImg_points.size(); i++) {
        if (vDis[i]<=58)  
            desc_inliers[i] = 1;
    }

    // Remove outliers
    vector<FeatureIDType> prev_tracked_ids(0);
    vector<int> prev_tracked_lifetime(0);
    vector<Point2f> prev_tracked_points(0);
    vector<Point2f> curr_tracked_points(0);
    vector<Point2f> init_tracked_position(0);
    vector<Mat> prev_tracked_desc(0);
    removeUnmarkedElements(    
            prev_inImg_ids, desc_inliers, prev_tracked_ids);
    removeUnmarkedElements(
            prev_inImg_lifetime, desc_inliers, prev_tracked_lifetime);
    removeUnmarkedElements(
            prev_inImg_points, desc_inliers, prev_tracked_points);
    removeUnmarkedElements(
            curr_inImg_points, desc_inliers, curr_tracked_points);
    removeUnmarkedElements(
            init_inImg_position, desc_inliers, init_tracked_position);
    removeUnmarkedElements(
            prev_imImg_desc, desc_inliers, prev_tracked_desc);

    // Return if not enough inliers
    if ( prev_tracked_points.size()==0 ){
        printf("No feature is tracked after descriptor matching!\n");
        vector<Point2f>().swap(prev_pts_);
        vector<Point2f>().swap(curr_pts_);
        vector<FeatureIDType>().swap(pts_ids_);
        vector<int>().swap(pts_lifetime_);
        vector<Point2f>().swap(init_pts_);
        vector<Mat>().swap(vOrbDescriptors);
        return;
    }

    // Further remove outliers by RANSAC.
    vector<Point2f> prev_tracked_unpts(prev_tracked_points.size());
    vector<Point2f> curr_tracked_unpts(curr_tracked_points.size());
    undistortPoints(
            prev_tracked_points, cam_intrinsics, cam_distortion_model,
            cam_distortion_coeffs, prev_tracked_unpts, 
            cv::Matx33d::eye(), cam_intrinsics);
    undistortPoints(
            curr_tracked_points, cam_intrinsics, cam_distortion_model,
            cam_distortion_coeffs, curr_tracked_unpts, 
            cv::Matx33d::eye(), cam_intrinsics);

    vector<unsigned char> ransac_inliers;

    float fx = cam_intrinsics[0];
    float fy = cam_intrinsics[1];
    float cx = cam_intrinsics[2];
    float cy = cam_intrinsics[3];
    Mat K = ( cv::Mat_<double> (3,3) << fx, 0, cx, 0, fy, cy, 0, 0, 1 );
    // findEssentialMat(
    //         prev_tracked_unpts, curr_tracked_unpts,
    //         K, cv::RANSAC, 0.999, 1.0, ransac_inliers);
    findFundamentalMat(
            prev_tracked_unpts, curr_tracked_unpts,
            cv::FM_RANSAC, 1.0, 0.99, ransac_inliers);

    // Remove outliers
    vector<FeatureIDType> prev_matched_ids(0);
    vector<int> prev_matched_lifetime(0);
    vector<Point2f> prev_matched_points(0);
    vector<Point2f> curr_matched_points(0);
    vector<Point2f> init_matched_position(0);
    vector<Mat> prev_matched_desc(0);
    removeUnmarkedElements(
            prev_tracked_ids, ransac_inliers, prev_matched_ids);
    removeUnmarkedElements(
            prev_tracked_lifetime, ransac_inliers, prev_matched_lifetime);
    removeUnmarkedElements(
            prev_tracked_points, ransac_inliers, prev_matched_points);
    removeUnmarkedElements(
            curr_tracked_points, ransac_inliers, curr_matched_points);
    removeUnmarkedElements(
            init_tracked_position, ransac_inliers, init_matched_position);
    removeUnmarkedElements(
            prev_tracked_desc, ransac_inliers, prev_matched_desc);

    // Number of matched features left after RANSAC.
    after_ransac = curr_matched_points.size();

    // debug log
    if (0 == after_ransac) {
        printf("No feature survive after RANSAC !");
        vector<Point2f>().swap(prev_pts_);
        vector<Point2f>().swap(curr_pts_);
        vector<FeatureIDType>().swap(pts_ids_);
        vector<int>().swap(pts_lifetime_);
        vector<Point2f>().swap(init_pts_);
        vector<Mat>().swap(vOrbDescriptors);
        return;
    }

    // Puts tracked and mateched points into grids
    vector<Point2f>().swap(prev_pts_);
    vector<Point2f>().swap(curr_pts_);
    vector<FeatureIDType>().swap(pts_ids_);
    vector<int>().swap(pts_lifetime_);
    vector<Point2f>().swap(init_pts_);
    vector<Mat>().swap(vOrbDescriptors);
    for (int i = 0; i < curr_matched_points.size(); ++i) {
        prev_pts_.push_back(prev_matched_points[i]);    
        curr_pts_.push_back(curr_matched_points[i]);
        pts_ids_.push_back(prev_matched_ids[i]);
        pts_lifetime_.push_back(++prev_matched_lifetime[i]);
        init_pts_.push_back(init_matched_position[i]);
        vOrbDescriptors.push_back(prev_matched_desc[i]);
    }

    return;
}

void ImageProcessor::trackNewFeatures() {
    // Return if no new features
    int num_new = new_pts_.size();
    if ( num_new<=0 ) {
        // printf("NO NEW FEATURES EXTRACTED IN LAST IMAGE");
        return;
    }
    // else
    //     printf("%d NEW FEATURES EXTRACTED IN LAST IMAGE",num_new);

    // Pridict features in current image
    vector<Point2f> curr_pts(new_pts_.size());
    predictFeatureTracking(
        new_pts_, R_Prev2Curr, cam_intrinsics, curr_pts);

    // Using LK optical flow to track feaures
    vector<unsigned char> track_inliers(new_pts_.size());
    calcOpticalFlowPyrLK(
        prev_pyramid_, curr_pyramid_,
        new_pts_, curr_pts,
        track_inliers, noArray(),
        Size(processor_config.patch_size, processor_config.patch_size),
        processor_config.pyramid_levels,
        TermCriteria(TermCriteria::COUNT+TermCriteria::EPS,
            processor_config.max_iteration,
            processor_config.track_precision),
        cv::OPTFLOW_USE_INITIAL_FLOW);

    // Mark those tracked points out of the image region
    // as untracked.
    for (int i = 0; i < curr_pts.size(); ++i) {   
        if (track_inliers[i] == 0) continue;
        if (curr_pts[i].y < 0 ||
            curr_pts[i].y > curr_img_ptr->image.rows-1 ||
            curr_pts[i].x < 0 ||
            curr_pts[i].x > curr_img_ptr->image.cols-1)
            track_inliers[i] = 0;
    }  

    // Use inliers to do RANSAC and further remove outliers
    vector<Point2f> prev_pts_inImg_(0);
    vector<Point2f> curr_pts_inImg_(0);
    removeUnmarkedElements(   
            new_pts_, track_inliers, prev_pts_inImg_);
    removeUnmarkedElements(
            curr_pts, track_inliers, curr_pts_inImg_);

    // Return if no new feature was tracked
    int num_tracked = prev_pts_inImg_.size();
    if ( num_tracked<=0 ) {
        // printf("NO NEW FEATURE IN LAST IMAGE WAS TRACKED");
        return;
    }

    // Using reverse LK optical flow tracking to eliminate outliers
    vector<unsigned char> reverse_inliers(curr_pts_inImg_.size());
    vector<Point2f> prev_pts_cpy(prev_pts_inImg_);
    calcOpticalFlowPyrLK(
        curr_pyramid_, prev_pyramid_, 
        curr_pts_inImg_, prev_pts_cpy,
        reverse_inliers, noArray(),
        Size(processor_config.patch_size, processor_config.patch_size),
        processor_config.pyramid_levels,
        TermCriteria(TermCriteria::COUNT+TermCriteria::EPS,
            processor_config.max_iteration,
            processor_config.track_precision),
        cv::OPTFLOW_USE_INITIAL_FLOW);
    // Mark those tracked points out of the image region
    // as untracked.
    for (int i = 0; i < prev_pts_cpy.size(); ++i) {  
        if (reverse_inliers[i] == 0) continue;
        if (prev_pts_cpy[i].y < 0 ||
            prev_pts_cpy[i].y > prev_pyramid_[0].rows-1 ||
            prev_pts_cpy[i].x < 0 ||
            prev_pts_cpy[i].x > prev_pyramid_[0].cols-1) {
            reverse_inliers[i] = 0;
            continue;
        }
        float dis = cv::norm(prev_pts_cpy[i]-prev_pts_inImg_[i]);
        if (dis > 1)    
            reverse_inliers[i] = 0;
    }
    // Remove outliers
    vector<Point2f> prev_pts_inImg(0);
    vector<Point2f> curr_pts_inImg(0);
    removeUnmarkedElements(    
            prev_pts_inImg_, reverse_inliers, prev_pts_inImg);
    removeUnmarkedElements(
            curr_pts_inImg_, reverse_inliers, curr_pts_inImg);
    // Return if no new feature was tracked
    num_tracked = prev_pts_inImg.size();
    if ( num_tracked<=0 ) {
        // printf("NO NEW FEATURE IN LAST IMAGE WAS TRACKED");
        return;
    }

    // Mark as outliers if descriptor distance is too large
    vector<int> levels(prev_pts_inImg.size(), 0);
    Mat prevDescriptors, currDescriptors;
    if (!prevORBDescriptor_ptr->computeDescriptors(prev_pts_inImg, levels, prevDescriptors) ||
        !currORBDescriptor_ptr->computeDescriptors(curr_pts_inImg, levels, currDescriptors)) {
        cerr << "error happen while compute descriptors" << endl;
        return;
    }
    vector<int> vDis;
    for (int j = 0; j < prevDescriptors.rows; ++j) {
        int dis = ORBdescriptor::computeDescriptorDistance(
                prevDescriptors.row(j), currDescriptors.row(j));
        vDis.push_back(dis);
    }
    vector<unsigned char> desc_inliers(prev_pts_inImg.size(), 0);
    vector<Mat> desc_new(0);
    for (int i = 0; i < prev_pts_inImg.size(); i++) {
        if (vDis[i]<=58) {  
            desc_inliers[i] = 1;
            desc_new.push_back(prevDescriptors.row(i));
        }
    }

    // Remove outliers
    vector<Point2f> prev_pts_inlier(0);
    vector<Point2f> curr_pts_inlier(0);
    removeUnmarkedElements(    
            prev_pts_inImg, desc_inliers, prev_pts_inlier);
    removeUnmarkedElements(
            curr_pts_inImg, desc_inliers, curr_pts_inlier);

    // Return if not enough inliers
    if ( prev_pts_inlier.size()<20 ){
        // printf("NO NEW FEATURE IN LAST IMAGE WAS TRACKED");
        return;
    }

    // Undistort inliers
    vector<Point2f> prev_unpts_inlier(prev_pts_inlier.size());
    vector<Point2f> curr_unpts_inlier(curr_pts_inlier.size());
    undistortPoints(
            prev_pts_inlier, cam_intrinsics, cam_distortion_model,
            cam_distortion_coeffs, prev_unpts_inlier, 
            cv::Matx33d::eye(), cam_intrinsics);
    undistortPoints(
            curr_pts_inlier, cam_intrinsics, cam_distortion_model,
            cam_distortion_coeffs, curr_unpts_inlier, 
            cv::Matx33d::eye(), cam_intrinsics);

    vector<unsigned char> ransac_inliers;

    float fx = cam_intrinsics[0];
    float fy = cam_intrinsics[1];
    float cx = cam_intrinsics[2];
    float cy = cam_intrinsics[3];
    Mat K = ( cv::Mat_<double> (3,3) << fx, 0, cx, 0, fy, cy, 0, 0, 1 );
    // findEssentialMat(
    //         prev_unpts_inlier, curr_unpts_inlier,
    //         K, cv::RANSAC, 0.999, 1.0, ransac_inliers);
    findFundamentalMat(
            prev_unpts_inlier, curr_unpts_inlier,
            cv::FM_RANSAC, 1.0, 0.99, ransac_inliers);

    vector<Point2f> prev_pts_matched(0);
    vector<Point2f> curr_pts_matched(0);
    vector<Mat> prev_desc_matched(0);
    removeUnmarkedElements(
            prev_pts_inlier, ransac_inliers, prev_pts_matched);
    removeUnmarkedElements(
            curr_pts_inlier, ransac_inliers, curr_pts_matched);
    removeUnmarkedElements(
            desc_new, ransac_inliers, prev_desc_matched);

    // Return if no new feature was tracked
    int num_ransac = curr_pts_matched.size();
    if ( num_ransac<=0 ) {
        // printf("NO NEW FEATURE IN LAST IMAGE WAS TRACKED");
        return;
    }

    // Fill initialized features into init_pts_, curr_pts_, 
    // and set their ids and lifetime
    for (int i = 0; i < prev_pts_matched.size(); ++i) {
        prev_pts_.push_back(prev_pts_matched[i]);
        curr_pts_.push_back(curr_pts_matched[i]);
        pts_ids_.push_back(next_feature_id++);
        pts_lifetime_.push_back(2);
        init_pts_.push_back(prev_pts_matched[i]);
        vOrbDescriptors.push_back(prev_desc_matched[i]);
    }

    // Clear new_pts_
    vector<Point2f>().swap(new_pts_);
}


void ImageProcessor::findNewFeaturesToBeTracked() {
    const Mat& curr_img = curr_pyramid_[0];

    // Create a mask to avoid redetecting existing features.
    cv::Mat mask(curr_img.rows,curr_img.cols,CV_8UC1,255);
    for (const auto& pt : curr_pts_) {
        // int startRow = round(pt.y) - processor_config.patch_size;
        int startRow = round(pt.y) - processor_config.min_distance;
        startRow = (startRow<0) ? 0 : startRow;

        // int endRow = round(pt.y) + processor_config.patch_size;
        int endRow = round(pt.y) + processor_config.min_distance;
        endRow = (endRow>curr_img.rows-1) ? curr_img.rows-1 : endRow;

        // int startCol = round(pt.x) - processor_config.patch_size;
        int startCol = round(pt.x) - processor_config.min_distance;
        startCol = (startCol<0) ? 0 : startCol;

        // int endCol = round(pt.x) + processor_config.patch_size;
        int endCol = round(pt.x) + processor_config.min_distance;
        endCol = (endCol>curr_img.cols-1) ? curr_img.cols-1 : endCol;

        cv::Mat mROI(mask,
                    cv::Rect(startCol,startRow,endCol-startCol+1,endRow-startRow+1));
        mROI.setTo(0);
    }

    // detect new features to be tracked
    vector<Point2f>().swap(new_pts_);
    if (processor_config.max_features_num-curr_pts_.size() > 0)
        cv::goodFeaturesToTrack(curr_img, new_pts_, 
            processor_config.max_features_num-curr_pts_.size(), 0.01, processor_config.min_distance, mask);
}


void ImageProcessor::undistortPoints(
    const vector<cv::Point2f>& pts_in,
    const cv::Vec4d& intrinsics,
    const string& distortion_model,
    const cv::Vec4d& distortion_coeffs,
    vector<cv::Point2f>& pts_out,
    const cv::Matx33d &rectification_matrix,
    const cv::Vec4d &new_intrinsics) {
    if (pts_in.size() == 0) return;

    const cv::Matx33d K(
            intrinsics[0], 0.0, intrinsics[2],
            0.0, intrinsics[1], intrinsics[3],
            0.0, 0.0, 1.0);

    const cv::Matx33d K_new(
            new_intrinsics[0], 0.0, new_intrinsics[2],
            0.0, new_intrinsics[1], new_intrinsics[3],
            0.0, 0.0, 1.0);

    if (distortion_model == "radtan") {
        cv::undistortPoints(pts_in, pts_out, K, distortion_coeffs,
                            rectification_matrix, K_new);       
    } else if (distortion_model == "equidistant") {
        cv::fisheye::undistortPoints(pts_in, pts_out, K, distortion_coeffs,
                                     rectification_matrix, K_new);
    } else {
        printf("The model %s is unrecognized, use radtan instead...",
                      distortion_model.c_str());
        cv::undistortPoints(pts_in, pts_out, K, distortion_coeffs,
                            rectification_matrix, K_new);
    }
}


// Get processed feature msg.
void ImageProcessor::getFeatureMsg(MonoCameraMeasurementPtr feature_msg_ptr) {
    feature_msg_ptr->timeStampToSec = curr_img_ptr->timeStampToSec;

    vector<Point2f> curr_points_undistorted(0);
    vector<Point2f> init_points_undistorted(0);     
    vector<Point2f> prev_points_undistorted(0);     

    undistortPoints(
            curr_pts_, cam_intrinsics, cam_distortion_model,
            cam_distortion_coeffs, curr_points_undistorted);
    undistortPoints(
            init_pts_, cam_intrinsics, cam_distortion_model,
            cam_distortion_coeffs, init_points_undistorted);
    undistortPoints(
            prev_pts_, cam_intrinsics, cam_distortion_model,
            cam_distortion_coeffs, prev_points_undistorted);

    // time interval between current and previous image
    double dt_1 = curr_img_time-prev_img_time;
    bool prev_is_last = prev_img_time==last_pub_time;
    double dt_2 = (prev_is_last ? dt_1 : prev_img_time-last_pub_time);

    for (int i = 0; i < pts_ids_.size(); ++i) {
        feature_msg_ptr->features.push_back(MonoFeatureMeasurement());
        feature_msg_ptr->features[i].id = pts_ids_[i];
        feature_msg_ptr->features[i].u = curr_points_undistorted[i].x;
        feature_msg_ptr->features[i].v = curr_points_undistorted[i].y;
        feature_msg_ptr->features[i].u_vel =
                (curr_points_undistorted[i].x-prev_points_undistorted[i].x)/dt_1;
        feature_msg_ptr->features[i].v_vel =
                (curr_points_undistorted[i].y-prev_points_undistorted[i].y)/dt_1;
        if (init_pts_[i].x==-1 && init_pts_[i].y==-1) {    
            feature_msg_ptr->features[i].u_init = -1;
            feature_msg_ptr->features[i].v_init = -1;
        } else {
            feature_msg_ptr->features[i].u_init = init_points_undistorted[i].x;
            feature_msg_ptr->features[i].v_init = init_points_undistorted[i].y;
            init_pts_[i].x = -1;    
            init_pts_[i].y = -1;
            if (prev_is_last) {
                feature_msg_ptr->features[i].u_init_vel =
                        (curr_points_undistorted[i].x-init_points_undistorted[i].x)/dt_2;
                feature_msg_ptr->features[i].v_init_vel =
                        (curr_points_undistorted[i].y-init_points_undistorted[i].y)/dt_2;
            } else {
                feature_msg_ptr->features[i].u_init_vel =
                        (prev_points_undistorted[i].x-init_points_undistorted[i].x)/dt_2;
                feature_msg_ptr->features[i].v_init_vel =
                        (prev_points_undistorted[i].y-init_points_undistorted[i].y)/dt_2;
            }
        }
    }
}


void ImageProcessor::publish() {
    // Colors for different features.
    Scalar tracked(255, 0, 0);
    Scalar new_feature(0, 255, 0);
    // Create an output image.
    int img_height = curr_img_ptr->image.rows;
    int img_width = curr_img_ptr->image.cols;
    Mat out_img(img_height, img_width, CV_8UC3);
    cvtColor(curr_img_ptr->image, out_img, COLOR_GRAY2RGB);
    // Collect feature points in the previous frame, 
    // and feature points in the current frame, 
    // and lifetime of tracked features
    map<FeatureIDType, Point2f> prev_points;
    map<FeatureIDType, Point2f> curr_points;
    map<FeatureIDType, int> points_lifetime;
    for (int i = 0; i < pts_ids_.size(); i++) {
        prev_points[pts_ids_[i]] = prev_pts_[i];
        curr_points[pts_ids_[i]] = curr_pts_[i];
        points_lifetime[pts_ids_[i]] = pts_lifetime_[i];
    }
    // Draw tracked features.
    for (const auto& id : pts_ids_) {
        if (prev_points.find(id) != prev_points.end() &&
            curr_points.find(id) != curr_points.end()) {
            cv::Point2f prev_pt = prev_points[id];
            cv::Point2f curr_pt = curr_points[id];
            int life = points_lifetime[id];

            double len = std::min(1.0, 1.0 * life / 50);
            circle(out_img, curr_pt, 2, cv::Scalar(255 * (1 - len), 0, 255 * len), 5);
            line(out_img, prev_pt, curr_pt, Scalar(0,128,0));

            prev_points.erase(id);
            curr_points.erase(id);
        }
    }

    visual_img = out_img;

    // Update last publish time and publish counter
    last_pub_time = curr_img_ptr->timeStampToSec;
    pub_counter++;

    return;
}

} // end namespace larvio
