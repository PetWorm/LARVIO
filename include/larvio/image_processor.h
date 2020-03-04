/*
 * COPYRIGHT AND PERMISSION NOTICE
 * Penn Software MSCKF_VIO
 * Copyright (C) 2017 The Trustees of the University of Pennsylvania
 * All rights reserved.
 */

// The original file belongs to MSCKF_VIO (https://github.com/KumarRobotics/msckf_vio/)
// Tremendous changes have been made to use it in LARVIO


#ifndef IMAGE_PROCESSOR_H
#define IMAGE_PROCESSOR_H

#include <larvio/feature_msg.h>

#include <vector>
#include <map>
#include <boost/shared_ptr.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/video.hpp>

#include <ORB/ORBDescriptor.h>

#include <fstream>

#include "sensors/ImuData.hpp"
#include "sensors/ImageData.hpp"

namespace larvio {

/*
 * @brief ImageProcessor Detects and tracks features
 *    in image sequences.
 */
class ImageProcessor {
public:
  // Constructor
  ImageProcessor(std::string& config_file_);
  // Disable copy and assign constructors.
  ImageProcessor(const ImageProcessor&) = delete;
  ImageProcessor operator=(const ImageProcessor&) = delete;

  // Destructor
  ~ImageProcessor();

  // Initialize the object.
  bool initialize();

  /*
   * @brief processImage
   *    Processing function for the monocular images.
   * @param image msg.
   * @param imu msg buffer.
   * @return true if have feature msg.
   */
  bool processImage(
        const ImageDataPtr& msg,
        const std::vector<ImuData>& imu_msg_buffer,
        MonoCameraMeasurementPtr features);

  // Get publish image.
  cv::Mat getVisualImg() {
      return visual_img;
  }

  typedef boost::shared_ptr<ImageProcessor> Ptr;
  typedef boost::shared_ptr<const ImageProcessor> ConstPtr;

private:

  /*
   * @brief ProcessorConfig Configuration parameters for
   *    feature detection and tracking.
   */
  struct ProcessorConfig {
    int pyramid_levels;
    int patch_size;
    int fast_threshold;
    int max_iteration;
    double track_precision;
    double ransac_threshold;

    int max_features_num;
    int min_distance;
    bool flag_equalize;

    int img_rate;
    int pub_frequency;
  };

  /*
   * @brief FeatureIDType An alias for unsigned long long int.
   */
  typedef unsigned long long int FeatureIDType;

  /*
   * @brief loadParameters
   *    Load parameters from the parameter server.
   */
  bool loadParameters();

  /*
   * @brief integrateImuData Integrates the IMU gyro readings
   *    between the two consecutive images, which is used for
   *    both tracking prediction and 2-point RANSAC.
   * @param imu msg buffer
   * @return cam_R_p2c: a rotation matrix which takes a vector
   *    from previous cam0 frame to current cam0 frame.
   */
  void integrateImuData(cv::Matx33f& cam_R_p2c,
            const std::vector<ImuData>& imu_msg_buffer);

  /*
   * @brief predictFeatureTracking Compensates the rotation
   *    between consecutive camera frames so that feature
   *    tracking would be more robust and fast.
   * @param input_pts: features in the previous image to be tracked.
   * @param R_p_c: a rotation matrix takes a vector in the previous
   *    camera frame to the current camera frame.
   * @param intrinsics: intrinsic matrix of the camera.
   * @return compensated_pts: predicted locations of the features
   *    in the current image based on the provided rotation.
   *
   * Note that the input and output points are of pixel coordinates.
   */
  void predictFeatureTracking(
      const std::vector<cv::Point2f>& input_pts,
      const cv::Matx33f& R_p_c,
      const cv::Vec4d& intrinsics,
      std::vector<cv::Point2f>& compenstated_pts);

  /*
   * @brief initializeFirstFrame
   *    Initialize the image processing sequence, which is
   *    bascially detect new features on the first image.
   */
  bool initializeFirstFrame();

  /*
   * @brief initializeFirstFeatures
   *    Initialize the image processing sequence, which is
   *    bascially detect new features on the first set of
   *    stereo images.
   * @param imu msg buffer
   */
  bool initializeFirstFeatures(
          const std::vector<ImuData>& imu_msg_buffer);

  /*
   * @brief findNewFeaturesToBeTracked
   *    Find new features in current image to be tracked,
   *    until being tracked successfully in next image,
   *    features found in this function would not be valid
   *    features.
   */
  void findNewFeaturesToBeTracked();

  /*
   * @brief trackFeatures
   *    Tracker features on the newly received image.
   */
  void trackFeatures();

  /*
   * @brief trackNewFeatures
   *    Track new features extracted in last image.
   */
  void trackNewFeatures();

  /*
   * @brief publish
   *    Publish the features on the current image including
   *    both the tracked and newly detected ones.
   */
  void publish();

  /*
   * @brief createImagePyramids
   *    Create image pyramids used for klt tracking.
   */
  void createImagePyramids();

  /*
   * @brief undistortPoints Undistort points based on camera
   *    calibration intrinsics and distort model.
   * @param pts_in: input distorted points.
   * @param intrinsics: intrinsics of the camera.
   * @param distortion_model: distortion model of the camera.
   * @param distortion_coeffs: distortion coefficients.
   * @param rectification_matrix: matrix to rectify undistorted points.
   * @param new_intrinsics: intrinsics on new camera for
   *    undistorted points to projected into.
   * @return pts_out: undistorted points.
   */
  void undistortPoints(
      const std::vector<cv::Point2f>& pts_in,
      const cv::Vec4d& intrinsics,
      const std::string& distortion_model,
      const cv::Vec4d& distortion_coeffs,
      std::vector<cv::Point2f>& pts_out,
      const cv::Matx33d &rectification_matrix = cv::Matx33d::eye(),
      const cv::Vec4d &new_intrinsics = cv::Vec4d(1,1,0,0));

  /*
   * @brief removeUnmarkedElements Remove the unmarked elements
   *    within a vector.
   * @param raw_vec: vector with outliers.
   * @param markers: 0 will represent a outlier, 1 will be an inlier.
   * @return refined_vec: a vector without outliers.
   *
   * Note that the order of the inliers in the raw_vec is perserved
   * in the refined_vec.
   */
  template <typename T>
  void removeUnmarkedElements(
      const std::vector<T>& raw_vec,
      const std::vector<unsigned char>& markers,
      std::vector<T>& refined_vec) {
    if (raw_vec.size() != markers.size()) {
      for (int i = 0; i < raw_vec.size(); ++i)
        refined_vec.push_back(raw_vec[i]);
      return;
    }
    for (int i = 0; i < markers.size(); ++i) {
      if (markers[i] == 0) continue;
      refined_vec.push_back(raw_vec[i]);
    }
    return;
  }

  /*
   * @brief rescalePoints Rescale image coordinate of pixels to gain
   *    numerical stability.
   * @param pts1: an array of image coordinate of some pixels,
   *    they will be rescaled.
   * @param pts2: corresponding image coordinate of some pixels
   *    in another image as pts2, they will be rescaled.
   * @return scaling_factor: scaling factor of the rescaling process.
   */
  void rescalePoints(
      std::vector<cv::Point2f>& pts1,
      std::vector<cv::Point2f>& pts2,
      float& scaling_factor);

  /*
   * @brief getFeatureMsg Get processed feature msg.
   * @return pointer for processed features
   */
  void getFeatureMsg(MonoCameraMeasurementPtr features);

  // Enum type for image state.
  enum eImageState {
      FIRST_IMAGE = 1,
      SECOND_IMAGE = 2,
      OTHER_IMAGES = 3
  };

  // Indicate if this is the first or second image message.
  eImageState image_state;

  // ID for the next new feature.
  FeatureIDType next_feature_id;

  // Feature detector
  ProcessorConfig processor_config;

  // Camera calibration parameters
  std::string cam_distortion_model;
  cv::Vec2i cam_resolution;
  cv::Vec4d cam_intrinsics;
  cv::Vec4d cam_distortion_coeffs;

  // Take a vector from cam frame to the IMU frame.
  cv::Matx33d R_cam_imu;  
  cv::Vec3d t_cam_imu;    

  // Take a vector from prev cam frame to curr cam frame
  cv::Matx33f R_Prev2Curr;  

  // Previous and current images
  ImageDataPtr prev_img_ptr;
  ImageDataPtr curr_img_ptr;

  // Pyramids for previous and current image
  std::vector<cv::Mat> prev_pyramid_;
  std::vector<cv::Mat> curr_pyramid_;

  // Number of features after each outlier removal step.
  int before_tracking;
  int after_tracking;
  int after_ransac;

  // Config file path
  std::string config_file;

  // Image for visualization
  cv::Mat visual_img;

  // Points for tracking, added by QXC
  std::vector<cv::Point2f> new_pts_;
  std::vector<cv::Point2f> prev_pts_;
  std::vector<cv::Point2f> curr_pts_;
  std::vector<FeatureIDType> pts_ids_;
  std::vector<int> pts_lifetime_;
  std::vector<cv::Point2f> init_pts_;

  // Time of last published image
  double last_pub_time;
  double curr_img_time;
  double prev_img_time;

  // Publish counter
  long pub_counter;

  // debug log:
  std::string output_dir;

  // ORB descriptor pointer, added by QXC
  boost::shared_ptr<ORBdescriptor> prevORBDescriptor_ptr;
  boost::shared_ptr<ORBdescriptor> currORBDescriptor_ptr;
  std::vector<cv::Mat> vOrbDescriptors;

  // flag for first useful image msg
  bool bFirstImg;
};

typedef ImageProcessor::Ptr ImageProcessorPtr;
typedef ImageProcessor::ConstPtr ImageProcessorConstPtr;

} // end namespace larvio

#endif
