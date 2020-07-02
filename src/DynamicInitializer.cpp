//
// Created by xiaochen at 19-8-13.
// A vio initializer that utlize dynamic imu and img data to initialize.
// The core method comes from VINS-MONO (https://github.com/HKUST-Aerial-Robotics/VINS-Mono)
//

#include "Initializer/DynamicInitializer.h"
#include "larvio/math_utils.hpp"

#include <opencv2/opencv.hpp>

// debug log
#include <iostream>
#include <iomanip>

using namespace std;
using namespace Eigen;

namespace larvio {

bool DynamicInitializer::tryDynInit(const std::vector<ImuData>& imu_msg_buffer,
        MonoCameraMeasurementPtr img_msg) {

    // Counter how many IMU msgs in the buffer are used.
    int used_imu_msg_cntr = 0;

    double time_bound = img_msg->timeStampToSec+td;

    for (const auto& imu_msg : imu_msg_buffer) {
        double imu_time = imu_msg.timeStampToSec;
        if ( imu_time <= lower_time_bound )  continue;
        if ( imu_time-time_bound > imu_img_timeTh ) break;  // threshold is adjusted according to the imu frequency
        ddt = imu_time-time_bound;  // ddt
        processIMU(imu_msg);
        ++used_imu_msg_cntr;
    }
    lower_time_bound = time_bound+imu_img_timeTh;

    processImage(img_msg);

    if (bInit) {
        printf("Dynamic initialization success !\n\n");
        return true;
    } else
        return false;
}


void DynamicInitializer::processIMU(const ImuData& imu_msg) {

    Vector3d angular_velocity, linear_acceleration;
    linear_acceleration = Ma*imu_msg.linear_acceleration;
    angular_velocity = Tg*(imu_msg.angular_velocity-As*linear_acceleration);

    // qxc: into this if only at the first time
    if (!first_imu) {
        first_imu = true;
        acc_0 = linear_acceleration;
        gyr_0 = angular_velocity;
        curr_time = imu_msg.timeStampToSec;
    }

    double dt = imu_msg.timeStampToSec-curr_time;

    if (!pre_integrations[frame_count])     // qxc: into this if only at the first time
    {
        pre_integrations[frame_count].reset(new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count],
                                                                acc_n, acc_w, gyr_n, gyr_w});
    }

    if (frame_count != 0)
    {
        pre_integrations[frame_count]->push_back(dt, linear_acceleration, angular_velocity);    
        tmp_pre_integration->push_back(dt, linear_acceleration, angular_velocity);    

        dt_buf[frame_count].push_back(dt);                                       
        linear_acceleration_buf[frame_count].push_back(linear_acceleration);
        angular_velocity_buf[frame_count].push_back(angular_velocity);

        int j = frame_count;
        Vector3d un_acc_0 = Rs[j] * (acc_0 - Bas[j]) - g;
        Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - Bgs[j];
        Rs[j] *= getSmallAngleQuaternion(un_gyr * dt).toRotationMatrix();         
        Vector3d un_acc_1 = Rs[j] * (linear_acceleration - Bas[j]) - g;
        Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
        Ps[j] += dt * Vs[j] + 0.5 * dt * dt * un_acc;
        Vs[j] += dt * un_acc;
    }

    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity;
    curr_time = imu_msg.timeStampToSec;

    // QXC: update last m_gyro and last m_acc
    last_gyro = angular_velocity;
    last_acc = linear_acceleration;
}


void DynamicInitializer::processImage(MonoCameraMeasurementPtr img_msg) {
    
    if (f_manager.addFeatureCheckParallax(frame_count, img_msg, td+ddt))
        marginalization_flag = MARGIN_OLD;
    else
        marginalization_flag = MARGIN_SECOND_NEW;

    Times[frame_count] = img_msg->timeStampToSec;

    ImageFrame imageframe(img_msg, td+ddt);
    imageframe.pre_integration = tmp_pre_integration;
    all_image_frame.insert(make_pair(img_msg->timeStampToSec+td, imageframe));
    tmp_pre_integration.reset(new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count],
                                              acc_n, acc_w, gyr_n, gyr_w});

    if (frame_count == WINDOW_SIZE) {
        bool result = false;
        if((img_msg->timeStampToSec-initial_timestamp) > 0.1) {
            result = initialStructure();    
            initial_timestamp = img_msg->timeStampToSec;
        }
        if(result)
            bInit = true;
        else {
            slideWindow();  
        }
    } else
        frame_count++;
}


bool DynamicInitializer::initialStructure() {

    //check imu observibility
    {
        map<double, ImageFrame>::iterator frame_it;
        Vector3d sum_g;     
        for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
        {
            double dt = frame_it->second.pre_integration->sum_dt;
            Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;      
            sum_g += tmp_g;
        }
        Vector3d aver_g;   
        aver_g = sum_g * 1.0 / ((int)all_image_frame.size() - 1);  
        double var = 0;
        for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
        {
            double dt = frame_it->second.pre_integration->sum_dt;
            Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
            var += (tmp_g - aver_g).transpose() * (tmp_g - aver_g);
        }
        var = sqrt(var / ((int)all_image_frame.size() - 1));   
        if(var < 0.25)
        {
        //    printf("IMU excitation not enough!\n");  
        }
    }

    // global sfm
    Quaterniond Q[frame_count + 1];
    Vector3d T[frame_count + 1];
    map<int, Vector3d> sfm_tracked_points;
    vector<SFMFeature> sfm_f;      
    for (auto &it_per_id : f_manager.feature)
    {
        int imu_j = it_per_id.start_frame - 1;
        SFMFeature tmp_feature;
        tmp_feature.state = false;
        tmp_feature.id = it_per_id.feature_id;
        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            Vector3d pts_j = it_per_frame.point;
            tmp_feature.observation.push_back(make_pair(imu_j, Eigen::Vector2d{pts_j.x(), pts_j.y()}));
        }
        sfm_f.push_back(tmp_feature);  
    } 
    Matrix3d relative_R;
    Vector3d relative_T;
    int l;
    if (!relativePose(relative_R, relative_T, l))      
    {
        // printf("Not enough features or parallax; Move device around\n");
        return false;
    }
    GlobalSFM sfm;
    if(!sfm.construct(frame_count + 1, Q, T, l,
              relative_R, relative_T,
              sfm_f, sfm_tracked_points))      
    {                                                 
        // printf("global SFM failed!\n");
        marginalization_flag = MARGIN_OLD;
        return false;
    }

    // solve pnp for all frame
    map<double, ImageFrame>::iterator frame_it;
    map<int, Vector3d>::iterator it;
    frame_it = all_image_frame.begin();
    for (int i = 0; frame_it != all_image_frame.end( ); frame_it++)    
    {
        // provide initial guess
        cv::Mat r, rvec, t, D, tmp_r;
        if((frame_it->first) == Times[i]+td) {
            frame_it->second.is_key_frame = true;      
            frame_it->second.R = Q[i].toRotationMatrix() * RIC.transpose();     
            frame_it->second.T = T[i];                                           
            i++;
            continue;
        }   
        if((frame_it->first) > Times[i]+td)
            i++;

        Matrix3d R_inital = (Q[i].inverse()).toRotationMatrix();   
        Vector3d P_inital = - R_inital * T[i];
        cv::eigen2cv(R_inital, tmp_r);
        cv::Rodrigues(tmp_r, rvec);
        cv::eigen2cv(P_inital, t);

        frame_it->second.is_key_frame = false;     
        vector<cv::Point3f> pts_3_vector;
        vector<cv::Point2f> pts_2_vector;
        for (auto &id_pts : frame_it->second.points)   
        {
            int feature_id = id_pts.first;      
            auto i_p = id_pts.second;     
            it = sfm_tracked_points.find(feature_id);  
            if(it != sfm_tracked_points.end())
            {
                Vector3d world_pts = it->second;
                cv::Point3f pts_3(world_pts(0), world_pts(1), world_pts(2));
                pts_3_vector.push_back(pts_3);
                Vector2d img_pts = i_p.head<2>();
                cv::Point2f pts_2(img_pts(0), img_pts(1));
                pts_2_vector.push_back(pts_2);
            }
        }
        cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);     
        if(pts_3_vector.size() < 6)    
        {
            // printf("Not enough points for solve pnp !\n");
            return false; 
        }
        if (!cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1))
        {
            // printf("solve pnp fail!\n");
            return false;  
        }
        cv::Rodrigues(rvec, r);
        MatrixXd R_pnp,tmp_R_pnp;
        cv::cv2eigen(r, tmp_R_pnp);
        R_pnp = tmp_R_pnp.transpose();
        MatrixXd T_pnp;
        cv::cv2eigen(t, T_pnp);
        T_pnp = R_pnp * (-T_pnp);
        frame_it->second.R = R_pnp * RIC.transpose();
        frame_it->second.T = T_pnp;
    }

    if (visualInitialAlign())    
    {
        return true;
    }
    else
    {
//        printf("misalign visual structure with IMU\n");
        return false;
    }
}


bool DynamicInitializer::visualInitialAlign() {

    VectorXd x;
    // solve scale
    bool result = VisualIMUAlignment(all_image_frame, Bgs, g, x, TIC);  
    if(!result)
    {
        // printf("solve g failed!\n");
        return false;
    }

    // change state
    for (int i = 0; i <= frame_count; i++)
    {
        Matrix3d Ri = all_image_frame[Times[i]+td].R;
        Vector3d Pi = all_image_frame[Times[i]+td].T;
        Ps[i] = Pi;
        Rs[i] = Ri;
        all_image_frame[Times[i]+td].is_key_frame = true;
    }

    // update velocity: express it under reference camera frame
    int kv = -1;
    map<double, ImageFrame>::iterator frame_i;
    for (frame_i = all_image_frame.begin(); frame_i != all_image_frame.end(); frame_i++)   
    {
        if(frame_i->second.is_key_frame)
        {
            kv++;
            Vs[kv] = frame_i->second.R * x.segment<3>(kv * 3);
        }
    }

    // Compute intial state
    // Initialize the initial orientation, so that the estimation
    // is consistent with the inertial frame.
    Vector3d gravity_c0(g);
    double gravity_norm = gravity_c0.norm();
    Vector3d gravity_world(0.0, 0.0, -gravity_norm);

    // Set rotation
    Matrix3d R_c02w = Quaterniond::FromTwoVectors(
        gravity_c0, -gravity_world).toRotationMatrix(); 
    Matrix3d R_bl2w = R_c02w*Rs[frame_count];
    Quaterniond q0_w_i = Quaterniond(R_bl2w);
    orientation = q0_w_i.coeffs();  

    // Set other state and timestamp
    state_time = Times[frame_count]+td+ddt;
    position = Vector3d(0.0, 0.0, 0.0);
    velocity = R_c02w*Vs[frame_count];
    acc_bias = Vector3d(0.0, 0.0, 0.0);
    gyro_bias = Bgs[frame_count];

    return true;
}


bool DynamicInitializer::relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l) {

    // find previous frame which contians enough correspondance and parallex with newest frame
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        vector<pair<Vector3d, Vector3d>> corres;
        corres = f_manager.getCorresponding(i, WINDOW_SIZE);    
        if (corres.size() > 20)
        {
            double sum_parallax = 0;
            double average_parallax;
            for (int j = 0; j < int(corres.size()); j++)   
            {
                Vector2d pts_0(corres[j].first(0), corres[j].first(1));
                Vector2d pts_1(corres[j].second(0), corres[j].second(1));
                double parallax = (pts_0 - pts_1).norm();
                sum_parallax = sum_parallax + parallax;
            }
            average_parallax = 1.0 * sum_parallax / int(corres.size());    
            if(average_parallax * 460 > 30 && m_estimator.solveRelativeRT(corres, relative_R, relative_T))
            {   
                l = i;
                // printf("average_parallax %f choose l %d and newest frame to triangulate the whole structure\n", average_parallax * 460, l);
                return true;
            }
        }
    }
    return false;
}


void DynamicInitializer::slideWindow() {

    if (marginalization_flag == MARGIN_OLD)
    {
        if (frame_count == WINDOW_SIZE)    
        {
            for (int i = 0; i < WINDOW_SIZE; i++)     
            {
                Rs[i].swap(Rs[i + 1]);

                pre_integrations[i].swap(pre_integrations[i + 1]);

                dt_buf[i].swap(dt_buf[i + 1]);
                linear_acceleration_buf[i].swap(linear_acceleration_buf[i + 1]);
                angular_velocity_buf[i].swap(angular_velocity_buf[i + 1]);

                Times[i] = Times[i + 1];
                Ps[i].swap(Ps[i + 1]);
                Vs[i].swap(Vs[i + 1]);
                Bas[i].swap(Bas[i + 1]);
                Bgs[i].swap(Bgs[i + 1]);
            }
            Times[WINDOW_SIZE] = Times[WINDOW_SIZE - 1];
            Ps[WINDOW_SIZE] = Ps[WINDOW_SIZE - 1];
            Vs[WINDOW_SIZE] = Vs[WINDOW_SIZE - 1];
            Rs[WINDOW_SIZE] = Rs[WINDOW_SIZE - 1];
            Bas[WINDOW_SIZE] = Bas[WINDOW_SIZE - 1];
            Bgs[WINDOW_SIZE] = Bgs[WINDOW_SIZE - 1];

            pre_integrations[WINDOW_SIZE].reset(new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE],
                                                                acc_n, acc_w, gyr_n, gyr_w});

            dt_buf[WINDOW_SIZE].clear();
            linear_acceleration_buf[WINDOW_SIZE].clear();
            angular_velocity_buf[WINDOW_SIZE].clear();

            double t_0 = Times[0]+td;          
            map<double, ImageFrame>::iterator it_0;
            it_0 = all_image_frame.find(t_0);
            it_0->second.pre_integration.reset();                  
            all_image_frame.erase(all_image_frame.begin(), it_0);   
            
            f_manager.removeBack(); 
        }
    }
    else
    {
        if (frame_count == WINDOW_SIZE)    
        {
            for (unsigned int i = 0; i < dt_buf[frame_count].size(); i++)  
            {
                double tmp_dt = dt_buf[frame_count][i];
                Vector3d tmp_linear_acceleration = linear_acceleration_buf[frame_count][i];
                Vector3d tmp_angular_velocity = angular_velocity_buf[frame_count][i];

                pre_integrations[frame_count - 1]->push_back(tmp_dt, tmp_linear_acceleration, tmp_angular_velocity);

                dt_buf[frame_count - 1].push_back(tmp_dt);
                linear_acceleration_buf[frame_count - 1].push_back(tmp_linear_acceleration);
                angular_velocity_buf[frame_count - 1].push_back(tmp_angular_velocity);
            }

            Times[frame_count - 1] = Times[frame_count];    
            Ps[frame_count - 1] = Ps[frame_count];
            Vs[frame_count - 1] = Vs[frame_count];
            Rs[frame_count - 1] = Rs[frame_count];
            Bas[frame_count - 1] = Bas[frame_count];
            Bgs[frame_count - 1] = Bgs[frame_count];

            pre_integrations[WINDOW_SIZE].reset(new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE],
                                                                acc_n, acc_w, gyr_n, gyr_w});      

            dt_buf[WINDOW_SIZE].clear();                
            linear_acceleration_buf[WINDOW_SIZE].clear();
            angular_velocity_buf[WINDOW_SIZE].clear();

            f_manager.removeFront(frame_count);  
        }
    }
}


void DynamicInitializer::assignInitialState(std::vector<ImuData>& imu_msg_buffer,
        Eigen::Vector3d& m_gyro_old, Eigen::Vector3d& m_acc_old, IMUState& imu_state) {
    if (!bInit) {
        printf("Cannot assign initial state before initialization !!!\n");
        return;
    }

    // Remove used imu data
    int usefulImuSize = 0;
    for (const auto& imu_msg : imu_msg_buffer) {
        double imu_time = imu_msg.timeStampToSec;
        if (imu_time > state_time) break;
        usefulImuSize++;
    }

    // Earse used imu data
    imu_msg_buffer.erase(imu_msg_buffer.begin(),
        imu_msg_buffer.begin()+usefulImuSize);

    // Initialize last m_gyro and last m_acc
    m_gyro_old = last_gyro;
    m_acc_old = last_acc;

    // Set initial state
    imu_state.time = state_time;
    imu_state.gyro_bias = gyro_bias;
    imu_state.acc_bias = acc_bias;
    imu_state.orientation = orientation;
    imu_state.position = position;
    imu_state.velocity = velocity;

    for (int i = 0; i < WINDOW_SIZE + 1; i++) {
        dt_buf[i].clear();
        linear_acceleration_buf[i].clear();
        angular_velocity_buf[i].clear();
    }

    return;
}


}
