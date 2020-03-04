/*
 * @Descripttion: 
 * @Author: Xiaochen Qiu
 */

#ifndef VISUALIZE_H
#define VISUALIZE_H

#include <pangolin/pangolin.h>
#include <Eigen/Geometry>
#include <iostream>

#include "larvio/imu_state.h"


namespace larvio {

namespace visualize {


void DrawCurrentPose(pangolin::OpenGlMatrix &Twc) {
    const float &w = 0.08;
    const float h = w*0.75;
    const float z = w*0.6;

    glPushMatrix();

#ifdef HAVE_GLES
        glMultMatrixf(Twc.m);
#else
        glMultMatrixd(Twc.m);
#endif

    glLineWidth(3);
    glColor3f(0.0f,0.0f,1.0f);
    glBegin(GL_LINES);
    glVertex3f(0,0,0);
    glVertex3f(w,h,z);
    glVertex3f(0,0,0);
    glVertex3f(w,-h,z);
    glVertex3f(0,0,0);
    glVertex3f(-w,-h,z);
    glVertex3f(0,0,0);
    glVertex3f(-w,h,z);

    glVertex3f(w,h,z);
    glVertex3f(w,-h,z);

    glVertex3f(-w,h,z);
    glVertex3f(-w,-h,z);

    glVertex3f(-w,h,z);
    glVertex3f(w,h,z);

    glVertex3f(-w,-h,z);
    glVertex3f(w,-h,z);
    glEnd();

    glPopMatrix();
}


void DrawKeyFrames(const std::vector<pangolin::OpenGlMatrix>& poses) {
    const float &w = 0.05;
    const float h = w*0.75;
    const float z = w*0.6;

    for(auto pose : poses) {
        glPushMatrix();

#ifdef HAVE_GLES
        glMultMatrixf(pose.m);
#else
        glMultMatrixd(pose.m);
#endif

        glLineWidth(1);
        glColor3f(1.0f,1.0f,0.0f);
        glBegin(GL_LINES);
        glVertex3f(0,0,0);
        glVertex3f(w,h,z);
        glVertex3f(0,0,0);
        glVertex3f(w,-h,z);
        glVertex3f(0,0,0);
        glVertex3f(-w,-h,z);
        glVertex3f(0,0,0);
        glVertex3f(-w,h,z);

        glVertex3f(w,h,z);
        glVertex3f(w,-h,z);

        glVertex3f(-w,h,z);
        glVertex3f(-w,-h,z);

        glVertex3f(-w,h,z);
        glVertex3f(w,h,z);

        glVertex3f(-w,-h,z);
        glVertex3f(w,-h,z);
        glEnd();

        glPopMatrix();
    }
}


void GetCurrentOpenGLPoseMatrix(pangolin::OpenGlMatrix &M, const Eigen::Isometry3d &Tbw) {
    Eigen::Matrix3d R = Tbw.rotation();
    Eigen::Vector3d t = Tbw.translation();

    M.m[0] = R(0,0);
    M.m[1] = R(1,0);
    M.m[2] = R(2,0);
    M.m[3]  = 0.0;

    M.m[4] = R(0,1);
    M.m[5] = R(1,1);
    M.m[6] = R(2,1);
    M.m[7]  = 0.0;

    M.m[8] = R(0,2);
    M.m[9] = R(1,2);
    M.m[10] = R(2,2);
    M.m[11]  = 0.0;

    M.m[12] = t(0);
    M.m[13] = t(1);
    M.m[14] = t(2);
    M.m[15]  = 1.0;
}


void GetSwOpenGLPoseMatrices(std::vector<pangolin::OpenGlMatrix>& vPoses_SW, const std::vector<Eigen::Isometry3d>& swPoses) {
    vPoses_SW.clear();
    for (auto pose : swPoses) {
        pangolin::OpenGlMatrix M;
        GetCurrentOpenGLPoseMatrix(M, pose);
        vPoses_SW.push_back(M);
    }
}


void DrawSlideWindow(const std::vector<pangolin::OpenGlMatrix>& vPoses_SW) {
    const float &w = 0.05;
    const float h = w*0.75;
    const float z = w*0.6;

    for (auto pose : vPoses_SW) {
        glPushMatrix();

#ifdef HAVE_GLES
        glMultMatrixf(pose.m);
#else
        glMultMatrixd(pose.m);
#endif

        glLineWidth(1);
        glColor3f(1.0f,0.0f,0.0f);
        glBegin(GL_LINES);
        glVertex3f(0,0,0);
        glVertex3f(w,h,z);
        glVertex3f(0,0,0);
        glVertex3f(w,-h,z);
        glVertex3f(0,0,0);
        glVertex3f(-w,-h,z);
        glVertex3f(0,0,0);
        glVertex3f(-w,h,z);

        glVertex3f(w,h,z);
        glVertex3f(w,-h,z);

        glVertex3f(-w,h,z);
        glVertex3f(-w,-h,z);

        glVertex3f(-w,h,z);
        glVertex3f(w,h,z);

        glVertex3f(-w,-h,z);
        glVertex3f(w,-h,z);
        glEnd();

        glPopMatrix();
    }
}


void DrawActiveMapPoints(const std::map<larvio::FeatureIDType,Eigen::Vector3d>& mMapPoints) {
    if (mMapPoints.empty())
        return;

    glPointSize(10);
    glBegin(GL_POINTS);
    glColor3f(1.0,0.0,0.0);

    for (auto pt : mMapPoints)
        glVertex3f((pt.second)(0),(pt.second)(1),(pt.second)(2));

    glEnd();
}


void DrawStableMapPoints(const std::map<larvio::FeatureIDType,Eigen::Vector3d>& mMapPoints) {
    if (mMapPoints.empty())
        return;

    glPointSize(3);
    glBegin(GL_POINTS);
    glColor3f(1.0,1.0,1.0);

    for (auto pt : mMapPoints)
        glVertex3f((pt.second)(0),(pt.second)(1),(pt.second)(2));

    glEnd();
}


};

};

#endif