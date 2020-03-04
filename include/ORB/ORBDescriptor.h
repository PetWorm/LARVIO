//
// Created by xiaochen at 18-8-27.
// Functions for calculate ORB descriptor of an arbitrary image point.
// Stems from ORB_SLAM2: ORBextractor.cc and opencv/orb.cpp
//

#ifndef LKTTRACKER_ORBDESCRIPTOR_H
#define LKTTRACKER_ORBDESCRIPTOR_H

// #include <opencv/cv.h>
#include <opencv2/opencv.hpp>
#include <vector>

using namespace std;

namespace larvio {

class ORBdescriptor     // this class is created by QXC based on ORB_SLAM2::ORBextractor
{
public:

    ORBdescriptor(const cv::Mat& _image, float _scaleFactor, int _nlevels,
                  int _edgeThreshold=31, int _patchSize=31);

    ~ORBdescriptor(){}

    // modified based on ORBextractor::computeDescriptors by QXC
    // @INPUTs:
    //      @pts: cv::Point object of key points in source image, i.e. image at level 0
    //      @levels: level of @image in pyramid
    // @OUTPUTs
    //      @descriptors: each row store the descriptor of corresponding key point
    // @BRIEF:
    //      This function calculate the descriptors of keypoints of same level.
    //      Should be called after calling function initializeLayerAndPyramid().
    bool computeDescriptors(const vector<cv::Point2f>& pts,
                            const vector<int>& levels,
                            cv::Mat& descriptors);

    // copy from ORBmatcher.cc, below is original comment
    // Bit set count operation from
    // http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel
    static int computeDescriptorDistance(const cv::Mat &a, const cv::Mat &b)
    {
        const int *pa = a.ptr<int32_t>();
        const int *pb = b.ptr<int32_t>();

        int dist=0;

        for(int i=0; i<8; i++, pa++, pb++)
        {
            unsigned int v = *pa ^ *pb;		
            v = v - ((v >> 1) & 0x55555555);
            v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
            dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
        }

        return dist;
    }

    static const float factorPI;

    static const int TH_HIGH;   // threshold for worst descriptor distance in all levels
    static const int TH_LOW;    // threshold for best descriptor distance in all levels

private:

    static const int HARRIS_BLOCK_SIZE;

    int patchSize;
    int halfPatchSize;
    int edgeThreshold;

    double scaleFactor;
    int nlevels;

    vector<cv::Point> pattern;
    vector<int> umax;

    vector<float> mvScaleFactor;
    vector<float> mvInvScaleFactor;
    vector<cv::Rect> mvLayerInfo;
    vector<float> mvLayerScale;
    cv::Mat mImagePyramid;
    cv::Mat mBluredImagePyramid;

private:
// compute descriptor of a key point and save it into @desc
    void computeOrbDescriptor(const cv::KeyPoint& kpt,
                              uchar* desc);

    // copy from opencv/orb.cpp
    void makeRandomPattern(int patchSize, cv::Point* pattern_, int npoints);

    // initialize layer information and image of pyramid
    void initializeLayerAndPyramid(const cv::Mat& image);

public:
    // calculate Angle for an ordinary point
    float IC_Angle(const int& levels, const cv::Point2f& pt);
};

}


#endif //LKTTRACKER_ORBDESCRIPTOR_H