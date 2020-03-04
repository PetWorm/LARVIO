/*
 * @Descripttion: This header include functions and types for reading IMU and image data, and methods to manipulate.
 * @Author: Xiaochen Qiu
 */


#ifndef DATA_READER_H
#define DATA_READER_H


#include <iostream>
#include <fstream>
#include <vector>

#include "sensors/ImuData.hpp"

using namespace std;

namespace larvio {

struct ImgInfo {
	double timeStampToSec;
	string imgName;
};

/**
 * @description: Read data.csv containing image file names
 * @param imagePath Path of data.csv
 * @return: iListData Cotaining image informations.
 */
void loadImageList(char* imagePath, vector<ImgInfo> &iListData) {
    ifstream inf;
    inf.open(imagePath, ifstream::in);
    const int cnt = 2;         
    string line;
    int j = 0;
    size_t comma = 0;
    size_t comma2 = 0;
    ImgInfo temp;

    getline(inf,line);	
    while (!inf.eof()) {
        getline(inf,line);

        comma = line.find(',',0);	
		temp.timeStampToSec = 1e-9*atol(line.substr(0,comma).c_str());		
        
        while (comma < line.size() && j != cnt-1) {
            comma2 = line.find(',',comma + 1);	
            temp.imgName = line.substr(comma + 1,comma2-comma-1).c_str();
            ++j;
            comma = comma2;
        }

        iListData.push_back(temp);
        j = 0;
    }

    inf.close();
}

/**
 * @description: Read data.csv containing IMU data
 * @param imuPath Path of data.csv
 * @return: vimuData Cotaining IMU informations.
 */
// QXC：读取imu的data.csv文件
void loadImuFile(char* imuPath, vector<ImuData> &vimuData) {
    ifstream inf;
    inf.open(imuPath, ifstream::in);
    const int cnt = 7;        
    string line;
    int j = 0;
    size_t comma = 0;
    size_t comma2 = 0;
    char imuTime[14] = {0};
    double acc[3] = {0.0};
    double grad[3] = {0.0};
    double imuTimeStamp = 0;

    getline(inf,line);		
    while (!inf.eof()) {
        getline(inf,line);

        comma = line.find(',',0);
		string temp = line.substr(0,comma);
		imuTimeStamp = 1e-9*atol(line.substr(0,comma).c_str());	
        
        while (comma < line.size() && j != cnt-1) {
            comma2 = line.find(',',comma + 1);
            switch(j) {
			case 0:
				grad[0] = atof(line.substr(comma + 1,comma2-comma-1).c_str());
				break;
			case 1:
				grad[1] = atof(line.substr(comma + 1,comma2-comma-1).c_str());
				break;
			case 2:
				grad[2] = atof(line.substr(comma + 1,comma2-comma-1).c_str());
				break;
			case 3:
				acc[0] = atof(line.substr(comma + 1,comma2-comma-1).c_str());
				break;
			case 4:
				acc[1] = atof(line.substr(comma + 1,comma2-comma-1).c_str());
				break;
			case 5:
				acc[2] = atof(line.substr(comma + 1,comma2-comma-1).c_str());	
				break;
            }
            ++j;
            comma = comma2;
        }
		ImuData tempImu(imuTimeStamp, grad[0], grad[1], grad[2], acc[0], acc[1], acc[2]);
        vimuData.push_back(tempImu);
        j = 0;
    }

	inf.close();
}


bool findFirstAlign(const vector<ImuData> &vImu, const vector<ImgInfo> &vImg, pair<int,int> &ImgImuAlign) {
	double imuTime0 = vImu[0].timeStampToSec;
	double imgTime0 = vImg[0].timeStampToSec;
	
	if(imuTime0>imgTime0) {		
		for(size_t i=1; i<vImg.size(); i++) {
			double imgTime = vImg[i].timeStampToSec;
			if(imuTime0<=imgTime) {
				for(size_t j=0; j<vImu.size(); j++) {
					double imuTime = vImu[j].timeStampToSec;
					if(imuTime==imgTime) {
						int imgID = i;
						int imuID = j;
						ImgImuAlign = make_pair(imgID,imuID);
						return true;
					}
				}
				return false;
			}
		}
		return false;
	}
	else if(imuTime0<imgTime0) {	
		for(size_t i=1; i<vImu.size(); i++) {
			double imuTime = vImu[i].timeStampToSec;
			if(imuTime==imgTime0) {
				int imgID = 0;
				int imuID = i;
				ImgImuAlign = make_pair(imgID,imuID);
				return true;
			}
		}
		return false;
	}
	else {						
		int imgID = 0;
		int imuID = 0;
		ImgImuAlign = make_pair(imgID,imuID);
		return true;
	}
}

}


#endif // DATA_READER_H