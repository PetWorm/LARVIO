FROM ubuntu:18.04
MAINTAINER QIU Xiaochen <ares43490@126.com>

ENV CERES_VERSION="1.14.0"
ENV TIME_ZONE="Asia/Shanghai"
ENV JOBS_NUM="4"

RUN	ln -snf /usr/share/zoneinfo/${TIME_ZONE} /etc/localtime && \
	echo ${TIME_ZONE} > /etc/timezone

RUN	apt-get -y update && apt-get install -y \
	libopencv-dev \
	libboost-all-dev \
	cmake wget git \
	libgoogle-glog-dev \
	libeigen3-dev \
	libsuitesparse-dev \
	libgl1-mesa-dev \
	libglew-dev

RUN	cd /root && \
	git clone https://github.com/ceres-solver/ceres-solver.git && \
	cd ceres-solver && git checkout tags/${CERES_VERSION} && \
	mkdir build && cd build && \
	cmake .. && make -j${JOBS_NUM} install && \
	cd /root && \
	git clone https://github.com/stevenlovegrove/Pangolin.git && \
	cd Pangolin && mkdir build && cd build && \
	cmake .. && make -j${JOBS_NUM} install && \
	rm -rf /root/*

RUN	cd /root && \
	git clone https://github.com/PetWorm/LARVIO.git && \
	cd LARVIO && mkdir build && cd build && \
	cmake -D CMAKE_BUILD_TYPE=Release .. && make -j${JOBS_NUM}