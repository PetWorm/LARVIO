FROM ros:melodic-perception
MAINTAINER QIU Xiaochen <ares43490@126.com>

ENV CERES_VERSION="1.14.0"
ENV JOBS_NUM="4"

RUN	apt-get -y update && apt-get install -y \
	libboost-all-dev \
	cmake git \
	libgoogle-glog-dev \
	libeigen3-dev \
	libsuitesparse-dev \
	libgl1-mesa-dev \
	libglew-dev \
	libxkbcommon-x11-dev \
	ros-${ROS_DISTRO}-tf-conversions \
	ros-${ROS_DISTRO}-rviz && \
	rm -rf /var/lib/apt/lists/*

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
	cmake -D CMAKE_BUILD_TYPE=Release .. && make -j${JOBS_NUM} && \
	cd ../ros_wrapper && . /opt/ros/${ROS_DISTRO}/setup.sh && \
	catkin_make -j${JOBS_NUM}