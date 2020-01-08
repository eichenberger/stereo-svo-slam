FROM ubuntu:14.04

RUN apt-get -y update && apt-get -y install git libopencv-dev libeigen3-dev libboost-thread-dev libboost-system-dev cmake
RUN apt-get -y update && apt-get -y install libgl1-mesa-dev libgles2-mesa-dev libglu1-mesa-dev mesa-common-dev
RUN apt-get -y update && apt-get -y install libglew-dev
RUN apt-get -y update && apt-get -y install libxkbcommon-dev
RUN apt-get -y update && apt-get -y install wget
RUN apt-get -y remove cmake
RUN apt-get -y update && apt-get -y install libssl-dev
RUN wget https://github.com/Kitware/CMake/releases/download/v3.16.2/cmake-3.16.2.tar.gz && tar xf cmake-3.16.2.tar.gz && cd cmake-3.16.2 && ./configure && make && make install

RUN mkdir /svo
RUN cd /svo && git clone https://github.com/stevenlovegrove/Pangolin.git -b v0.5 && cd Pangolin && mkdir build && cd build && cmake .. && make -j && make install 
RUN cd /svo && git clone https://github.com/uzh-rpg/fast.git && cd fast && mkdir build && cd build && cmake .. && make -j && make install
RUN cd /svo && git clone https://github.com/strasdat/Sophus.git -b master && cd Sophus && git checkout e656ebcd3441d7d89fe6380a093df2cd7ee441c5 && mkdir build && cd build && cmake .. && make -j && make install
RUN cd /svo && git clone https://github.com/uzh-rpg/rpg_vikit.git && cd rpg_vikit/vikit_common && sed -i 's/USE_ROS TRUE/USE_ROS FALSE/g' CMakeLists.txt && mkdir build && cd build && cmake -DUSE_ROS=FALSE .. && make -j && make install
RUN cd /svo && git clone https://github.com/uzh-rpg/rpg_svo.git && cd rpg_svo/svo && sed -i 's/USE_ROS TRUE/USE_ROS FALSE/g' CMakeLists.txt && mkdir build && cd build && cmake .. && make -j

RUN mkdir /orb && cd /orb && git clone https://github.com/raulmur/ORB_SLAM2.git && cd ORB_SLAM2 && ./build.sh

CMD /bin/bash
