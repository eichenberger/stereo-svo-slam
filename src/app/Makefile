CXXFLAGS  ?= -O0 -ggdb
LDFLAGS ?= -O0

SRC_FILES = main.cpp websocketserver.cpp svo_slam_backend.cpp image_input.cpp \
			econ_input.cpp video_input.cpp slam_app.cpp

OBJ_FILES = $(patsubst %.cpp, %.o, $(SRC_FILES))

IMU_FILES =test_imu.cpp econ_input.cpp image_input.cpp 
IMU_OBJ_FILES = $(patsubst %.cpp, %.o, $(IMU_FILES))

PROG_NAME = slam_app

INCLUDEDIRS = -I../include -I$(OPENCV_INC_DIR) \
			  -I$(QT_INC_DIR)

LIBDIRS = -L$(OPENCV_LIB_DIR) -L../lib
LIBS = -lstereosvo -lopencv_core -lopencv_videoio -lopencv_imgproc \
	   -lopencv_highgui -lQt5WebSockets -lQt5Core -lQt5Network -lQt5Widgets
CXXFLAGS += -std=c++17 -pedantic -Wall -fopenmp -fPIC $(INCLUDEDIRS)
LDFLAGS += -fPIC $(LIBDIRS) $(LIBS) -pthread -Wl,-rpath,$(OPENCV_LIB_DIR),-rpath,../lib/,-rpath,.

all: $(OBJ_FILES) test_imu
	$(CXX) $(LDFLAGS) -o $(PROG_NAME) $(OBJ_FILES) $(LDFLAGS)

test_imu: $(IMU_OBJ_FILES)
	$(CXX) -o test_imu $(IMU_OBJ_FILES) $(LDFLAGS)

clean:
	rm -f $(LIB_NAME) $(OBJ_FILES) $(IMU_OBJ_FILES)
	rm -f *.gch *.tmp
	rm -f $(PROG_NAME) test_imu

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

