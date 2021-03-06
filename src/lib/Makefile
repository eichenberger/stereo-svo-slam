CXX      ?= gcc
CXXFLAGS  ?= -O0 -ggdb
LDFLAGS ?= -O0

SRC_FILES = corner_detector.cpp \
			depth_calculator.cpp \
			image_comparison.cpp \
			keyframe_manager.cpp \
			optical_flow.cpp \
			pose_estimator.cpp \
			pose_refinement.cpp \
			pose_manager.cpp \
			rotation_matrix.cpp \
			stereo_slam.cpp \
			transform_keypoints.cpp \
			depth_filter.cpp


OBJ_FILES = $(patsubst %.cpp, %.o, $(SRC_FILES))

LIB_NAME = libstereosvo.so

INCLUDEDIRS = -I../include -I$(OPENCV_INC_DIR)

LIBDIRS = -L$(OPENCV_LIB_DIR)
LIBS = -lm -lopencv_core -lopencv_features2d -lopencv_imgproc -lopencv_calib3d -lopencv_video -lopencv_highgui

CXXFLAGS += -std=c++17 -pedantic -Wall -fPIC -fopenmp $(INCLUDEDIRS)
LDFLAGS += -fPIC -fopenmp $(LIBDIRS) $(LIBS) -Wl,-rpath,$(OPENCV_LIB_DIR),-rpath,.

all: lib

lib: $(OBJ_FILES)
	@OBJ_FILES=$(echo $(SRC_FILES) |sed 's/\.cpp/.o/g" )
	$(CXX) $(LDFLAGS) -shared -o $(LIB_NAME) $(OBJ_FILES) $(LDFLAGS)

clean:
	rm -f $(LIB_NAME) $(OBJ_FILES)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

