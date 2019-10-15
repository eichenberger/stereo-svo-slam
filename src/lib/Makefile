CXX      ?= gcc
CXXFLAGS  ?= -O0 -ggdb
LDFLAGS ?= -O0

SRC_FILES = corner_detector.cpp \
			depth_calculator.cpp \
			image_comparison.cpp \
			keyframe_inserter.cpp \
			optical_flow.cpp \
			pose_estimator.cpp \
			pose_refinement.cpp \
			rotation_matrix.cpp \
			stereo_slam.cpp \
			transform_keypoints.cpp


OBJ_FILES = $(patsubst %.cpp, %.o, $(SRC_FILES))

LIB_NAME = libstereosvo.so

INCLUDEDIRS = -I../include -I/usr/local/include/opencv4

LIBDIRS = -L/usr/local/lib
LIBS = -lm -lopencv_core -lomp5 -lopencv_features2d -lopencv_imgproc -lopencv_calib3d -lopencv_video -lopencv_highgui

CXXFLAGS += -std=c++17 -pedantic -Wall -fopenmp -fPIC $(INCLUDEDIRS)
LDFLAGS += -fPIC $(LIBDIRS) $(LIBS) -Wl,-rpath,/usr/local/lib,-rpath,.

all: lib

lib: $(OBJ_FILES)
	@OBJ_FILES=$(echo $(SRC_FILES) |sed 's/\.cpp/.o/g" )
	$(CXX) $(LDFLAGS) -shared -o $(LIB_NAME) $(OBJ_FILES) $(LDFLAGS)

clean:
	rm -f $(LIB_NAME) $(OBJ_FILES)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

