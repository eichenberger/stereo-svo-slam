CXX      ?= gcc
CXXFLAGS  ?= -O0 -ggdb
LDFLAGS ?= -O0

SRC_FILES = main.cpp websocketserver.cpp svo_slam_backend.cpp

OBJ_FILES = $(patsubst %.cpp, %.o, $(SRC_FILES))

PROG_NAME = slam_app

INCLUDEDIRS = -I../include -I/usr/local/include/opencv4 \
			  -I/usr/include/x86_64-linux-gnu/qt5

LIBDIRS = -L/usr/local/lib -L../lib
LIBS = -lstereosvo -lopencv_core -lopencv_videoio -lopencv_imgproc \
	   -lopencv_highgui -lQt5WebSockets -lQt5Core -lQt5Network -lQt5Widgets
CXXFLAGS += -std=c++17 -pedantic -Wall -fopenmp -fPIC $(INCLUDEDIRS)
LDFLAGS += -fPIC $(LIBDIRS) $(LIBS) -Wl,-rpath,/usr/local/lib,-rpath,../lib/,-rpath,.

all: $(OBJ_FILES)
	$(CXX) $(LDFLAGS) -o $(PROG_NAME) $(OBJ_FILES) $(LDFLAGS)

clean:
	rm -f $(LIB_NAME) $(OBJ_FILES)
	rm -f $(PROG_NAME)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

