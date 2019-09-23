CXX      ?= g++
#CXXFLAGS  ?= -std=c++14 -O3 -ggdb -Wall -I../Stopwatch -fopenmp -mavx -ftree-vectorize -ftree-vectorizer-verbose=5
#LDFLAGS ?= -O3 -lfreeimageplus -lstdc++ -lm -lOpenCL -fopenmp -mavx
CXXFLAGS  ?= -std=c++17 -O0 -ggdb -Wall -fopenmp -I. -I../lib -I/usr/local/include/opencv4
LDFLAGS ?= -O0 -lstdc++ -fopenmp -L/usr/local/lib/ -lopencv_features2d -lopencv_core -lopencv_imgproc -lopencv_imgcodecs

OBJ = ../lib/corner_detector.o test.o test_corner_detector.o ../lib/depth_calculator.o test_depth_calculator.o \
	  test_exponential_map.o

OBJ = test.o test_exponential_map.o
PROGNAME = test

exec_prefix ?= /usr
bindir ?= $(exec_prefix)/bin

all: $(OBJ)
	$(CXX) -o $(PROGNAME) $(OBJ) $(LDFLAGS)
	@sh -c "./test"

install: all
	install -d $(DESTDIR)$(bindir)
	install -m 0755 $(PROGNAME) $(DESTDIR)$(bindir)

clean: $(OBJ)
	@echo "Clean object files"
	@rm -f $(OBJ)
	@rm -f $(PROGNAME)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c $<
