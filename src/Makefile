CXX      ?= gcc
CXXFLAGS  ?= -O3 #-fopt-info-vec-missed -fopt-info-vec # -fopt-info-vec-all
LDFLAGS ?= -O3 

#CXXFLAGS  ?= -O0 -ggdb
#LDFLAGS ?= -O0 -ggdb

OPENCV_INC_DIR ?= /usr/local/include/opencv4
OPENCV_LIB_DIR ?= /usr/local/lib
QT_INC_DIR ?= /usr/include/x86_64-linux-gnu/qt5
#OMP_LIB ?= omp5

QMAKE ?= qmake

.PHONY: all app ar-app lib qt-viewer python test clean doc 

export

all: lib app ar-app

lib:
	$(MAKE) -C lib lib

python: lib
	$(MAKE) -C python all

app: lib 
	$(MAKE) -C app

ar-app: lib
	cd ar-app && $(QMAKE)
	$(MAKE) -C ar-app

qt-viewer:
	cd qt-viewer && $(QMAKE)
	$(MAKE) -C qt-viewer

test:
	$(MAKE) -C test all

doc:
	doxygen

clean:
	$(MAKE) -C lib clean
	$(MAKE) -C test clean
	$(MAKE) -C app clean
	$(MAKE) -C ar-app clean
	$(MAKE) -C qt-viewer clean
