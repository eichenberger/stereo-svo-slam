CXX      ?= gcc
CXXFLAGS  ?= -std=c++11 -pedantic -Wall
LDFLAGS ?=

.PHONY: test clean lib python

all: lib python

python:
	$(MAKE) -C python all

lib: 
	$(MAKE) -C lib lib

test:
	$(MAKE) -C test all

clean:
	$(MAKE) -C lib clean
	$(MAKE) -C python clean
	$(MAKE) -C test clean
