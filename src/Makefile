CXX      ?= gcc
CXXFLAGS  ?= -std=c++11 -pedantic -Wall
LDFLAGS ?=

.PHONY: test clean lib python

all: lib python app


lib: 
	$(MAKE) -C lib lib

python: lib
	$(MAKE) -C python all

app: lib
	$(MAKE) -C app

test:
	$(MAKE) -C test all

clean:
	$(MAKE) -C lib clean
	$(MAKE) -C python clean
	$(MAKE) -C test clean
	$(MAKE) -C app clean
