.PHONY: clean

all:
	python3 setup.py build_ext --inplace

clean:
	rm -f *.so
	rm -rf build
	rm -rf __pycache__
	rm wrapper/*.cpp

