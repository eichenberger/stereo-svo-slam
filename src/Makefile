.PHONY: test clean

all:
	python3 setup.py build_ext --inplace

test:
	$(MAKE) -C test all

clean:
	python3 setup.py clean --all
	rm -f slam_accelerator*.so
	rm -f lib/*.gch
	rm -f lib/*.o
	rm -f lib/slam_accelerator.cpp
