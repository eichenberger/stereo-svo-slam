.PHONY: test clean

all:
	python3 setup.py build_ext --inplace

test:
	$(MAKE) -C test all

clean:
	python3 setup.py clean --all
	rm slam_accelerator*.so
