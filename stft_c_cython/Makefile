all:
	cython stft.pyx
	gcc -g -O3 -fpic -c stft.c `python-config --cflags`  -std=c99 -lm -lfftw3
	gcc stft.o -o stft.so -shared `python-config --ldflags` -std=c99 -lm -lfftw3
