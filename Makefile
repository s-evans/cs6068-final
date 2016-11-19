NVCC=nvcc

NVCC_OPTS=-O3 -arch=sm_20 -m64

GCC_OPTS=-O3 -m64

OPENCV_LIBPATH=/users/PES0721/ucn2523/local/opencv/2.4.11/lib

OPENCV_INCLUDEPATH=/users/PES0721/ucn2523/local/opencv/2.4.11/include

CUDA_INCLUDEPATH=/usr/local/cuda/6.5.14/include

OPENCV_LIBS=-lopencv_core -lopencv_imgproc -lopencv_highgui

SOURCES=compare.cpp\
		loadSaveImage.cpp\
		main.cpp\
		reference_calc.cpp

CUDA_SOURCES=kernels.cu

OBJECTS=$(SOURCES:.cpp=.o)

CUDA_OBJECTS=$(addsuffix .o,$(CUDA_SOURCES))

DEPENDENCIES=$(SOURCES:.cpp=.d) $(CUDA_SOURCES:.cu=.d)

.PHONY: all clean

all: compress

compress: $(CUDA_OBJECTS) $(OBJECTS)
	$(NVCC) -o $@ $^ -L $(OPENCV_LIBPATH) $(OPENCV_LIBS) $(NVCC_OPTS)

%.cu.o: %.cu Makefile
	$(NVCC) -o $(<:.cu=.d) $< $(NVCC_OPTS) -M
	$(NVCC) -o $@ -c $< $(NVCC_OPTS)

%.o: %.cpp Makefile
	g++ -c $< $(GCC_OPTS) -I $(CUDA_INCLUDEPATH) -I $(OPENCV_INCLUDEPATH) -MD

clean:
	rm -f compress $(OBJECTS) $(CUDA_OBJECTS) $(DEPENDENCIES)

-include $(DEPENDENCIES)
