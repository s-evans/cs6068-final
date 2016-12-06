.SUFFIXES:

NVCC=nvcc

NVCC_OPTS=-O3 -arch=sm_20 -m64

GCC_OPTS=-O3 -m64

CUDA_INCLUDEPATH=/usr/local/cuda/6.5.14/include

SOURCES=main.cpp

CUDA_SOURCES=parallel_huffman.cu\
			 huffman_tree.cu

OBJECTS=$(SOURCES:.cpp=.o)

CUDA_OBJECTS=$(addsuffix .o,$(CUDA_SOURCES))

DEPENDENCIES=$(SOURCES:.cpp=.d) $(CUDA_SOURCES:.cu=.d)

.PHONY: all clean

all: encode

encode: $(CUDA_OBJECTS) $(OBJECTS)
	$(NVCC) -o $@ $^ $(NVCC_OPTS) -lboost_system -lboost_program_options -lboost_filesystem -lboost_iostreams

%.cu.o: %.cu Makefile
	$(NVCC) -o $(<:.cu=.d) $< $(NVCC_OPTS) -M &
	$(NVCC) -o $@ -c $< $(NVCC_OPTS)

%.o: %.cpp Makefile
	g++ -c $< $(GCC_OPTS) -I $(CUDA_INCLUDEPATH) -MD

clean:
	rm -f encode $(OBJECTS) $(CUDA_OBJECTS) $(DEPENDENCIES)

-include $(DEPENDENCIES)
