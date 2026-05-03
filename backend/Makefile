NVCC = nvcc
CXX_OPTS = -O3 -march=native -ffast-math -fopenmp
NVCC_FLAGS = -O3 -Xcompiler "$(CXX_OPTS)" -lgomp
TARGETS = bin/gaussian_blur bin/grayscale bin/sobel bin/median bin/kuwahara

all: bin $(TARGETS)

bin:
	mkdir -p bin

bin/gaussian_blur: gaussian_blur.cu stb_image.h stb_image_write.h
	$(NVCC) $(NVCC_FLAGS) -o bin/gaussian_blur gaussian_blur.cu -lm

bin/grayscale: grayscale.cu stb_image.h stb_image_write.h
	$(NVCC) $(NVCC_FLAGS) -o bin/grayscale grayscale.cu -lm

bin/sobel: sobel.cu stb_image.h stb_image_write.h
	$(NVCC) $(NVCC_FLAGS) -o bin/sobel sobel.cu -lm

bin/median: median.cu stb_image.h stb_image_write.h
	$(NVCC) $(NVCC_FLAGS) -o bin/median median.cu -lm

bin/kuwahara: kuwahara.cu stb_image.h stb_image_write.h
	$(NVCC) $(NVCC_FLAGS) -o bin/kuwahara kuwahara.cu -lm

clean:
	rm -rf bin
