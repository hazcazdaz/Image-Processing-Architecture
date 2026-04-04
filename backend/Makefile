NVCC = nvcc
TARGETS = bin/gaussian_blur bin/grayscale bin/sobel

all: bin $(TARGETS)

bin:
	mkdir -p bin

bin/gaussian_blur: gaussian_blur.cu stb_image.h stb_image_write.h
	$(NVCC) -o bin/gaussian_blur gaussian_blur.cu -lm

bin/grayscale: grayscale.cu stb_image.h stb_image_write.h
	$(NVCC) -o bin/grayscale grayscale.cu -lm

bin/sobel: sobel.cu stb_image.h stb_image_write.h
	$(NVCC) -o bin/sobel sobel.cu -lm

clean:
	rm -rf bin