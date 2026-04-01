NVCC = nvcc
TARGETS = gaussian_blur grayscale

all: $(TARGETS)

gaussian_blur: gaussian_blur.cu stb_image.h stb_image_write.h
	$(NVCC) -o gaussian_blur gaussian_blur.cu -lm

grayscale: grayscale.cu stb_image.h stb_image_write.h
	$(NVCC) -o grayscale grayscale.cu -lm

clean:
	rm -f $(TARGETS)
