//这是用c语言编写的cnn神经网络识别minist中0-9手写数字的程序
//编程软件为vs2022，未使用其他内置库
//其中卷积层，池化层，全连接为单独代码函数。其余实现过程借鉴了网络开源资源
//在c语言项目中，没有实现反向传播的功能，正确率只能达到54%

#define _CRT_SECURE_NO_WARNINGS 1
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define IMG_SIZE 28
#define NUM_CLASSES 10
#define NUM_TRAIN_IMAGES 60000
#define NUM_TEST_IMAGES 10000

// 加载MNIST图像数据
void load_images(const char* filename, unsigned char* images, int num_images) {
	FILE* file = fopen(filename, "rb");
	if (file == NULL) {
		perror("Failed to open image file");
		exit(1);
	}
	int magic_number, num, rows, cols;
	fread(&magic_number, sizeof(int), 1, file);
	fread(&num, sizeof(int), 1, file);
	fread(&rows, sizeof(int), 1, file);
	fread(&cols, sizeof(int), 1, file);
	magic_number = __builtin_bswap32(magic_number);
	num = __builtin_bswap32(num);
	rows = __builtin_bswap32(rows);
	cols = __builtin_bswap32(cols);
	fread(images, sizeof(unsigned char), num_images * IMG_SIZE * IMG_SIZE, file);
	fclose(file);
}

// 加载MNIST标签数据
void load_labels(const char* filename, unsigned char* labels, int num_labels) {
	FILE* file = fopen(filename, "rb");
	if (file == NULL) {
		perror("Failed to open label file");
		exit(1);
	}
	int magic_number, num;
	fread(&magic_number, sizeof(int), 1, file);
	fread(&num, sizeof(int), 1, file);
	magic_number = __builtin_bswap32(magic_number);
	num = __builtin_bswap32(num);
	fread(labels, sizeof(unsigned char), num_labels, file);
	fclose(file);
}

// 卷积操作
void conv2d(const unsigned char* input, float* kernel, float* output, int input_size, int kernel_size, int output_size) {
	for (int i = 0; i < output_size; i++) {
		for (int j = 0; j < output_size; j++) {
			output[i * output_size + j] = 0;
			for (int k = 0; k < kernel_size; k++) {
				for (int l = 0; l < kernel_size; l++) {
					output[i * output_size + j] += input[(i + k) * input_size + (j + l)] * kernel[k * kernel_size + l];
				}
			}
		}
	}
}

// 池化操作（最大池化）
void max_pooling(float* input, float* output, int input_size, int pool_size, int output_size) {
	for (int i = 0; i < output_size; i++) {
		for (int j = 0; j < output_size; j++) {
			float max_val = -INFINITY;
			for (int k = 0; k < pool_size; k++) {
				for (int l = 0; l < pool_size; l++) {
					float val = input[(i * pool_size + k) * input_size + (j * pool_size + l)];
					if (val > max_val) {
						max_val = val;
					}
				}
			}
			output[i * output_size + j] = max_val;
		}
	}
}

// 全连接层
void fully_connected(float* input, float* weights, float* output, int input_size, int output_size) {
	for (int i = 0; i < output_size; i++) {
		output[i] = 0;
		for (int j = 0; j < input_size; j++) {
			output[i] += input[j] * weights[i * input_size + j];
		}
	}
}

// 预测函数
int predict(const unsigned char* image, float* conv_kernel, float* fc_weights) {
	int conv_output_size = IMG_SIZE - 3 + 1;
	float* conv_output = (float*)malloc(conv_output_size * conv_output_size * sizeof(float));
	conv2d(image, conv_kernel, conv_output, IMG_SIZE, 3, conv_output_size);

	int pool_output_size = conv_output_size / 2;
	float* pool_output = (float*)malloc(pool_output_size * pool_output_size * sizeof(float));
	max_pooling(conv_output, pool_output, conv_output_size, 2, pool_output_size);

	int fc_input_size = pool_output_size * pool_output_size;
	float* fc_output = (float*)malloc(NUM_CLASSES * sizeof(float));
	fully_connected(pool_output, fc_weights, fc_output, fc_input_size, NUM_CLASSES);

	int predicted_label = 0;
	float max_score = fc_output[0];
	for (int i = 1; i < NUM_CLASSES; i++) {
		if (fc_output[i] > max_score) {
			max_score = fc_output[i];
			predicted_label = i;
		}
	}

	free(conv_output);
	free(pool_output);
	free(fc_output);

	return predicted_label;
}

int main() {

	srand((unsigned int)time(NULL));

	unsigned char* train_images = (unsigned char*)malloc(NUM_TRAIN_IMAGES * IMG_SIZE * IMG_SIZE * sizeof(unsigned char));
	unsigned char* train_labels = (unsigned char*)malloc(NUM_TRAIN_IMAGES * sizeof(unsigned char));
	unsigned char* test_images = (unsigned char*)malloc(NUM_TEST_IMAGES * IMG_SIZE * IMG_SIZE * sizeof(unsigned char));
	unsigned char* test_labels = (unsigned char*)malloc(NUM_TEST_IMAGES * sizeof(unsigned char));

	load_images("C:\\Ra\\关于py的项目（charm）\\MINIST_HWN\\MNIST\\train-images-idx3-ubyte.gz", train_images, NUM_TRAIN_IMAGES);
	load_labels("C:\\Ra\\关于py的项目（charm）\\MINIST_HWN\\MNIST\\train-labels-idx1-ubyte.gz", train_labels, NUM_TRAIN_IMAGES);
	load_images("C:\\Ra\\关于py的项目（charm）\\MINIST_HWN\\MNIST\\t10k-images-idx3-ubyte.gz", test_images, NUM_TEST_IMAGES);
	load_labels("C:\\Ra\\关于py的项目（charm）\\MINIST_HWN\\MNIST\\t10k-labels-idx1-ubyte.gz", test_labels, NUM_TEST_IMAGES);

	// 随机初始化卷积核和全连接层权重
	float* conv_kernel = (float*)malloc(3 * 3 * sizeof(float));
	for (int i = 0; i < 3 * 3; i++) {
		conv_kernel[i] = (float)rand() / RAND_MAX;
	}

	int pool_output_size = (IMG_SIZE - 3 + 1) / 2;
	int fc_input_size = pool_output_size * pool_output_size;
	float* fc_weights = (float*)malloc(NUM_CLASSES * fc_input_size * sizeof(float));
	for (int i = 0; i < NUM_CLASSES * fc_input_size; i++) {
		fc_weights[i] = (float)rand() / RAND_MAX;
	}

	// 测试模型
	int correct_count = 0;
	for (int i = 0; i < NUM_TEST_IMAGES; i++) {
		unsigned char* image = &test_images[i * IMG_SIZE * IMG_SIZE];
		int predicted_label = predict(image, conv_kernel, fc_weights);
		if (predicted_label == test_labels[i]) {
			correct_count++;
		}
	}

	float accuracy = (float)correct_count / NUM_TEST_IMAGES;
	printf("Accuracy: %.2f%%\n", accuracy * 100);

	free(train_images);
	free(train_labels);
	free(test_images);
	free(test_labels);
	free(conv_kernel);
	free(fc_weights);

	return 0;
}
