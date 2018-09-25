#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/video/tracking.hpp>
#include "stdafx.h"

typedef unsigned char uchar;
typedef unsigned int uint;

#define BLOCK_SIZE 32

int iDivUp(int a, int b)
{
	return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

__device__
int Signe(int pValue)
{
	int ret = 0;
	if (pValue >= 0)
		ret = 1;
	return ret;
}

// kernel pour l'opérateur LBP
__global__
void Kernel_LBP(uchar* imgIn, uchar* imgOut, int ImgWidth, int imgHeight) {
	int ImgNumColonne = blockIdx.x  * blockDim.x + threadIdx.x;
	int ImgNumLigne = blockIdx.y  * blockDim.y + threadIdx.y;

	int Index = (ImgNumLigne * ImgWidth) + ImgNumColonne;

	if ((ImgNumColonne < ImgWidth - 2) && (ImgNumLigne < imgHeight - 2)) { // ne pas calculer les bordures
		// valeur de centre
		int centerValue = imgIn[Index + ImgWidth + 1];
		
		// signe des valeurs
		int tl = Signe(imgIn[Index] - centerValue);
		int tc = Signe(imgIn[Index + 1] - centerValue);
		int tr = Signe(imgIn[Index + 2] - centerValue);
		int cl = Signe(imgIn[Index + ImgWidth] - centerValue);
		int cr = Signe(imgIn[Index + ImgWidth + 2] - centerValue);
		int bl = Signe(imgIn[Index + ImgWidth * 2] - centerValue);
		int bc = Signe(imgIn[Index + ImgWidth * 2 + 1] - centerValue);
		int br = Signe(imgIn[Index + ImgWidth * 2 + 2] - centerValue);

		imgOut[Index] = tl * 1 + tc * 2 + tr * 4 + cl * 8 + cr * 16 + bl * 32 + bc * 64 + bc * 128;
	}
}

extern "C" bool GPGPU_LBP(cv::Mat* imgIn, cv::Mat* imgOut)
{
	//	1. Initialize data
	cudaError_t cudaStatus;
	uchar* gDevImage;
	uchar* gDevImageOut;

	uint imageSize = imgIn->rows * imgIn->step1();

	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(iDivUp(imgIn->cols, BLOCK_SIZE), iDivUp(imgIn->rows, BLOCK_SIZE));

	//	2. Allocation data
	cudaStatus = cudaMalloc(&gDevImage, imageSize);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	cudaStatus = cudaMalloc(&gDevImageOut, imageSize);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	//	3. Copy data on GPU
	cudaStatus = cudaMemcpy(gDevImage, imgIn->data, imageSize, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	//	4. Launch kernel
	Kernel_LBP << <dimGrid, dimBlock >> >(gDevImage, gDevImageOut, imgIn->cols, imgIn->rows);

	//Wait for the kernel to end
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaDeviceSynchronize failed!");
		goto Error;
	}

	//	5. Copy data on CPU
	cudaStatus = cudaMemcpy(imgOut->data, gDevImageOut, imageSize, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	//	6. Free GPU memory
Error:
	cudaFree(gDevImage);
	cudaFree(gDevImageOut);

	return cudaStatus;
}