// TextureLBP_PascalCanuel.cpp : définit le point d'entrée pour l'application console.
//

#include "stdafx.h"
#include "cuda_runtime.h" 
#include "device_launch_parameters.h"
#include <opencv2/core/core.hpp> 
#include <opencv2/highgui/highgui.hpp> 
#include <opencv2/imgproc/types_c.h> 
#include <opencv2/imgproc/imgproc.hpp> 
#include <iostream>
#include <vector>

using namespace cv;

extern "C" bool GPGPU_LBP(cv::Mat* imgIn, cv::Mat* imgOut);

int main()
{
	std::vector<String> pathList;
	pathList.push_back("../Pictures/lena.bmp");
	pathList.push_back("../Pictures/legumes.jpg");
	pathList.push_back("../Pictures/penguins.jpg");
	pathList.push_back("../Pictures/player.jpg");

	// loop through all the images of the list
	while (!pathList.empty())
	{
		String currentPath = pathList.back();
		pathList.pop_back();

		Mat imgGrayscale = imread(currentPath, IMREAD_GRAYSCALE); // loads image in grayscale mode
		Mat imgLBP = imgGrayscale.clone();

		GPGPU_LBP(&imgGrayscale, &imgLBP);

		imshow("LBP", imgLBP);
		waitKey(0); // go to next image if key entered	
	}

    return 0;
}

