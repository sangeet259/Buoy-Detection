#include <iostream>
#include <math.h>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;



Mat(Mat img_hsv)
{

	//iterate over all the pixels of this hsv image
	for (int y = 0; y < img_hsv.rows; ++y) {
		for (int x = 0; x < img_hsv.cols; ++x) {
			// Get value current channel and current pixel
			const cv::Vec3b& s = img_hsv.at<cv::Vec3b>(y, x);
			// For  the value channel (hue, sat, val)
			unsigned int pxl_val_current = (unsigned int)s.val[2];
			// ... do stuff with pxl_val
			// Define a bo



		}
	}


}



int main() {

  VideoCapture cap("buoy.avi");
  if(!cap.isOpened()){
    cout<<"Can't open video file.";
    return -1;
  }
    Mat frame, hsv, grey,res;
  	while(cap.read(frame)) {
  
    cvtColor(frame, hsv, COLOR_BGR2HSV);

    Mat image = cfar(hsv);
    cvtColor(image, grey, COLOR_BGR2GRAY);
    dilate(grey, grey, 0, Point(-1, -1), 5, 1, 1);
    imshow("cfar_output", image);    
    imshow("object", frame);

/*    pyrMeanShiftFiltering( image, res, 20, 20, 6);
    imshow("result", res);*/
 	   

    if(waitKey(1)=='q')
	break;
   
  }


  return 0;
}
