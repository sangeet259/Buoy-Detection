#include <iostream>
#include <math.h>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/segmentation.hpp"

using namespace cv;
using namespace std;

Mat cfar(Mat& frame, Mat img, int num_train, int num_guard, float rate1, float rate2, int block_size) {

  Mat a(img.rows, img.cols, CV_8UC3, Scalar(0,0,0));
  
  int num_rows = a.rows-(a.rows%(num_train + num_guard));
  int num_cols = a.cols-(a.cols%(num_train + num_guard));
  int num_side = (num_train/2)+(num_guard/2);
  
  double alpha1 = num_train * ( pow(rate1, -1.00/num_train) -1 );	
  double alpha2 = num_train * ( pow(rate2, -1.00/num_train) -1 );  

  for(int i = num_side; i <= num_rows; i += num_side*2) 
    for(int j = num_side; j <= num_cols; j += num_side*2) {
       
       int sum1 = 0, sum2 = 0;
       double thresh, p_noise;
      
	//CASO-CFAR
       for(int x = i-num_side; x <= i-num_guard; x++) 
         for(int y = j-num_side; y <= j-num_guard; y++)
           sum1 += img.at<Vec3b>(x,y)[0];

       for(int x = i+num_guard; x <= i+num_side; x++)
         for(int y = j+num_guard; y <= j+num_side; y++)
           sum2 += img.at<Vec3b>(x,y)[0];
  	
       p_noise=min(sum1,sum2)/(num_train*num_train);
       thresh = alpha1*p_noise;

       if( img.at<Vec3b>(i,j)[0] < thresh) {

         for(int k = i-block_size/2; k <= i+block_size/2; k++)
           for(int l = j-block_size/2; l <= j+block_size/2; l++)
             a.at<Vec3b>(k,l)[0] = 255;	
	}

	//CA-CFAR 
       for(int x = i-(num_guard/2); x <= i+(num_guard/2)+1; x++)
         for(int y = j-(num_guard/2); y <= j+(num_guard/2)+1; y++)
           sum1 += img.at<Vec3b>(x,y)[0];

       for(int x = i-num_side; x <= i+num_side+1; x++)
         for(int y = j-num_side; y <= j+num_side+1; y++)
           sum2 += img.at<Vec3b>(x,y)[0];
  
       p_noise = fabs(sum1-sum2)/(num_train*num_train);
       thresh = alpha2*p_noise;
  
       if( img.at<Vec3b>(i,j)[0] > thresh) {

         for(int k = i-block_size/2; k <= i+block_size/2; k++)
           for(int l = j-block_size/2; l <= j+block_size/2; l++)
             a.at<Vec3b>(k,l)[0] = 255;
	}   
 
  }

/* AVERAGE 

    int count=0;
    int sum_x=0 ; int sum_y = 0;

    for(int i=0; i<a.rows; i++)
      for(int j=0; j<a.cols; j++) {

	if( a.at<Vec3b>(i,j)[0]==255 || a.at<Vec3b>(i,j)[1]==255 ) {
	
		sum_x+=i;
		sum_y+=j;
		count++;
	}		
     }
	if(count!=0)
    circle(frame, Point(sum_x/count, sum_y/count), 100,Scalar(0,0,255));
*/

  return a;
}

int main() {

  VideoCapture cap("/home/ayu/Videos/buoy.avi");
  
  if(!cap.isOpened()){
    cout<<"Can't open video file.";
    return -1;
  }
    Mat frame, hsv, grey,res;
  while(cap.read(frame)) {
  
    cvtColor(frame, hsv, COLOR_BGR2HSV);

    Mat image = cfar(frame, hsv, 9, 2, 0.5, 0.2, 10);
    cvtColor(image, grey, COLOR_BGR2GRAY);
  
    dilate(grey, grey, 0, Point(-1, -1), 5, 1, 1);

  /* HOUGH CIRCLES

    vector<Vec3f> circles;
    HoughCircles( grey, circles, CV_HOUGH_GRADIENT, 1, 5, 1, 100, 10, 0 );
  
    for( size_t i = 0; i < circles.size(); i++ ) {
      Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
      int radius = cvRound(circles[i][2]);
      // circle center
      circle( frame, center, 3, Scalar(0,255,0), -1, 8, 0 );
      // circle outline
      circle( frame, center, radius, Scalar(0,0,255), 3, 8, 0 );
    }
  */

  /* CONTOUR DETECTIOIN
    
    vector< vector< Point > > contours;
    findContours(grey, contours,CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

    for (size_t idx = 0; idx < contours.size(); idx++) {
        drawContours(frame, contours, idx, Scalar(0));
    }
  */
    imshow("cfar_output", image);    
    imshow("object", frame);
    pyrMeanShiftFiltering( image, res, 20, 20, 6);
    imshow("result", res);
 	   
    if(waitKey(1)=='q')
	break;
   
  }

  return 0;
}
