#include <iostream>
#include <math.h>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"
#include <fstream>

using namespace cv;
using namespace std;
ofstream outf("Sample.txt");

bool isGuardCell(int i,int j , int y, int x)
{
  if(j>=y-3 && j<=y+3 && i>=x-3 && i<=x+3)
    return true;
}

Mat cfar(Mat img_hsv)
{
  Mat img_grayscale(img_hsv.rows, img_hsv.cols, CV_8UC1, Scalar(255));

  //iterate over all the pixels of this- hsv image
  for (int y = 0; y < img_hsv.rows; ++y) {

    for (int x = 0; x < img_hsv.cols; ++x) {

      // Get value current channel and current pixel
      const cv::Vec3b& s = img_hsv.at<cv::Vec3b>(y, x);
      // For  the value channel (hue, sat, val)
      unsigned int pxl_val_current = (unsigned int)s.val[2];
      // ... do stuff with pxl_val
      // For the current pixel take a 7X7 box around it
      //first the 'y'
      float sum_value_channel_train=0;
      float sum_value_channel_guard=0;

      for (int j =y-3 ; j <= y+3; y++)
      {
        //then the 'x'
        for (int i =x-3 ; i <= x+3; x++)
        {
          //take the value channel of this pixel and add to the local scope around this pixel
          const cv::Vec3b& r = img_hsv.at<cv::Vec3b>(y, x);
          unsigned int pxl_val_inside = (unsigned int)r.val[2];
          if(!isGuardCell(i,j,y,x))
          {
            sum_value_channel_train+=pxl_val_inside;
          }
          else
            if(!(j==y && i==x))
            {
              sum_value_channel_guard+=pxl_val_inside;
            }

          // Now we have the data
          
            float avg_value_train = sum_value_channel_train/40.0;
            float avg_value_guard = sum_value_channel_guard/8.0;
          //Compare and excecute
            if (pxl_val_current>avg_value_train)
              //color a pixel black
              img_grayscale.at<unsigned char>(y,x)=0;


        }
      }

    }
  }


}

//Lets Clear some trash :)

int main() {
outf<<"Inside main";
  //Open the video
    VideoCapture cap("buoy.avi");
    // Check if opened properly
    if(!cap.isOpened()){
      cout<<"Can't open video file.";
      return -1;
    }
    Mat frame, hsv;
    while(cap.read(frame)) {
  
    cvtColor(frame, hsv, COLOR_BGR2HSV);
    Mat image = cfar(hsv);
      /*cvtColor(image, grey, COLOR_BGR2GRAY);*/
      /*dilate(grey, grey, 0, Point(-1, -1), 5, 1, 1);*/
      imshow("cfar_output", image);    
      imshow("object", frame);
      if(waitKey(1)=='q')
    break;
   }


  return 0;
}
