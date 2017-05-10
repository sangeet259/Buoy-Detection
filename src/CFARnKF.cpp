#include <iostream>
#include <opencv2/opencv.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/video/tracking.hpp"

using namespace cv;
using namespace std;

Mat cfar(Mat img, int num_train, int num_guard, float rate1, float rate2, int block_size) {

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
      
	//CASO-CFAR :: blue blocks
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

	//CA-CFAR :: green blocks
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
             a.at<Vec3b>(k,l)[1] = 255;
	}   
 
  }

  return a;
}


void kalmanfilter(Mat image, Mat frame, KalmanFilter KF, Mat_<float> measurement) {

    int count=0;
    int sum_x=0 ; int sum_y = 0;

    for(int i=200; i<frame.rows; i++)
      for(int j=0; j<frame.cols; j++) {

	if( frame.at<Vec3b>(i,j)[0]==255 || frame.at<Vec3b>(i,j)[1]==255 ) {	
		sum_x+=i;
		sum_y+=j;
		count++;
	}		
     }

	// to ensure it works only when object is in the frame
	if( !count ) return;

        // PREDICTION
        Mat prediction = KF.predict();
 	Point predictPt(prediction.at<float>(0),prediction.at<float>(1));

	// MEASUREMENT
        measurement(1)= (float)sum_x/count;
   	measurement(0) = (float)sum_y/count;

 	Point measPt(measurement(0), measurement(1));
      
        cout<<"CENTRE :  "<<measurement<<endl;   
 	
        // CORRECTION        
          Mat estimated = KF.correct(measurement);
 	  Point statePt(estimated.at<float>(0),estimated.at<float>(1));

       circle(image, measPt, 85,Scalar(255,0,255)); 
       circle(image, statePt, 85, Scalar(0,255,255));
}


int main() {

  VideoCapture cap("/home/ayu/Videos/buoy.avi");

  if(!cap.isOpened()){
    cout<<"Cannot open video file";
    return -1;
  }


  Mat_<float> measurement(2,1);
  KalmanFilter KF(4, 2, 0);
  
  float dt = 1;
  
  KF.transitionMatrix = (Mat_<float>(4, 4) << 1,0,dt,0,   0,1,0,dt,  0,0,1,0,  0,0,0,1);

  setIdentity(KF.measurementMatrix);
  setIdentity(KF.processNoiseCov, Scalar::all(0.01));
  setIdentity(KF.measurementNoiseCov, Scalar::all(10));
  setIdentity(KF.errorCovPost, Scalar::all(1));
  
  Mat image;

  while(cap.read(image)) {
    
    Mat hsv, gray;
    
    cvtColor(image,hsv, COLOR_BGR2HSV);

    Mat frame = cfar(hsv, 9, 2, 0.6, 0.3, 6);
    
    kalmanfilter(image, frame, KF, measurement);
    imshow("Frame", frame);
    imshow("Object", image);

    if(waitKey(100)=='q') 
	break;
  }
  
  return 0;
}
