#include <iostream>
#include <opencv2/opencv.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/video/tracking.hpp"

using namespace std;
using namespace cv;

int main(){
  VideoCapture cap("/home/ayu/Videos/buoy2.avi");
  
  if(!cap.isOpened()){
    cout<<"There is some load in the video and it can't be opened y'all";
    return -1;
  }
  
  namedWindow("Frame", WINDOW_AUTOSIZE);
  resizeWindow("Frame", 640, 480);
  
  int hue, saturation, value;
  hue =0;
  saturation =148 ;
  value = 73;
  
  createTrackbar("hue", "Frame", &hue , 180);
  createTrackbar("saturation", "Frame", &saturation, 255);
  createTrackbar("Value", "Frame", &value, 255);
  
  Mat_<float> measurement(2,1);
  KalmanFilter KF(4, 2, 0);
  
  float t = 1;
  
  KF.transitionMatrix = (Mat_<float>(4, 4) << 1,0,t,0,   0,1,0,t,  0,0,1,0,  0,0,0,1);
  
  setIdentity(KF.measurementMatrix);
  setIdentity(KF.processNoiseCov, Scalar::all(0.001));
  setIdentity(KF.measurementNoiseCov, Scalar::all(10));
  setIdentity(KF.errorCovPost, Scalar::all(1));
  
  vector<Point> mousev,kalmanv;
    Mat frame,ref;
  
  while(cap.read(frame)){
    
    Mat canny_output;

    mousev.clear();
    kalmanv.clear();
    
    int count=0;
    int sum_x=0 ; int sum_y = 0;
    //double fps = cap.get(CV_CAP_PROP_FPS);    

    blur(frame,frame, Size(41,41) );
    Canny(ref, canny_output, 50, 100);
    
    cvtColor(frame,frame, CV_BGR2HSV);
    
    for(int i=0; i<frame.rows; i++){
      for(int j=0; j<frame.cols; j++){
        if(frame.at<Vec3b>(i,j)[0]>hue && frame.at<Vec3b>(i,j)[1]<saturation && frame.at<Vec3b>(i,j)[2]>value){
          frame.at<Vec3b>(i,j)[0]=180;
          frame.at<Vec3b>(i,j)[1]=255;
          frame.at<Vec3b>(i,j)[2]=255;
          sum_x+= i;
          sum_y+= j;
          count++;
          // cout<<i<<", "<<j<<endl;
        }
    
       else{
          frame.at<Vec3b>(i,j)[0]=0;
          frame.at<Vec3b>(i,j)[1]=0;
          frame.at<Vec3b>(i,j)[2]=0;
        }
      }
    }

   	measurement(1)= (float)sum_x/count;
   	measurement(0) = (float)sum_y/count;
   
        //cout<<"FPS :  "<<fps<<endl;
        cout<<"This is the measurement output :  "<<measurement<<endl;
   
        Mat prediction = KF.predict();
 	Point predictPt(prediction.at<float>(0),prediction.at<float>(1));
 	
        Mat estimated = KF.correct(measurement);

 	Point statePt(estimated.at<float>(0),estimated.at<float>(1));

 	Point measPt(measurement(0), measurement(1));
 	
    mousev.push_back(measPt);
    kalmanv.push_back(statePt);
    
    
    for(int j = 0; j< mousev.size() ; j++)
    	circle(frame, mousev[j], 5,Scalar(255,0,0));

    
    for (int i = 0; i < kalmanv.size(); i++)
        circle(frame, kalmanv[i], 5, Scalar(0,255,0));
    

    imshow("Frame", frame);
    
 //   imshow("Reference", ref);
    
    if(waitKey(100)=='q') 
	break;
  }
}
