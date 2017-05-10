#include "ros/ros.h"
#include <std_msgs/Int64MultiArray.h>

void callback1(const std_msgs::Int64MultiArray::ConstPtr& num )
{
 ROS_INFO("\nCentre: %ld , %d :: Radius: %ld ", num->data[0], num->data[1], num->data[2]);
 return;
}


int main(int argc, char **argv)
{
 ros::init(argc, argv, "print");
 ros::NodeHandle n;
 ros::Rate loop_rate(1);

 ros::Subscriber sub_n = n.subscribe("state", 1, callback1);

 ros::spin();
 loop_rate.sleep();
 return 0;
}
