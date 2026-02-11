// Copyright 2024 Stereolabs
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef ZED_ARUCO_LOC_COMPONENT_HPP_
#define ZED_ARUCO_LOC_COMPONENT_HPP_

#include <rcutils/logging_macros.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>

#include <atomic>
#include <image_transport/camera_publisher.hpp>
#include <image_transport/camera_subscriber.hpp>
#include <image_transport/image_transport.hpp>
#include <map>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/u_int32.hpp>
#include <zed_msgs/srv/set_pose.hpp>

#include "aruco_loc_visibility_control.hpp"

namespace stereolabs
{

typedef struct
{
  int idx;
  std::string marker_frame_id;
  std::vector<double> position;
  std::vector<double> orientation;
} ArucoPose;

class ZedArucoLoc : public rclcpp::Node
{
public:
  ZED_ARUCO_LOC_COMPONENT_PUBLIC
  explicit ZedArucoLoc(const rclcpp::NodeOptions & options);

  virtual ~ZedArucoLoc() {}

protected:
  void camera_callback(
    const sensor_msgs::msg::Image::ConstSharedPtr & img,
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr & cam_info);

  template<typename T>
  void getParam(
    std::string paramName, T defValue, T & outVal,
    std::string log_info = std::string(), bool dynamic = false);

  void getParams();
  void getGeneralParams();
  void getMarkerParams();
  void getQualityFilterParams();
  void applyResolutionParams(int image_width, int image_height);

  void initTFs();
  void broadcastMarkerTFs();
  bool getTransformFromTf(
    std::string targetFrame, std::string sourceFrame,
    tf2::Transform & out_tr);

  bool resetZedPose(tf2::Transform & new_pose);

private:
  // ----> ROS Messages
  image_transport::CameraPublisher _pubDetect;
  image_transport::CameraSubscriber _subImage;
  rclcpp::Publisher<std_msgs::msg::UInt32>::SharedPtr _pubCount;
  rclcpp::QoS _defaultQoS;
  // <---- ROS Messages

  // Service client
  rclcpp::Client<zed_msgs::srv::SetPose>::SharedPtr _setPoseClient;

  // ----> Running variables
  rclcpp::Time _detTime;
  std::atomic<bool> _detRunning;
  uint32_t _arucoCount = 0;
  // <---- Running variables

  // ----> General Parameters
  int _markerCount = 1;
  float _markerSize = 0.16f;
  float _detRate = 1.0f;
  std::string _worldFrameId;
  std::string _cameraName = "zed";
  double _maxDist;
  bool _refineDetection;
  std::map<int, ArucoPose> _tagPoses;
  bool _debugActive;
  // <---- General Parameters

  // ----> Quality Filter Parameters (shared, resolution-independent)
  double _maxObliqueAngle = 65.0;
  double _sameMarkerCooldown = 5.0;

  // Resolution-dependent params (set by applyResolutionParams on first frame)
  double _minLaplacianVariance = 50.0;
  double _maxReprojError = 2.0;
  double _minMarkerArea = 2500.0;

  // Stored from YAML for both resolutions
  double _minLaplacianVariance_1080 = 50.0;
  double _maxReprojError_1080 = 2.0;
  double _minMarkerArea_1080 = 2500.0;
  double _minLaplacianVariance_720 = 40.0;
  double _maxReprojError_720 = 1.5;
  double _minMarkerArea_720 = 1500.0;

  bool _resolutionDetected = false;
  // <---- Quality Filter Parameters

  // ----> Per-marker reset tracking (same-marker suppression)
  struct MarkerResetInfo {
    double reproj_error;    // reproj error at last accepted reset
    rclcpp::Time timestamp; // time of last accepted reset
  };
  std::map<int, MarkerResetInfo> _lastResetInfo;
  // <---- Per-marker reset tracking

  // ----> TF2
  std::unique_ptr<tf2_ros::Buffer> _tfBuffer;
  std::unique_ptr<tf2_ros::TransformListener> _tfListener;
  std::unique_ptr<tf2_ros::TransformBroadcaster> _tfBroadcaster;

  tf2::Transform _img2aruco;
  tf2::Transform _aruco2img;
  tf2::Transform _ros2img;
  tf2::Transform _img2ros;
  tf2::Transform _left2base;

  rclcpp::TimerBase::SharedPtr _tfTimer;
  // <---- TF2
};

}  // namespace stereolabs

#endif  // ZED_ARUCO_LOC_COMPONENT_HPP_