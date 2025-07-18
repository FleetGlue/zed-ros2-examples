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

#include "zed_aruco_localization_component.hpp"

#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Transform.h>

#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <sstream>

#include "aruco.hpp"
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#ifdef FOUND_HUMBLE
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#elif defined FOUND_IRON
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#elif defined FOUND_JAZZY
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#elif defined FOUND_ROLLING
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#elif defined FOUND_FOXY
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#else
#error Unsupported ROS2 distro
#endif

using namespace std::chrono_literals;
using namespace std::placeholders;

#define TIMEZERO_ROS rclcpp::Time(0, 0, RCL_ROS_TIME)

#ifndef DEG2RAD
#define DEG2RAD 0.017453293
#define RAD2DEG 57.295777937
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace stereolabs
{

ZedArucoLoc::ZedArucoLoc(const rclcpp::NodeOptions & options)
: Node("zed_aruco_loc_node", options), _defaultQoS(1), _detRunning(false)
{
  RCLCPP_INFO(get_logger(), "*********************************");
  RCLCPP_INFO(get_logger(), " ZED ArUco Localization Component ");
  RCLCPP_INFO(get_logger(), "*********************************");
  RCLCPP_INFO(get_logger(), " * namespace: %s", get_namespace());
  RCLCPP_INFO(get_logger(), " * node name: %s", get_name());
  RCLCPP_INFO(get_logger(), "*********************************");

  /* Note: it is very important to use a QOS profile for the subscriber that is
   * compatible with the QOS profile of the publisher. The ZED component node
   * uses a default QOS profile with reliability set as "RELIABLE" and
   * durability set as "VOLATILE". To be able to receive the subscribed topic
   * the subscriber must use compatible parameters.
   */

  // https://github.com/ros2/ros2/wiki/About-Quality-of-Service-Settings

  _defaultQoS.keep_last(10);
  _defaultQoS.reliable();
  _defaultQoS.durability_volatile();

  // Initialize detection time for throttling
  _detTime = get_clock()->now();

  // Initialize pose correction transform to identity
  _poseCorrection.setIdentity();
  _hasPoseCorrection = false;
  _lastCorrectionTime = get_clock()->now();

  // Load parameters
  getParams();

  // ----> TF2 Transform
  _tfBuffer = std::make_unique<tf2_ros::Buffer>(get_clock());
  _tfListener = std::make_unique<tf2_ros::TransformListener>(
    *_tfBuffer);    // Start TF Listener thread
  _tfBroadcaster = std::make_unique<tf2_ros::TransformBroadcaster>(this);

  initTFs();

  int msec = static_cast<int>(1000. / (_detRate * 10.));
  _tfTimer =
    create_wall_timer(
    std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::milliseconds(msec)),
    std::bind(&ZedArucoLoc::broadcastMarkerTFs, this));
  // <---- TF2 Transform

  // Create image publisher
  _pubDetect = image_transport::create_camera_publisher(
    this, "out/aruco_result", _defaultQoS.get_rmw_qos_profile());
  RCLCPP_INFO_STREAM(
    get_logger(),
    "Advertised on topic: " << _pubDetect.getTopic());
  RCLCPP_INFO_STREAM(
    get_logger(),
    "Advertised on topic: " << _pubDetect.getInfoTopic());

  // Create camera image subscriber
  _subImage = image_transport::create_camera_subscription(
    this, "in/zed_image",
    std::bind(&ZedArucoLoc::camera_callback, this, _1, _2), "raw",
    _defaultQoS.get_rmw_qos_profile());

  RCLCPP_INFO_STREAM(
    get_logger(),
    "Subscribed to topic: " << _subImage.getTopic());
  RCLCPP_INFO_STREAM(
    get_logger(),
    "Subscribed to topic: " << _subImage.getInfoTopic());

  // Create pose subscriber
  _subPose = create_subscription<geometry_msgs::msg::PoseStamped>(
    "/zed/zed_node/pose", _defaultQoS,
    std::bind(&ZedArucoLoc::pose_callback, this, _1));
  RCLCPP_INFO_STREAM(
    get_logger(),
    "Subscribed to pose topic: " << _subPose->get_topic_name());

  // Create corrected pose publisher
  _pubCorrectedPose = create_publisher<geometry_msgs::msg::PoseStamped>(
    "/zed/zed_node/pose_corrected", _defaultQoS);
  RCLCPP_INFO_STREAM(
    get_logger(),
    "Publishing corrected pose on topic: " << _pubCorrectedPose->get_topic_name());

  // Create service caller (commented out but kept for compatibility)
  _setPoseClient = create_client<zed_msgs::srv::SetPose>("set_pose");
}

template<typename T>
void ZedArucoLoc::getParam(
  std::string paramName, T defValue, T & outVal,
  std::string log_info, bool dynamic)
{
  rcl_interfaces::msg::ParameterDescriptor descriptor;
  descriptor.read_only = !dynamic;

  declare_parameter(paramName, rclcpp::ParameterValue(defValue), descriptor);

  if (!get_parameter(paramName, outVal)) {
    RCLCPP_WARN_STREAM(
      get_logger(),
      "The parameter '"
        << paramName
        << "' is not available or is not valid, using the default value: "
        << defValue);
  }

  if (!log_info.empty()) {
    RCLCPP_INFO_STREAM(get_logger(), log_info << outVal);
  }
}

void ZedArucoLoc::getParams()
{
  getParam("debug.active", _debugActive, _debugActive);

  if (_debugActive) {
    rcutils_ret_t res = rcutils_logging_set_logger_level(
      get_logger().get_name(), RCUTILS_LOG_SEVERITY_DEBUG);

    if (res != RCUTILS_RET_OK) {
      RCLCPP_INFO(get_logger(), "Error setting DEBUG level for logger");
    } else {
      RCLCPP_INFO(get_logger(), "+++ Debug Mode enabled +++");
    }
  } else {
    rcutils_ret_t res = rcutils_logging_set_logger_level(
      get_logger().get_name(), RCUTILS_LOG_SEVERITY_INFO);

    if (res != RCUTILS_RET_OK) {
      RCLCPP_INFO(get_logger(), "Error setting INFO level for logger");
    }
  }

  getGeneralParams();
  getMarkerParams();
}

void ZedArucoLoc::getGeneralParams()
{
  RCLCPP_INFO(get_logger(), "*** GENERAL parameters ***");

  getParam(
    "general.marker_count", _markerCount, _markerCount,
    " * Marker count: ");
  getParam(
    "general.marker_size", _markerSize, _markerSize,
    " * Marker size [m]: ");
  getParam(
    "general.detection_rate", _detRate, _detRate,
    " * Detection rate [Hz]: ");
  getParam("general.camera_name", _cameraName, _cameraName, " * Camera name: ");
  getParam(
    "general.world_frame_id", _worldFrameId, _worldFrameId,
    " * World frame id: ");
  getParam(
    "general.maximum_distance", _maxDist, _maxDist,
    " * Maximum distance [m]: ");
  getParam("general.refine_detection", _refineDetection, _refineDetection);
  RCLCPP_INFO(
    get_logger(), " * Refine detection: %s",
    _refineDetection ? "TRUE" : "FALSE");
}

void ZedArucoLoc::getMarkerParams()
{
  RCLCPP_INFO(get_logger(), "*** MARKER parameters ***");

  rcl_interfaces::msg::ParameterDescriptor read_only_descriptor;
  read_only_descriptor.read_only = true;

  for (int i = 0; i < _markerCount; i++) {
    ArucoPose aruco_pose;
    std::stringstream ns;
    ns << "marker_" << std::setfill('0') << std::setw(3) << i;

    std::string par_pref = ns.str() + ".";

    RCLCPP_INFO_STREAM(get_logger(), " * " << par_pref);

    std::string par_name = par_pref + "aruco_id";
    getParam(par_name, aruco_pose.idx, aruco_pose.idx, "   * ArUco idx: ");

    std::stringstream ss;
    ss << "marker_" << std::setfill('0') << std::setw(3) << aruco_pose.idx;
    aruco_pose.marker_frame_id = ss.str();

    par_name = par_pref + "position";
    declare_parameter(
      par_name, rclcpp::ParameterValue(aruco_pose.position),
      read_only_descriptor);
    if (!get_parameter(par_name, aruco_pose.position)) {
      RCLCPP_ERROR_STREAM(
        get_logger(),
        "The parameter '"
          << par_name
          << "' is not available or is not valid.");
      exit(EXIT_FAILURE);
    }
    if (aruco_pose.position.size() != 3) {
      RCLCPP_ERROR_STREAM(
        get_logger(),
        "The parameter '"
          << par_name
          << "' must be a vector of 3 values of type FLOAT64 (double).");
      exit(EXIT_FAILURE);
    }
    RCLCPP_INFO(
      get_logger(), "   * Position: [%g,%g,%g]",
      aruco_pose.position[0], aruco_pose.position[1],
      aruco_pose.position[2]);

    par_name = par_pref + "orientation";
    declare_parameter(
      par_name, rclcpp::ParameterValue(aruco_pose.orientation),
      read_only_descriptor);
    if (!get_parameter(par_name, aruco_pose.orientation)) {
      RCLCPP_ERROR_STREAM(
        get_logger(),
        "The parameter '"
          << par_name
          << "' is not available or is not valid.");
      exit(EXIT_FAILURE);
    }
    if (aruco_pose.orientation.size() != 3) {
      RCLCPP_ERROR_STREAM(
        get_logger(),
        "The parameter '"
          << par_name
          << "' must be a vector of 3 values of type FLOAT64 (double).");
      exit(EXIT_FAILURE);
    }
    RCLCPP_INFO(
      get_logger(), "   * Orientation: [%g,%g,%g]",
      aruco_pose.orientation[0], aruco_pose.orientation[1],
      aruco_pose.orientation[2]);

    _tagPoses[aruco_pose.idx] = aruco_pose;
  }
}

void ZedArucoLoc::pose_callback(
  const geometry_msgs::msg::PoseStamped::ConstSharedPtr & msg)
{
  // Check for null pointer
  if (!msg) {
    RCLCPP_ERROR(get_logger(), "Received null pose message");
    return;
  }

  // Create corrected pose message
  auto corrected_msg = std::make_shared<geometry_msgs::msg::PoseStamped>();
  *corrected_msg = *msg;  // Copy all fields

  // Apply correction if available
  if (_hasPoseCorrection) {
    // Convert pose to tf2::Transform
    tf2::Transform current_pose;
    tf2::Vector3 current_origin(
      msg->pose.position.x,
      msg->pose.position.y,
      0.0);  // Force Z to 0 for 2D mode
    tf2::Quaternion current_rotation(
      msg->pose.orientation.x,
      msg->pose.orientation.y,
      msg->pose.orientation.z,
      msg->pose.orientation.w);
    current_pose.setOrigin(current_origin);
    current_pose.setRotation(current_rotation);

    // Apply correction
    tf2::Transform corrected_pose;
    corrected_pose.mult(_poseCorrection, current_pose);

    // Extract yaw from corrected pose (2D rotation)
    double roll, pitch, yaw;
    corrected_pose.getBasis().getRPY(roll, pitch, yaw);
    
    // Create 2D quaternion (only yaw rotation)
    tf2::Quaternion corrected_rotation;
    corrected_rotation.setRPY(0.0, 0.0, yaw);

    // Update message with corrected pose (2D only)
    corrected_msg->pose.position.x = corrected_pose.getOrigin().x();
    corrected_msg->pose.position.y = corrected_pose.getOrigin().y();
    corrected_msg->pose.position.z = 0.0;  // Always 0 for 2D mode
    
    corrected_msg->pose.orientation.x = corrected_rotation.x();
    corrected_msg->pose.orientation.y = corrected_rotation.y();
    corrected_msg->pose.orientation.z = corrected_rotation.z();
    corrected_msg->pose.orientation.w = corrected_rotation.w();

    // Optional: Add debug info about correction age
    double correction_age = (get_clock()->now() - _lastCorrectionTime).nanoseconds() / 1e9;
    if (_debugActive && correction_age > 10.0) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 5000,
        "Pose correction is %.1f seconds old", correction_age);
    }
  } else {
    // Even without correction, enforce 2D constraints
    corrected_msg->pose.position.z = 0.0;
    
    // Extract and reapply only yaw rotation
    tf2::Quaternion q(
      msg->pose.orientation.x,
      msg->pose.orientation.y,
      msg->pose.orientation.z,
      msg->pose.orientation.w);
    double roll, pitch, yaw;
    tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);
    
    tf2::Quaternion q_2d;
    q_2d.setRPY(0.0, 0.0, yaw);
    
    corrected_msg->pose.orientation.x = q_2d.x();
    corrected_msg->pose.orientation.y = q_2d.y();
    corrected_msg->pose.orientation.z = q_2d.z();
    corrected_msg->pose.orientation.w = q_2d.w();
  }

  // Publish corrected pose
  if (_pubCorrectedPose) {
    _pubCorrectedPose->publish(*corrected_msg);
  }
}

void ZedArucoLoc::camera_callback(
  const sensor_msgs::msg::Image::ConstSharedPtr & img,
  const sensor_msgs::msg::CameraInfo::ConstSharedPtr & cam_info)
{
  // ----> Check for null pointers
  if (!img || !cam_info) {
    RCLCPP_ERROR(get_logger(), "Received null image or camera info");
    return;
  }
  // <---- Check for null pointers

  // ----> Check for correct input image encoding
  if (img->encoding != sensor_msgs::image_encodings::BGRA8) {
    RCLCPP_ERROR(
      get_logger(),
      "The input topic image requires 'BGRA8' encoding");
    exit(EXIT_FAILURE);
  }
  // <---- Check for correct input image encoding

  // If a detection is already running let's return
  if (_detRunning) {
    return;
  }

  double det_elapsed_sec = (get_clock()->now() - _detTime).nanoseconds() / 1e9;

  // If it's too early to perform another detection let's return
  if (det_elapsed_sec < (1.0 / _detRate)) {
    return;
  }

  _detRunning = true;

  // Time statistics
  rclcpp::Time start;
  double elapsed_sec;

  // Is result image subscribed?
  bool res_sub = (_pubDetect.getNumSubscribers() > 0);

  if (_debugActive) {
    RCLCPP_INFO_STREAM(get_logger(), "*****************************");
    RCLCPP_INFO_STREAM(get_logger(), "    ArUco detection");
  }

  // ----> Convert BGRA image for processing by using OpenCV
  try {
    start = get_clock()->now();
    
    // Check image data validity
    if (img->data.empty()) {
      RCLCPP_ERROR(get_logger(), "Image data is empty");
      _detRunning = false;
      return;
    }
    
    void * data =
      const_cast<void *>(reinterpret_cast<const void *>(&img->data[0]));
    cv::Mat bgra(img->height, img->width, CV_8UC4, data);
    cv::Mat bgr, gray;  // bgr is used to publish the detection image, gray for
                        // ArUco processing

    cv::cvtColor(bgra, gray, cv::COLOR_BGRA2GRAY);
    cv::cvtColor(bgra, bgr, cv::COLOR_BGRA2BGR);

    elapsed_sec = (get_clock()->now() - start).nanoseconds() / 1e9;
    if (_debugActive) {
      RCLCPP_INFO_STREAM(
        get_logger(),
        " * Color conversion: " << elapsed_sec << " sec");
    }
    // <---- Convert BGRA image for processing by using OpenCV

    // ----> Detect ArUco Markers
    start = get_clock()->now();
    std::vector<int> ids;
    std::vector<std::vector<cv::Point2f>> corners;

    auto dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_100);
    cv::aruco::detectMarkers(bgr, dictionary, corners, ids);
    elapsed_sec = (get_clock()->now() - start).nanoseconds() / 1e9;
    if (_debugActive) {
      RCLCPP_INFO_STREAM(
        get_logger(),
        " * Marker detection: " << elapsed_sec << " sec");
    }
    // <---- Detect ArUco Markers

    if (corners.empty() || ids.empty()) {
      if (_debugActive) {
        RCLCPP_INFO_STREAM(get_logger(), "  No Markers in view");
        RCLCPP_INFO_STREAM(get_logger(), "*****************************");
      }
      _detRunning = false;
      return;
    }

  if (_debugActive) {
    RCLCPP_INFO_STREAM(get_logger(), " * Detected tags: " << ids.size());
  }

  // ----> Refine the result
  if (_refineDetection) {
    start = get_clock()->now();
    for (size_t i = 0; i < corners.size(); ++i) {
      cv::cornerSubPix(
        gray, corners[i], cv::Size(5, 5), cv::Size(-1, -1),
        cv::TermCriteria(
          cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30,
          0.1));
    }
    elapsed_sec = (get_clock()->now() - start).nanoseconds() / 1e9;
    if (corners.empty()) {
      if (_debugActive) {
        RCLCPP_INFO_STREAM(get_logger(), "  No Markers in view");
        RCLCPP_INFO_STREAM(get_logger(), "*****************************");
      }
      _detRunning = false;
      return;
    }
    if (_debugActive) {
      RCLCPP_INFO_STREAM(
        get_logger(),
        " * Subpixel refinement: " << elapsed_sec << " sec");
    }
  }
  // <---- Refine the result

  // ----> Estimate Marker positions
  start = get_clock()->now();
  std::vector<cv::Vec3d> rvecs, tvecs;

  cv::Matx33d camera_matrix = cv::Matx33d::eye();  // Camera matrix
  camera_matrix(0, 0) = cam_info->k[0];
  camera_matrix(1, 1) = cam_info->k[4];
  camera_matrix(0, 2) = cam_info->k[2];
  camera_matrix(1, 2) = cam_info->k[5];
  cv::Matx<float, 4, 1> dist_coeffs =
    cv::Vec4f::zeros();    // No distortions if subscribing to rectified ZED
                           // images

  cv::aruco::estimatePoseSingleMarkers(
    corners, _markerSize, camera_matrix,
    dist_coeffs, rvecs, tvecs);
  elapsed_sec = (get_clock()->now() - start).nanoseconds() / 1e9;
  if (_debugActive) {
    RCLCPP_INFO_STREAM(
      get_logger(),
      " * Marker poses estimation: " << elapsed_sec << " sec");
  }
  // <---- Estimate Marker positions

  // ----> Find the closest marker
  size_t nearest_aruco_index = 999;
  double nearest_distance = 1e9;
  for (size_t i = 0; i < ids.size(); i++) {
    double x = tvecs[i](0);
    double y = tvecs[i](1);
    double z = tvecs[i](2);
    double current_distance = sqrt(x * x + y * y + z * z);
    if (nearest_distance > current_distance) {
      nearest_distance = current_distance;
      nearest_aruco_index = i;
    }
  }
  if (_debugActive) {
    RCLCPP_INFO_STREAM(
      get_logger(), " * Nearest marker: "
        << ids[nearest_aruco_index] << " -> "
        << nearest_distance << "m");
  }
  // <---- Find the closest marker

  if (nearest_distance > _maxDist) {
    if (_debugActive) {
      RCLCPP_INFO_STREAM(
        get_logger(), "  The closest marker is too far: "
          << nearest_distance << " m > "
          << _maxDist << " m");
      RCLCPP_INFO_STREAM(get_logger(), "*****************************");
    }
    _detRunning = false;
    return;
  }

  RCLCPP_INFO_STREAM(
    get_logger(), " * ArUco marker #"
      << ids[nearest_aruco_index]
      << " in range: " << nearest_distance
      << " m");

  // ----> Check if the marker is available in the parameters
  auto search = _tagPoses.find(ids[nearest_aruco_index]);
  if (search == _tagPoses.end()) {
    RCLCPP_WARN_STREAM(
      get_logger(), "The marker with id #"
        << ids[nearest_aruco_index]
        << " is not available.");
    RCLCPP_WARN_STREAM(
      get_logger(),
      "Please check the file `aruco_loc.yaml` and verify that "
      "it's listed in the `marker_xxx` section.");
    _detRunning = false;
    return;
  }
  // <---- Check if the marker is available in the parameters

  double r, p, y;

  // ----> ArUco rvec and tvec to TF2 Transform
  tf2::Vector3 tf2_origin(tvecs[nearest_aruco_index][0],
    tvecs[nearest_aruco_index][1],
    tvecs[nearest_aruco_index][2]);
  cv::Mat cv_rot(3, 3, CV_64F);
  cv::Rodrigues(rvecs[nearest_aruco_index], cv_rot);
  // Convert to a tf2::Matrix3x3
  tf2::Matrix3x3 tf2_rot(cv_rot.at<double>(0, 0), cv_rot.at<double>(0, 1),
    cv_rot.at<double>(0, 2), cv_rot.at<double>(1, 0),
    cv_rot.at<double>(1, 1), cv_rot.at<double>(1, 2),
    cv_rot.at<double>(2, 0), cv_rot.at<double>(2, 1),
    cv_rot.at<double>(2, 2));

  tf2::Transform pose_aruco(tf2_rot, tf2_origin);

  if (_debugActive) {
    pose_aruco.getBasis().getRPY(r, p, y);
    RCLCPP_INFO(
      get_logger(),
      "pose_aruco -> Pos: [%.3f,%.3f,%.3f] - Or: [%.3f°,%.3f°,%.3f°]",
      pose_aruco.getOrigin().x(), pose_aruco.getOrigin().y(),
      pose_aruco.getOrigin().z(), r * RAD2DEG, p * RAD2DEG,
      y * RAD2DEG);
  }
  // <---- ArUco rvec and tvec to TF2 Transform

  // ----> Change basis from ArUco to camera in image coordinate system
  tf2::Transform pose_img;
  pose_img.mult(_img2aruco, pose_aruco);
  pose_img = pose_img.inverse();

  if (_debugActive) {
    pose_img.getBasis().getRPY(r, p, y);
    RCLCPP_INFO(
      get_logger(),
      "pose_img -> Pos: [%.3f,%.3f,%.3f] - Or: [%.3f°,%.3f°,%.3f°]",
      pose_img.getOrigin().x(), pose_img.getOrigin().y(),
      pose_img.getOrigin().z(), r * RAD2DEG, p * RAD2DEG,
      y * RAD2DEG);
  }
  // <---- Change basis from ArUco to camera in image coordinate system

  // ----> ArUco pose in ROS coordinate system
  tf2::Transform ros2aruco;
  ros2aruco.mult(_img2aruco, _ros2img);
  tf2::Transform aruco2ros = ros2aruco.inverse();
  // <---- ArUco pose in ROS coordinate system

  // ----> Left camera sensor in ROS coordinate respect to the marker
  tf2::Transform left_pose_marker;
  left_pose_marker.mult(pose_img, ros2aruco);
  left_pose_marker.mult(aruco2ros, left_pose_marker);

  if (_debugActive) {
    left_pose_marker.getBasis().getRPY(r, p, y);

    RCLCPP_INFO(
      get_logger(),
      "pose_marker -> Pos: [%.3f,%.3f,%.3f] - Or: [%.3f°,%.3f°,%.3f°]",
      left_pose_marker.getOrigin().x(), left_pose_marker.getOrigin().y(),
      left_pose_marker.getOrigin().z(), r * RAD2DEG, p * RAD2DEG,
      y * RAD2DEG);
  }
  // <---- Left camera sensor in ROS coordinate  respect to the marker

  // ----> Camera base in ROS coordinate respect to the marker
  tf2::Transform base_pose_marker;
  base_pose_marker.mult(left_pose_marker, _left2base);

  if (_debugActive) {
    base_pose_marker.getBasis().getRPY(r, p, y);
    RCLCPP_INFO(
      get_logger(),
      "pose_marker -> Pos: [%.3f,%.3f,%.3f] - Or: [%.3f°,%.3f°,%.3f°]",
      base_pose_marker.getOrigin().x(), base_pose_marker.getOrigin().y(),
      base_pose_marker.getOrigin().z(), r * RAD2DEG, p * RAD2DEG,
      y * RAD2DEG);
  }
  // <---- Camera base in ROS coordinate respect to the marker

  // ----> New camera pose in ROS world
  tf2::Transform marker_world_pose;
  tf2::Vector3 orig(_tagPoses[ids[nearest_aruco_index]].position[0],
    _tagPoses[ids[nearest_aruco_index]].position[1],
    _tagPoses[ids[nearest_aruco_index]].position[2]);
  marker_world_pose.setOrigin(orig);

  tf2::Quaternion q;
  q.setRPY(
    _tagPoses[ids[nearest_aruco_index]].orientation[0],
    _tagPoses[ids[nearest_aruco_index]].orientation[1],
    _tagPoses[ids[nearest_aruco_index]].orientation[2]);
  marker_world_pose.setRotation(q);

  tf2::Transform map_pose;
  map_pose.mult(marker_world_pose, base_pose_marker);
  // <---- New camera pose in ROS world

  // ----> Calculate and store pose correction
  // Get current pose from TF
  tf2::Transform current_pose;
  if (getTransformFromTf(_worldFrameId, _cameraName + "_camera_link", current_pose)) {
    // For 2D mode, we need to constrain the correction to X, Y, and yaw
    // First, extract 2D components from both poses
    double current_roll, current_pitch, current_yaw;
    double desired_roll, desired_pitch, desired_yaw;
    tf2::Matrix3x3(current_pose.getRotation()).getRPY(current_roll, current_pitch, current_yaw);
    tf2::Matrix3x3(map_pose.getRotation()).getRPY(desired_roll, desired_pitch, desired_yaw);
    
    // Create 2D versions of the transforms
    tf2::Transform current_2d, desired_2d;
    
    // Current pose in 2D
    current_2d.setOrigin(tf2::Vector3(current_pose.getOrigin().x(), 
                                      current_pose.getOrigin().y(), 
                                      0.0));
    tf2::Quaternion current_q_2d;
    current_q_2d.setRPY(0.0, 0.0, current_yaw);
    current_2d.setRotation(current_q_2d);
    
    // Desired pose in 2D
    desired_2d.setOrigin(tf2::Vector3(map_pose.getOrigin().x(), 
                                      map_pose.getOrigin().y(), 
                                      0.0));
    tf2::Quaternion desired_q_2d;
    desired_q_2d.setRPY(0.0, 0.0, desired_yaw);
    desired_2d.setRotation(desired_q_2d);
    
    // Calculate correction transform: correction = desired * current^-1
    _poseCorrection.mult(desired_2d, current_2d.inverse());
    _hasPoseCorrection = true;
    _lastCorrectionTime = get_clock()->now();
    
    if (_debugActive) {
      double cx, cy, cz;
      _poseCorrection.getBasis().getRPY(cx, cy, cz);
      
      // Calculate the rotation offset for clarity
      double yaw_offset = desired_yaw - current_yaw;
      // Normalize to [-PI, PI]
      while (yaw_offset > M_PI) yaw_offset -= 2.0 * M_PI;
      while (yaw_offset < -M_PI) yaw_offset += 2.0 * M_PI;
      
      RCLCPP_INFO(
        get_logger(),
        "2D Pose correction -> Pos: [%.3f,%.3f,0.0] - Yaw: %.3f° (offset: %.3f°)",
        _poseCorrection.getOrigin().x(), _poseCorrection.getOrigin().y(), 
        cz * RAD2DEG, yaw_offset * RAD2DEG);
      RCLCPP_INFO(
        get_logger(),
        "  Current pose: [%.3f,%.3f] yaw=%.3f° -> Desired: [%.3f,%.3f] yaw=%.3f°",
        current_pose.getOrigin().x(), current_pose.getOrigin().y(), current_yaw * RAD2DEG,
        map_pose.getOrigin().x(), map_pose.getOrigin().y(), desired_yaw * RAD2DEG);
    }
  } else {
    RCLCPP_WARN(get_logger(), "Could not get current pose from TF for correction calculation");
  }
  // <---- Calculate and store pose correction

  // ----> Reset camera position (COMMENTED OUT)
  // RCLCPP_INFO(get_logger(), "*** set_pose service call is disabled ***");
  // resetZedPose(map_pose);
  // <---- Reset camera position (COMMENTED OUT)

  // ----> Debug TF
  if (_debugActive) {
    geometry_msgs::msg::TransformStamped transformStamped;

    transformStamped.header.stamp = get_clock()->now();

    transformStamped.header.frame_id =
      _tagPoses[ids[nearest_aruco_index]].marker_frame_id;
    transformStamped.child_frame_id = _cameraName + "_left_aruco";

    transformStamped.transform.rotation.x = left_pose_marker.getRotation().x();
    transformStamped.transform.rotation.y = left_pose_marker.getRotation().y();
    transformStamped.transform.rotation.z = left_pose_marker.getRotation().z();
    transformStamped.transform.rotation.w = left_pose_marker.getRotation().w();

    transformStamped.transform.translation.x = left_pose_marker.getOrigin().x();
    transformStamped.transform.translation.y = left_pose_marker.getOrigin().y();
    transformStamped.transform.translation.z = left_pose_marker.getOrigin().z();

    // Broadcast debug TF
    _tfBroadcaster->sendTransform(transformStamped);

    transformStamped.header.frame_id =
      _tagPoses[ids[nearest_aruco_index]].marker_frame_id;
    transformStamped.child_frame_id = _cameraName + "_base_aruco";

    transformStamped.transform.rotation.x = base_pose_marker.getRotation().x();
    transformStamped.transform.rotation.y = base_pose_marker.getRotation().y();
    transformStamped.transform.rotation.z = base_pose_marker.getRotation().z();
    transformStamped.transform.rotation.w = base_pose_marker.getRotation().w();

    transformStamped.transform.translation.x = base_pose_marker.getOrigin().x();
    transformStamped.transform.translation.y = base_pose_marker.getOrigin().y();
    transformStamped.transform.translation.z = base_pose_marker.getOrigin().z();

    // Broadcast debug TF
    _tfBroadcaster->sendTransform(transformStamped);

    tf2::Transform test;
    getTransformFromTf(
      _cameraName + "_left_aruco", _cameraName + "_base_aruco",
      test);

    base_pose_marker.getBasis().getRPY(r, p, y);
    RCLCPP_INFO(
      get_logger(),
      "test TF -> Pos: [%.3f,%.3f,%.3f] - Or: [%.3f°,%.3f°,%.3f°]",
      test.getOrigin().x(), test.getOrigin().y(),
      test.getOrigin().z(), r * RAD2DEG, p * RAD2DEG, y * RAD2DEG);
  }
  // <---- Debug TF

  // ----> Draw and publish the results
  if (res_sub) {
    start = get_clock()->now();

    // Draw the markers
    cv::aruco::drawDetectedMarkers(bgr, corners, ids);
    auto rvec = rvecs[nearest_aruco_index];
    auto tvec = tvecs[nearest_aruco_index];
    cv::drawFrameAxes(bgr, camera_matrix, dist_coeffs, rvec, tvec, 0.1);

    // Create the output message and copy coverted data
    std::shared_ptr<sensor_msgs::msg::Image> out_bgr =
      std::make_shared<sensor_msgs::msg::Image>();

    out_bgr->header.stamp = img->header.stamp;
    out_bgr->header.frame_id = img->header.frame_id;
    out_bgr->height = bgr.rows;
    out_bgr->width = bgr.cols;

    int num = 1;  // for endianness detection
    out_bgr->is_bigendian = !(*reinterpret_cast<char *>(&num) == 1);

    out_bgr->step = bgr.step;

    size_t size = out_bgr->step * out_bgr->height;
    out_bgr->data.resize(size);

    out_bgr->encoding = sensor_msgs::image_encodings::BGR8;
    memcpy(reinterpret_cast<char *>((&out_bgr->data[0])), &bgr.data[0], size);

    // Publish the new image message coupled with camera info from the original
    // message
    _pubDetect.publish(out_bgr, cam_info);
    elapsed_sec = (get_clock()->now() - start).nanoseconds() / 1e9;
    if (_debugActive) {
      RCLCPP_INFO_STREAM(
        get_logger(),
        " * Publish image result: " << elapsed_sec << " sec");
    }
  }
  // <---- Draw and publish the results

  _detTime = get_clock()->now();

  if (_debugActive) {
    RCLCPP_INFO_STREAM(get_logger(), "*****************************");
  }

  // Detection completed and camera relocated
  _detRunning = false;
  
  } catch (const std::exception& e) {
    RCLCPP_ERROR(get_logger(), "Exception in camera callback: %s", e.what());
    _detRunning = false;
    return;
  } catch (...) {
    RCLCPP_ERROR(get_logger(), "Unknown exception in camera callback");
    _detRunning = false;
    return;
  }
}

void ZedArucoLoc::broadcastMarkerTFs()
{
  for (auto pose : _tagPoses) {
    geometry_msgs::msg::TransformStamped transformStamped;

    transformStamped.header.stamp = get_clock()->now();

    transformStamped.header.frame_id = _worldFrameId;
    transformStamped.child_frame_id = pose.second.marker_frame_id;

    tf2::Quaternion q;
    q.setRPY(
      pose.second.orientation[0], pose.second.orientation[1],
      pose.second.orientation[2]);

    transformStamped.transform.rotation.x = q.getX();
    transformStamped.transform.rotation.y = q.getY();
    transformStamped.transform.rotation.z = q.getZ();
    transformStamped.transform.rotation.w = q.getW();

    transformStamped.transform.translation.x = pose.second.position[0];
    transformStamped.transform.translation.y = pose.second.position[1];
    transformStamped.transform.translation.z = pose.second.position[2];

    _tfBroadcaster->sendTransform(transformStamped);
  }
}

bool ZedArucoLoc::getTransformFromTf(
  std::string targetFrame,
  std::string sourceFrame,
  tf2::Transform & out_tr)
{
  std::string msg;
  geometry_msgs::msg::TransformStamped transf_msg;

  try {
    // ----> Without this code a warning is returned the first time... why???
    _tfBuffer->canTransform(
      targetFrame, sourceFrame, TIMEZERO_ROS, 1000ms,
      &msg);
    RCLCPP_INFO_STREAM(
      get_logger(), "[getTransformFromTf] canTransform '"
        << targetFrame.c_str() << "' -> '"
        << sourceFrame.c_str()
        << "':" << msg.c_str());
    std::this_thread::sleep_for(3ms);
    // <---- Without this code a warning is returned the first time... why???

    transf_msg =
      _tfBuffer->lookupTransform(targetFrame, sourceFrame, TIMEZERO_ROS, 1s);
  } catch (const tf2::TransformException & ex) {
    RCLCPP_ERROR(
      this->get_logger(),
      "[getTransformFromTf] Could not transform '%s' to '%s': %s",
      targetFrame.c_str(), sourceFrame.c_str(), ex.what());
    return false;
  }

  tf2::Stamped<tf2::Transform> tr_stamped;
  tf2::fromMsg(transf_msg, tr_stamped);
  out_tr = tr_stamped;
  double r, p, y;
  out_tr.getBasis().getRPY(r, p, y, 1);

  RCLCPP_INFO(
    get_logger(),
    "[getTransformFromTf] '%s' -> '%s': \n\t[%.3f,%.3f,%.3f] - "
    "[%.3f°,%.3f°,%.3f°]",
    sourceFrame.c_str(), targetFrame.c_str(), out_tr.getOrigin().x(),
    out_tr.getOrigin().y(), out_tr.getOrigin().z(), r * RAD2DEG,
    p * RAD2DEG, y * RAD2DEG);

  return true;
}

void ZedArucoLoc::initTFs()
{
  // Get the transform from camera optical frame to camera base frame
  std::string cam_left_frame = _cameraName + "_left_camera_frame";
  std::string cam_base_frame = _cameraName + "_camera_link";
  bool tf_ok = getTransformFromTf(cam_left_frame, cam_base_frame, _left2base);
  if (!tf_ok) {
    RCLCPP_ERROR(
      get_logger(),
      "The transform '%s' -> '%s' is not available. Please verify the "
      "parameters and the status of the 'ZED State Publisher' node.",
      cam_base_frame.c_str(), cam_left_frame.c_str());
    exit(EXIT_FAILURE);
  }

  double r, p, y;
  tf2::Matrix3x3 basis;

  // ----> ArUco coordinate system to Image coordinate system, and viceversa
  basis = tf2::Matrix3x3(-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0);
  _img2aruco.setIdentity();
  _img2aruco.setBasis(basis);
  _aruco2img = _img2aruco.inverse();
  // <---- ArUco coordinate system to Image coordinate system, and viceversa

  // ----> ROS coordinate system to Image coordinate system, and viceversa
  basis = tf2::Matrix3x3(0.0, -1.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0);
  _ros2img.setIdentity();
  _ros2img.setBasis(basis);
  _img2ros = _ros2img.inverse();
  // <---- ROS coordinate system to Image coordinate system, and viceversa

  _img2aruco.getBasis().getRPY(r, p, y);
  RCLCPP_INFO(
    get_logger(), "_img2aruco: %.3f°,%.3f°,%.3f°", r * RAD2DEG,
    p * RAD2DEG, y * RAD2DEG);
  _aruco2img.getBasis().getRPY(r, p, y);
  RCLCPP_INFO(
    get_logger(), "_aruco2img: %.3f°,%.3f°,%.3f°", r * RAD2DEG,
    p * RAD2DEG, y * RAD2DEG);
  _ros2img.getBasis().getRPY(r, p, y);
  RCLCPP_INFO(
    get_logger(), "_ros2img: %.3f°,%.3f°,%.3f°", r * RAD2DEG,
    p * RAD2DEG, y * RAD2DEG);
  _img2ros.getBasis().getRPY(r, p, y);
  RCLCPP_INFO(
    get_logger(), "_img2ros: %.3f°,%.3f°,%.3f°", r * RAD2DEG,
    p * RAD2DEG, y * RAD2DEG);
}

bool ZedArucoLoc::resetZedPose(tf2::Transform & new_pose)
{
  // COMMENTED OUT - Service call disabled
  /*
  RCLCPP_INFO(get_logger(), "*** Calling ZED 'set_pose' service ***");

  auto request = std::make_shared<zed_msgs::srv::SetPose::Request>();
  request->pos[0] = new_pose.getOrigin().x();
  request->pos[1] = new_pose.getOrigin().y();
  request->pos[2] = new_pose.getOrigin().z();

  double r, p, y;
  new_pose.getBasis().getRPY(r, p, y);
  request->orient[0] = r;
  request->orient[1] = p;
  request->orient[2] = y;

  RCLCPP_INFO(
    get_logger(),
    " * New camera pose [%s]-> Pos:[%.3f,%.3f,%.3f] Or:[%.3f°,%.3f°,%.3f°]",
    _worldFrameId.c_str(), request->pos[0], request->pos[1], request->pos[2],
    request->orient[0] * RAD2DEG, request->orient[1] * RAD2DEG,
    request->orient[2] * RAD2DEG);

  while (!_setPoseClient->wait_for_service(1s)) {
    if (!rclcpp::ok()) {
      RCLCPP_ERROR(
        get_logger(),
        " * Interrupted while waiting for the service. Exiting.");
      return false;
    }
    RCLCPP_INFO_STREAM(
      get_logger(),
      " * '" << _setPoseClient->get_service_name()
             << "' service not available, waiting...");
  }

  // We give the async_send_request() method a callback (lambda function) that
  // will get executed once the response is received. This way we can return
  // immediately from this method and allow other work to be done by the
  // executor.
  using ServiceResponseFuture =
    rclcpp::Client<zed_msgs::srv::SetPose>::SharedFuture;
  auto response_received_callback = [this](ServiceResponseFuture future) {
      auto result = future.get();
      RCLCPP_INFO_STREAM(
        get_logger(), " * ZED Node replied to `set_pose` call: "
          << result->message.c_str());
    };

  auto future_result =
    _setPoseClient->async_send_request(request, response_received_callback);
  */
  
  RCLCPP_INFO(get_logger(), "*** set_pose service call is disabled ***");
  return true;
}

}  // namespace stereolabs

#include "rclcpp_components/register_node_macro.hpp"

// Register the component with class_loader.
// This acts as a sort of entry point, allowing the component to be discoverable
// when its library is being loaded into a running process.
RCLCPP_COMPONENTS_REGISTER_NODE(stereolabs::ZedArucoLoc)
