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
#include <sensor_msgs/image_encodings.hpp>
#include <std_msgs/msg/u_int32.hpp>
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

namespace
{

// Compute quadrilateral area using the Shoelace formula
double quadArea(const std::vector<cv::Point2f> & corners)
{
  double area = 0.0;
  int n = static_cast<int>(corners.size());
  for (int i = 0; i < n; i++) {
    int j = (i + 1) % n;
    area += corners[i].x * corners[j].y;
    area -= corners[j].x * corners[i].y;
  }
  return std::abs(area) / 2.0;
}

// Compute Laplacian variance on the marker's bounding rect ROI
// Low variance = blurry (edges are smeared, less high-frequency content)
double laplacianVariance(const cv::Mat & gray, const std::vector<cv::Point2f> & corners)
{
  cv::Rect bbox = cv::boundingRect(corners);
  // Clamp to image bounds
  bbox &= cv::Rect(0, 0, gray.cols, gray.rows);
  if (bbox.area() <= 0) {
    return 0.0;
  }

  cv::Mat roi = gray(bbox);
  cv::Mat lap;
  cv::Laplacian(roi, lap, CV_64F);

  cv::Scalar mean, stddev;
  cv::meanStdDev(lap, mean, stddev);
  return stddev.val[0] * stddev.val[0];  // variance
}

// Compute viewing angle between marker normal and camera optical axis.
// Returns angle in degrees. 0 = head-on, 90 = edge-on.
double obliqueAngle(const cv::Vec3d & rvec)
{
  cv::Mat R;
  cv::Rodrigues(rvec, R);
  // Marker normal in marker frame is [0,0,1].
  // In camera frame it becomes the third column of R.
  // Dot product with camera Z-axis [0,0,1] is simply R(2,2).
  double cos_angle = std::abs(R.at<double>(2, 2));
  cos_angle = std::min(cos_angle, 1.0);  // clamp for numerical safety
  return std::acos(cos_angle) * RAD2DEG;
}

// Compute RMS reprojection error: reproject 3D marker corners through the
// estimated pose and measure pixel distance to the detected corners.
double reprojectionError(
  const std::vector<cv::Point2f> & detected,
  const cv::Vec3d & rvec, const cv::Vec3d & tvec,
  const cv::Matx33d & cam_mat,
  const cv::Matx<float, 4, 1> & dist_coeffs,
  float marker_size)
{
  // Object points must match the convention in estimatePoseSingleMarkers
  std::vector<cv::Point3f> obj = {
    cv::Point3f(-marker_size / 2.f, marker_size / 2.f, 0),
    cv::Point3f(marker_size / 2.f, marker_size / 2.f, 0),
    cv::Point3f(marker_size / 2.f, -marker_size / 2.f, 0),
    cv::Point3f(-marker_size / 2.f, -marker_size / 2.f, 0)
  };

  std::vector<cv::Point2f> reproj;
  cv::projectPoints(obj, rvec, tvec, cam_mat, dist_coeffs, reproj);

  double sum_sq = 0.0;
  for (int j = 0; j < 4; j++) {
    double dx = detected[j].x - reproj[j].x;
    double dy = detected[j].y - reproj[j].y;
    sum_sq += dx * dx + dy * dy;
  }
  return std::sqrt(sum_sq / 4.0);
}

}  // anonymous namespace

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

  _defaultQoS.keep_last(10);
  _defaultQoS.reliable();
  _defaultQoS.durability_volatile();

  _detTime = get_clock()->now();

  getParams();

  // ----> TF2 Transform
  _tfBuffer = std::make_unique<tf2_ros::Buffer>(get_clock());
  _tfListener = std::make_unique<tf2_ros::TransformListener>(*_tfBuffer);
  _tfBroadcaster = std::make_unique<tf2_ros::TransformBroadcaster>(this);

  initTFs();

  int msec = static_cast<int>(1000. / (_detRate * 10.));
  _tfTimer = create_wall_timer(
    std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::milliseconds(msec)),
    std::bind(&ZedArucoLoc::broadcastMarkerTFs, this));
  // <---- TF2 Transform

  _pubDetect = image_transport::create_camera_publisher(
    this, "out/aruco_result", _defaultQoS.get_rmw_qos_profile());
  RCLCPP_INFO_STREAM(get_logger(), "Advertised on topic: " << _pubDetect.getTopic());

  _pubCount = create_publisher<std_msgs::msg::UInt32>("/zed/aruco/count", _defaultQoS);
  RCLCPP_INFO_STREAM(get_logger(), "Advertised on topic: " << _pubCount->get_topic_name());

  _arucoCount = 0;
  auto init_msg = std_msgs::msg::UInt32();
  init_msg.data = _arucoCount;
  _pubCount->publish(init_msg);

  _subImage = image_transport::create_camera_subscription(
    this, "in/zed_image",
    std::bind(&ZedArucoLoc::camera_callback, this, _1, _2), "raw",
    _defaultQoS.get_rmw_qos_profile());
  RCLCPP_INFO_STREAM(get_logger(), "Subscribed to topic: " << _subImage.getTopic());

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
      "The parameter '" << paramName
                        << "' is not available or is not valid, using default: "
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
    rcutils_logging_set_logger_level(
      get_logger().get_name(), RCUTILS_LOG_SEVERITY_INFO);
  }

  getGeneralParams();
  getQualityFilterParams();
  getMarkerParams();
}

void ZedArucoLoc::getGeneralParams()
{
  RCLCPP_INFO(get_logger(), "*** GENERAL parameters ***");

  getParam("general.marker_count", _markerCount, _markerCount, " * Marker count: ");
  getParam("general.marker_size", _markerSize, _markerSize, " * Marker size [m]: ");
  getParam("general.detection_rate", _detRate, _detRate, " * Detection rate [Hz]: ");
  getParam("general.camera_name", _cameraName, _cameraName, " * Camera name: ");
  getParam("general.world_frame_id", _worldFrameId, _worldFrameId, " * World frame id: ");
  getParam("general.maximum_distance", _maxDist, _maxDist, " * Maximum distance [m]: ");
  getParam("general.refine_detection", _refineDetection, _refineDetection);
  RCLCPP_INFO(get_logger(), " * Refine detection: %s", _refineDetection ? "TRUE" : "FALSE");
}

void ZedArucoLoc::getQualityFilterParams()
{
  RCLCPP_INFO(get_logger(), "*** QUALITY FILTER parameters ***");

  // Shared (resolution-independent)
  getParam(
    "quality_filters.max_oblique_angle", _maxObliqueAngle,
    _maxObliqueAngle, " * Max oblique angle [deg]: ");
  getParam(
    "quality_filters.same_marker_cooldown", _sameMarkerCooldown,
    _sameMarkerCooldown, " * Same marker cooldown [s]: ");

  // HD1080
  getParam(
    "quality_filters_1080.min_laplacian_variance", _minLaplacianVariance_1080,
    _minLaplacianVariance_1080, " * [1080] Min Laplacian variance: ");
  getParam(
    "quality_filters_1080.max_reprojection_error", _maxReprojError_1080,
    _maxReprojError_1080, " * [1080] Max reprojection error [px]: ");
  getParam(
    "quality_filters_1080.min_marker_area", _minMarkerArea_1080,
    _minMarkerArea_1080, " * [1080] Min marker area [px^2]: ");

  // HD720
  getParam(
    "quality_filters_720.min_laplacian_variance", _minLaplacianVariance_720,
    _minLaplacianVariance_720, " * [720] Min Laplacian variance: ");
  getParam(
    "quality_filters_720.max_reprojection_error", _maxReprojError_720,
    _maxReprojError_720, " * [720] Max reprojection error [px]: ");
  getParam(
    "quality_filters_720.min_marker_area", _minMarkerArea_720,
    _minMarkerArea_720, " * [720] Min marker area [px^2]: ");

  RCLCPP_INFO(get_logger(), " * Resolution-dependent params will be applied on first frame");
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
    declare_parameter(par_name, rclcpp::ParameterValue(aruco_pose.position), read_only_descriptor);
    if (!get_parameter(par_name, aruco_pose.position)) {
      RCLCPP_ERROR_STREAM(get_logger(), "The parameter '" << par_name << "' is not available.");
      exit(EXIT_FAILURE);
    }
    if (aruco_pose.position.size() != 3) {
      RCLCPP_ERROR_STREAM(get_logger(), "'" << par_name << "' must have 3 FLOAT64 values.");
      exit(EXIT_FAILURE);
    }
    RCLCPP_INFO(
      get_logger(), "   * Position: [%g,%g,%g]",
      aruco_pose.position[0], aruco_pose.position[1], aruco_pose.position[2]);

    par_name = par_pref + "orientation";
    declare_parameter(par_name, rclcpp::ParameterValue(aruco_pose.orientation), read_only_descriptor);
    if (!get_parameter(par_name, aruco_pose.orientation)) {
      RCLCPP_ERROR_STREAM(get_logger(), "The parameter '" << par_name << "' is not available.");
      exit(EXIT_FAILURE);
    }
    if (aruco_pose.orientation.size() != 3) {
      RCLCPP_ERROR_STREAM(get_logger(), "'" << par_name << "' must have 3 FLOAT64 values.");
      exit(EXIT_FAILURE);
    }
    RCLCPP_INFO(
      get_logger(), "   * Orientation: [%g,%g,%g]",
      aruco_pose.orientation[0], aruco_pose.orientation[1], aruco_pose.orientation[2]);

    _tagPoses[aruco_pose.idx] = aruco_pose;
  }
}

void ZedArucoLoc::applyResolutionParams(int image_width, int image_height)
{
  // Classify by height: 1080 or 720. If neither matches exactly, pick closest.
  std::string res_label;
  if (image_height >= 900) {
    // HD1080 (1920x1080) or higher
    _minLaplacianVariance = _minLaplacianVariance_1080;
    _maxReprojError = _maxReprojError_1080;
    _minMarkerArea = _minMarkerArea_1080;
    res_label = "HD1080";
  } else {
    // HD720 (1280x720) or lower
    _minLaplacianVariance = _minLaplacianVariance_720;
    _maxReprojError = _maxReprojError_720;
    _minMarkerArea = _minMarkerArea_720;
    res_label = "HD720";
  }

  _resolutionDetected = true;

  RCLCPP_INFO(get_logger(), "*** Resolution detected: %dx%d -> using %s quality params ***",
    image_width, image_height, res_label.c_str());
  RCLCPP_INFO(get_logger(), " * Min Laplacian variance: %.1f", _minLaplacianVariance);
  RCLCPP_INFO(get_logger(), " * Max reprojection error: %.2f px", _maxReprojError);
  RCLCPP_INFO(get_logger(), " * Min marker area: %.0f px^2", _minMarkerArea);
  RCLCPP_INFO(get_logger(), " * Max oblique angle: %.1f deg", _maxObliqueAngle);
  RCLCPP_INFO(get_logger(), " * Same marker cooldown: %.1f s", _sameMarkerCooldown);
}

void ZedArucoLoc::camera_callback(
  const sensor_msgs::msg::Image::ConstSharedPtr & img,
  const sensor_msgs::msg::CameraInfo::ConstSharedPtr & cam_info)
{
  if (img->encoding != sensor_msgs::image_encodings::BGRA8) {
    RCLCPP_ERROR(get_logger(), "Input image requires 'BGRA8' encoding");
    exit(EXIT_FAILURE);
  }

  // Auto-detect resolution on first frame and apply matching quality params
  if (!_resolutionDetected) {
    applyResolutionParams(static_cast<int>(img->width), static_cast<int>(img->height));
  }

  if (_detRunning) {
    return;
  }

  double det_elapsed_sec = (get_clock()->now() - _detTime).nanoseconds() / 1e9;
  if (det_elapsed_sec < (1.0 / _detRate)) {
    return;
  }

  _detRunning = true;

  bool res_sub = (_pubDetect.getNumSubscribers() > 0);

  // ----> Convert BGRA to grayscale and BGR
  void * data = const_cast<void *>(reinterpret_cast<const void *>(&img->data[0]));
  cv::Mat bgra(img->height, img->width, CV_8UC4, data);
  cv::Mat bgr, gray;
  cv::cvtColor(bgra, gray, cv::COLOR_BGRA2GRAY);
  cv::cvtColor(bgra, bgr, cv::COLOR_BGRA2BGR);
  // <---- Convert

  // ----> Detect ArUco Markers
  std::vector<int> ids;
  std::vector<std::vector<cv::Point2f>> corners;

  auto dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_100);
  cv::aruco::detectMarkers(bgr, dictionary, corners, ids);
  // <---- Detect ArUco Markers

  if (corners.empty()) {
    _detRunning = false;
    return;
  }

  // ----> Optional subpixel refinement (keep false for motion blur scenarios)
  if (_refineDetection) {
    for (size_t i = 0; i < corners.size(); ++i) {
      cv::cornerSubPix(
        gray, corners[i], cv::Size(5, 5), cv::Size(-1, -1),
        cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.1));
    }
    if (corners.empty()) {
      _detRunning = false;
      return;
    }
  }
  // <---- Refine

  // ----> Estimate pose for all detected markers
  std::vector<cv::Vec3d> rvecs, tvecs;

  cv::Matx33d camera_matrix = cv::Matx33d::eye();
  camera_matrix(0, 0) = cam_info->k[0];
  camera_matrix(1, 1) = cam_info->k[4];
  camera_matrix(0, 2) = cam_info->k[2];
  camera_matrix(1, 2) = cam_info->k[5];
  cv::Matx<float, 4, 1> dist_coeffs = cv::Vec4f::zeros();

  cv::aruco::estimatePoseSingleMarkers(
    corners, _markerSize, camera_matrix, dist_coeffs, rvecs, tvecs);
  // <---- Estimate poses

  // =====================================================================
  // Quality filtering: reject blurry, oblique, or poorly estimated markers
  // =====================================================================
  // For each detected marker we compute four quality metrics and reject
  // markers that fail any check. Only markers present in our config AND
  // passing all filters are candidates for pose reset.

  struct ValidMarker {
    size_t idx;       // index into ids/corners/rvecs/tvecs
    double distance;  // 3D distance from camera
    double area;
    double lap_var;
    double angle_deg;
    double reproj_err;
  };
  std::vector<ValidMarker> valid_markers;

  if (_debugActive) {
    RCLCPP_DEBUG(get_logger(), "--- Quality filter: %zu raw detections ---", ids.size());
  }

  for (size_t i = 0; i < ids.size(); i++) {
    int id = ids[i];

    // Filter 0: Is this marker in our config?
    if (_tagPoses.find(id) == _tagPoses.end()) {
      if (_debugActive) {
        RCLCPP_DEBUG(get_logger(), "  [#%d] SKIP: not in config", id);
      }
      continue;
    }

    // Filter 1: Minimum marker pixel area
    double area = quadArea(corners[i]);
    if (area < _minMarkerArea) {
      if (_debugActive) {
        RCLCPP_DEBUG(get_logger(), "  [#%d] REJECT area=%.0f < %.0f", id, area, _minMarkerArea);
      }
      continue;
    }

    // Filter 2: Laplacian blur check
    double lap_var = laplacianVariance(gray, corners[i]);
    if (lap_var < _minLaplacianVariance) {
      if (_debugActive) {
        RCLCPP_DEBUG(
          get_logger(), "  [#%d] REJECT blur lap_var=%.1f < %.1f",
          id, lap_var, _minLaplacianVariance);
      }
      continue;
    }

    // Filter 3: Oblique viewing angle
    double angle_deg = obliqueAngle(rvecs[i]);
    if (angle_deg > _maxObliqueAngle) {
      if (_debugActive) {
        RCLCPP_DEBUG(
          get_logger(), "  [#%d] REJECT oblique=%.1f° > %.1f°",
          id, angle_deg, _maxObliqueAngle);
      }
      continue;
    }

    // Filter 4: Reprojection error
    double reproj = reprojectionError(
      corners[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs, _markerSize);
    if (reproj > _maxReprojError) {
      if (_debugActive) {
        RCLCPP_DEBUG(
          get_logger(), "  [#%d] REJECT reproj=%.2f px > %.2f px",
          id, reproj, _maxReprojError);
      }
      continue;
    }

    // Compute 3D distance
    double x = tvecs[i](0), y = tvecs[i](1), z = tvecs[i](2);
    double dist = std::sqrt(x * x + y * y + z * z);

    // Filter 5: Maximum distance (existing check, moved here)
    if (dist > _maxDist) {
      if (_debugActive) {
        RCLCPP_DEBUG(
          get_logger(), "  [#%d] REJECT dist=%.2f m > %.2f m",
          id, dist, _maxDist);
      }
      continue;
    }

    // All filters passed
    valid_markers.push_back({i, dist, area, lap_var, angle_deg, reproj});

    if (_debugActive) {
      RCLCPP_DEBUG(
        get_logger(),
        "  [#%d] PASS area=%.0f lap=%.1f angle=%.1f° reproj=%.2fpx dist=%.2fm",
        id, area, lap_var, angle_deg, reproj, dist);
    }
  }

  if (valid_markers.empty()) {
    if (_debugActive) {
      RCLCPP_DEBUG(get_logger(), "--- No markers passed quality filters ---");
    }
    _detRunning = false;
    return;
  }

  // ----> Select the nearest valid marker
  size_t best = 0;
  for (size_t v = 1; v < valid_markers.size(); v++) {
    if (valid_markers[v].distance < valid_markers[best].distance) {
      best = v;
    }
  }

  size_t sel = valid_markers[best].idx;
  int sel_id = ids[sel];
  double sel_reproj = valid_markers[best].reproj_err;

  // ----> Same-marker suppression
  // If we last reset from this exact marker, only re-accept if:
  //   (a) the new reprojection error is strictly better, OR
  //   (b) the cooldown period has elapsed.
  // A different marker is always accepted immediately.
  auto prev = _lastResetInfo.find(sel_id);
  if (prev != _lastResetInfo.end()) {
    double elapsed = (get_clock()->now() - prev->second.timestamp).nanoseconds() / 1e9;
    bool better_reproj = sel_reproj < prev->second.reproj_error;
    bool cooldown_passed = elapsed >= _sameMarkerCooldown;

    if (!better_reproj && !cooldown_passed) {
      if (_debugActive) {
        RCLCPP_DEBUG(
          get_logger(),
          "  [#%d] SUPPRESSED same marker: reproj=%.2f >= prev=%.2f, cooldown %.1f/%.1fs",
          sel_id, sel_reproj, prev->second.reproj_error, elapsed, _sameMarkerCooldown);
      }
      _detRunning = false;
      return;
    }

    if (_debugActive) {
      RCLCPP_DEBUG(
        get_logger(),
        "  [#%d] Re-accepted: %s (reproj %.2f vs prev %.2f, elapsed %.1fs)",
        sel_id,
        better_reproj ? "better reproj" : "cooldown passed",
        sel_reproj, prev->second.reproj_error, elapsed);
    }
  }
  // <---- Same-marker suppression

  RCLCPP_INFO(
    get_logger(),
    "ArUco #%d accepted | dist=%.2fm area=%.0f lap=%.1f angle=%.1f° reproj=%.2fpx",
    sel_id, valid_markers[best].distance, valid_markers[best].area,
    valid_markers[best].lap_var, valid_markers[best].angle_deg,
    sel_reproj);
  // <---- Select nearest

  double r, p, y;

  // ----> ArUco rvec/tvec to TF2 Transform
  tf2::Vector3 tf2_origin(tvecs[sel][0], tvecs[sel][1], tvecs[sel][2]);
  cv::Mat cv_rot(3, 3, CV_64F);
  cv::Rodrigues(rvecs[sel], cv_rot);
  tf2::Matrix3x3 tf2_rot(
    cv_rot.at<double>(0, 0), cv_rot.at<double>(0, 1), cv_rot.at<double>(0, 2),
    cv_rot.at<double>(1, 0), cv_rot.at<double>(1, 1), cv_rot.at<double>(1, 2),
    cv_rot.at<double>(2, 0), cv_rot.at<double>(2, 1), cv_rot.at<double>(2, 2));

  tf2::Transform pose_aruco(tf2_rot, tf2_origin);
  // <---- ArUco to TF2

  // ----> Change basis from ArUco to camera in image coordinate system
  tf2::Transform pose_img;
  pose_img.mult(_img2aruco, pose_aruco);
  pose_img = pose_img.inverse();
  // <---- Change basis

  // ----> ArUco pose in ROS coordinate system
  tf2::Transform ros2aruco;
  ros2aruco.mult(_img2aruco, _ros2img);
  tf2::Transform aruco2ros = ros2aruco.inverse();
  // <---- ArUco in ROS

  // ----> Left camera sensor in ROS coordinate w.r.t. marker
  tf2::Transform left_pose_marker;
  left_pose_marker.mult(pose_img, ros2aruco);
  left_pose_marker.mult(aruco2ros, left_pose_marker);
  // <---- Left camera w.r.t. marker

  // ----> Camera base in ROS coordinate w.r.t. marker
  tf2::Transform base_pose_marker;
  base_pose_marker.mult(left_pose_marker, _left2base);
  // <---- Camera base w.r.t. marker

  // ----> New camera pose in ROS world
  tf2::Transform marker_world_pose;
  tf2::Vector3 orig(
    _tagPoses[sel_id].position[0],
    _tagPoses[sel_id].position[1],
    _tagPoses[sel_id].position[2]);
  marker_world_pose.setOrigin(orig);

  tf2::Quaternion q;
  q.setRPY(
    _tagPoses[sel_id].orientation[0],
    _tagPoses[sel_id].orientation[1],
    _tagPoses[sel_id].orientation[2]);
  marker_world_pose.setRotation(q);

  tf2::Transform map_pose;
  map_pose.mult(marker_world_pose, base_pose_marker);
  // <---- New camera pose

  // ----> Reset camera position
  resetZedPose(map_pose);
  // <---- Reset camera position

  // ----> Update per-marker reset tracking
  _lastResetInfo[sel_id] = {sel_reproj, get_clock()->now()};
  // <---- Update tracking

  // ----> Increment and publish ArUco count
  _arucoCount++;
  auto count_msg = std_msgs::msg::UInt32();
  count_msg.data = _arucoCount;
  _pubCount->publish(count_msg);
  // <---- Publish count

  // ----> Debug TF
  if (_debugActive) {
    geometry_msgs::msg::TransformStamped ts;
    ts.header.stamp = get_clock()->now();

    ts.header.frame_id = _tagPoses[sel_id].marker_frame_id;
    ts.child_frame_id = _cameraName + "_left_aruco";
    ts.transform.rotation.x = left_pose_marker.getRotation().x();
    ts.transform.rotation.y = left_pose_marker.getRotation().y();
    ts.transform.rotation.z = left_pose_marker.getRotation().z();
    ts.transform.rotation.w = left_pose_marker.getRotation().w();
    ts.transform.translation.x = left_pose_marker.getOrigin().x();
    ts.transform.translation.y = left_pose_marker.getOrigin().y();
    ts.transform.translation.z = left_pose_marker.getOrigin().z();
    _tfBroadcaster->sendTransform(ts);

    ts.child_frame_id = _cameraName + "_base_aruco";
    ts.transform.rotation.x = base_pose_marker.getRotation().x();
    ts.transform.rotation.y = base_pose_marker.getRotation().y();
    ts.transform.rotation.z = base_pose_marker.getRotation().z();
    ts.transform.rotation.w = base_pose_marker.getRotation().w();
    ts.transform.translation.x = base_pose_marker.getOrigin().x();
    ts.transform.translation.y = base_pose_marker.getOrigin().y();
    ts.transform.translation.z = base_pose_marker.getOrigin().z();
    _tfBroadcaster->sendTransform(ts);
  }
  // <---- Debug TF

  // ----> Draw and publish results (only valid markers + axes on selected)
  if (res_sub) {
    // Draw only the markers that passed quality filters
    std::vector<int> valid_ids;
    std::vector<std::vector<cv::Point2f>> valid_corners;
    for (const auto & vm : valid_markers) {
      valid_ids.push_back(ids[vm.idx]);
      valid_corners.push_back(corners[vm.idx]);
    }
    cv::aruco::drawDetectedMarkers(bgr, valid_corners, valid_ids);

    // Draw coordinate axes only on the selected marker
    cv::drawFrameAxes(bgr, camera_matrix, dist_coeffs, rvecs[sel], tvecs[sel], 0.1);

    std::shared_ptr<sensor_msgs::msg::Image> out_bgr =
      std::make_shared<sensor_msgs::msg::Image>();
    out_bgr->header.stamp = img->header.stamp;
    out_bgr->header.frame_id = img->header.frame_id;
    out_bgr->height = bgr.rows;
    out_bgr->width = bgr.cols;
    int num = 1;
    out_bgr->is_bigendian = !(*reinterpret_cast<char *>(&num) == 1);
    out_bgr->step = bgr.step;
    size_t size = out_bgr->step * out_bgr->height;
    out_bgr->data.resize(size);
    out_bgr->encoding = sensor_msgs::image_encodings::BGR8;
    memcpy(reinterpret_cast<char *>(&out_bgr->data[0]), &bgr.data[0], size);

    _pubDetect.publish(out_bgr, cam_info);
  }
  // <---- Draw and publish

  _detTime = get_clock()->now();
  _detRunning = false;
}

void ZedArucoLoc::broadcastMarkerTFs()
{
  for (auto pose : _tagPoses) {
    geometry_msgs::msg::TransformStamped ts;
    ts.header.stamp = get_clock()->now();
    ts.header.frame_id = _worldFrameId;
    ts.child_frame_id = pose.second.marker_frame_id;

    tf2::Quaternion q;
    q.setRPY(pose.second.orientation[0], pose.second.orientation[1], pose.second.orientation[2]);
    ts.transform.rotation.x = q.getX();
    ts.transform.rotation.y = q.getY();
    ts.transform.rotation.z = q.getZ();
    ts.transform.rotation.w = q.getW();
    ts.transform.translation.x = pose.second.position[0];
    ts.transform.translation.y = pose.second.position[1];
    ts.transform.translation.z = pose.second.position[2];

    _tfBroadcaster->sendTransform(ts);
  }
}

bool ZedArucoLoc::getTransformFromTf(
  std::string targetFrame, std::string sourceFrame, tf2::Transform & out_tr)
{
  std::string msg;
  geometry_msgs::msg::TransformStamped transf_msg;

  try {
    _tfBuffer->canTransform(targetFrame, sourceFrame, TIMEZERO_ROS, 1000ms, &msg);
    std::this_thread::sleep_for(3ms);
    transf_msg = _tfBuffer->lookupTransform(targetFrame, sourceFrame, TIMEZERO_ROS, 1s);
  } catch (const tf2::TransformException & ex) {
    RCLCPP_ERROR(
      this->get_logger(), "Could not transform '%s' to '%s': %s",
      targetFrame.c_str(), sourceFrame.c_str(), ex.what());
    return false;
  }

  tf2::Stamped<tf2::Transform> tr_stamped;
  tf2::fromMsg(transf_msg, tr_stamped);
  out_tr = tr_stamped;
  double r, p, y;
  out_tr.getBasis().getRPY(r, p, y, 1);

  RCLCPP_INFO(
    get_logger(), "TF '%s' -> '%s': [%.3f,%.3f,%.3f] [%.1f°,%.1f°,%.1f°]",
    sourceFrame.c_str(), targetFrame.c_str(),
    out_tr.getOrigin().x(), out_tr.getOrigin().y(), out_tr.getOrigin().z(),
    r * RAD2DEG, p * RAD2DEG, y * RAD2DEG);

  return true;
}

void ZedArucoLoc::initTFs()
{
  std::string cam_left_frame = _cameraName + "_left_camera_frame";
  std::string cam_base_frame = _cameraName + "_camera_link";
  bool tf_ok = getTransformFromTf(cam_left_frame, cam_base_frame, _left2base);
  if (!tf_ok) {
    RCLCPP_ERROR(
      get_logger(),
      "Transform '%s' -> '%s' not available. Check ZED State Publisher.",
      cam_base_frame.c_str(), cam_left_frame.c_str());
    exit(EXIT_FAILURE);
  }

  tf2::Matrix3x3 basis;

  // ArUco coordinate system <-> Image coordinate system
  basis = tf2::Matrix3x3(-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0);
  _img2aruco.setIdentity();
  _img2aruco.setBasis(basis);
  _aruco2img = _img2aruco.inverse();

  // ROS coordinate system <-> Image coordinate system
  basis = tf2::Matrix3x3(0.0, -1.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0);
  _ros2img.setIdentity();
  _ros2img.setBasis(basis);
  _img2ros = _ros2img.inverse();

  if (_debugActive) {
    double r, p, y;
    _img2aruco.getBasis().getRPY(r, p, y);
    RCLCPP_DEBUG(get_logger(), "_img2aruco: %.1f°,%.1f°,%.1f°", r * RAD2DEG, p * RAD2DEG, y * RAD2DEG);
    _ros2img.getBasis().getRPY(r, p, y);
    RCLCPP_DEBUG(get_logger(), "_ros2img: %.1f°,%.1f°,%.1f°", r * RAD2DEG, p * RAD2DEG, y * RAD2DEG);
  }
}

bool ZedArucoLoc::resetZedPose(tf2::Transform & new_pose)
{
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
    get_logger(), "set_pose -> Pos:[%.2f,%.2f,%.2f] Or:[%.1f°,%.1f°,%.1f°]",
    request->pos[0], request->pos[1], request->pos[2],
    r * RAD2DEG, p * RAD2DEG, y * RAD2DEG);

  while (!_setPoseClient->wait_for_service(1s)) {
    if (!rclcpp::ok()) {
      RCLCPP_ERROR(get_logger(), "Interrupted while waiting for set_pose service.");
      return false;
    }
    RCLCPP_INFO(get_logger(), "Waiting for '%s' service...", _setPoseClient->get_service_name());
  }

  using ServiceResponseFuture = rclcpp::Client<zed_msgs::srv::SetPose>::SharedFuture;
  auto cb = [this](ServiceResponseFuture future) {
      auto result = future.get();
      RCLCPP_INFO(get_logger(), "set_pose response: %s", result->message.c_str());
    };

  _setPoseClient->async_send_request(request, cb);
  return true;
}

}  // namespace stereolabs

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(stereolabs::ZedArucoLoc)