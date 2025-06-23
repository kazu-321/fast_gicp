#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/io/pcd_io.h>
#include <rclcpp/qos.hpp>
#include <fast_gicp/gicp/fast_vgicp_cuda.hpp>
#include <fast_gicp/ndt/ndt_cuda.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2/convert.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <autoware_internal_debug_msgs/msg/float32_stamped.hpp>
#include <autoware_internal_debug_msgs/msg/int32_stamped.hpp>
#include "autoware/localization_util/util_func.hpp"

#define TARGET_TOPIC "/localization/util/downsample/pointcloud"
// #define PCD_FILE "/home/kazusahashimoto/autoware_map/sample-map-rosbag/pointcloud_map.pcd"
#define PCD_FILE "/home/kazusahashimoto/ros2_ws/shinagawa_odaiba/map.pcd"


using autoware::localization_util::pose_to_matrix4f;

class AlignNode : public rclcpp::Node {
public:
  AlignNode() : Node("align_node") {
    source_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      TARGET_TOPIC, rclcpp::SensorDataQoS(), std::bind(&AlignNode::targetCallback, this, std::placeholders::_1));
    result_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/vgicp/aligned_cloud", 10);
    source_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/vgicp/source_cloud", 10);
    target_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/vgicp/target_cloud", 10);
    pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("/vgicp/pose", 10);
    fitness_pub_ = this->create_publisher<autoware_internal_debug_msgs::msg::Float32Stamped>("/vgicp/fitness_score", 10);
    exec_time_pub_ = this->create_publisher<autoware_internal_debug_msgs::msg::Float32Stamped>("/vgicp/exec_time_ms", 10);
    iter_pub_ = this->create_publisher<autoware_internal_debug_msgs::msg::Int32Stamped>("/vgicp/iter", 10);
    pose_cov_pub_ = this->create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>("/vgicp/pose_with_covariance", 10);
    pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
      "/localization/pose_with_covariance", rclcpp::SensorDataQoS(), [this](const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg) {
        initial_guess_ = pose_to_matrix4f(msg->pose.pose);
        RCLCPP_INFO(this->get_logger(), "Initial guess updated: %f %f %f %f %f %f %f", initial_guess_(0, 0), initial_guess_(1, 0), initial_guess_(2, 0), initial_guess_(3, 0), initial_guess_(4, 0), initial_guess_(5, 0), initial_guess_(6, 0));
      });
    
    
    tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);
    
    // vgicp.setNumThreads(omp_get_max_threads());
    // Load source cloud from PCD file
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(PCD_FILE, *target_cloud_) == -1) {
      RCLCPP_ERROR(this->get_logger(), "Couldn't read source PCD file");
      throw std::runtime_error("Failed to load source PCD file");
    }
    // vgicp.setResolution(1.0);
    double resolution;
    int max_iterations;

    this->declare_parameter("vgicp.resolution", 4.0);
    this->get_parameter("vgicp.resolution", resolution);
    this->declare_parameter("vgicp.max_iterations", 50);
    this->get_parameter("vgicp.max_iterations", max_iterations);

    vgicp.clearTarget();
    vgicp.setInputTarget(target_cloud_);
    // vgicp.setNumThreads(4);
    vgicp.setResolution(resolution);
    vgicp.setMaxIterations(max_iterations);
    vgicp.setNearestNeighborSearchMethod(fast_gicp::NearestNeighborMethod::GPU_BRUTEFORCE);
    // vgicp.setDistanceMode(fast_gicp::NDTDistanceMode::P2D);
    RCLCPP_INFO(this->get_logger(), "Loaded source cloud with %zu points", target_cloud_->size());
  }
  
  void targetCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    pcl::fromROSMsg(*msg, *source_cloud_);
    // RCLCPP_INFO(this->get_logger(), "Received target cloud with %zu points", source_cloud_->size());
    
    if (target_cloud_->empty() || source_cloud_->empty()) {
      RCLCPP_WARN(this->get_logger(), "Target or source cloud is empty, skipping alignment");
      return;
    }
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr aligned(new pcl::PointCloud<pcl::PointXYZ>());
    auto t1 = std::chrono::high_resolution_clock::now();
    
    vgicp.clearSource();
    vgicp.setInputSource(source_cloud_);

    vgicp.align(*aligned, initial_guess_);

    auto t2 = std::chrono::high_resolution_clock::now();
    double dt = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1e6;

    // RCLCPP_INFO(this->get_logger(), "Alignment took %f ms", dt);
    if (!vgicp.hasConverged()) {
      RCLCPP_WARN(this->get_logger(), "Alignment failed");
      return;
    }

    autoware_internal_debug_msgs::msg::Float32Stamped exec_time_msg;
    exec_time_msg.stamp = this->get_clock()->now();
    exec_time_msg.data = dt;
    exec_time_pub_->publish(exec_time_msg);

    autoware_internal_debug_msgs::msg::Int32Stamped iter_msg;
    iter_msg.stamp = this->get_clock()->now();
    iter_msg.data = vgicp.getFinalNumIteration();
    iter_pub_->publish(iter_msg);

    
    // publish tf transform map -> base_link
    Eigen::Matrix4f transform = vgicp.getFinalTransformation();
    Eigen::Quaternionf q(transform.block<3, 3>(0, 0));
    Eigen::Vector3f t = transform.block<3, 1>(0, 3);
    
    // fitness score
    double fitness_score = vgicp.getFitnessScore();
    RCLCPP_INFO(this->get_logger(), "Fitness score: %f", fitness_score);

    autoware_internal_debug_msgs::msg::Float32Stamped fitness_msg;
    fitness_msg.stamp = this->get_clock()->now();
    fitness_msg.data = fitness_score;
    fitness_pub_->publish(fitness_msg);

    // info xyz
    RCLCPP_INFO(this->get_logger(), "Translation: [%f, %f, %f]", t.x(), t.y(), t.z());
    
    geometry_msgs::msg::TransformStamped tf_msg;
    tf_msg.header.stamp = this->get_clock()->now();
    tf_msg.header.frame_id = "map";
    tf_msg.child_frame_id = "vgicp_base_link";
    tf_msg.transform.translation.x = t.x();
    tf_msg.transform.translation.y = t.y();
    tf_msg.transform.translation.z = t.z();
    tf_msg.transform.rotation.x = q.x();
    tf_msg.transform.rotation.y = q.y();
    tf_msg.transform.rotation.z = q.z();
    tf_msg.transform.rotation.w = q.w();

    
    tf_broadcaster_->sendTransform(tf_msg);
    
    // publish pose
    geometry_msgs::msg::PoseStamped pose_msg;
    pose_msg.header.stamp = this->get_clock()->now();
    pose_msg.header.frame_id = "map";
    pose_msg.pose.position.x = t.x();
    pose_msg.pose.position.y = t.y();
    pose_msg.pose.position.z = t.z();
    pose_msg.pose.orientation.x = q.x();
    pose_msg.pose.orientation.y = q.y();
    pose_msg.pose.orientation.z = q.z();
    pose_msg.pose.orientation.w = q.w();
    pose_pub_->publish(pose_msg);

    // publish pose with covariance
    geometry_msgs::msg::PoseWithCovarianceStamped pose_cov_msg;
    pose_cov_msg.header.stamp = this->get_clock()->now();
    pose_cov_msg.header.frame_id = "map";
    pose_cov_msg.pose.pose.position.x = t.x();
    pose_cov_msg.pose.pose.position.y = t.y();
    pose_cov_msg.pose.pose.position.z = t.z();
    pose_cov_msg.pose.pose.orientation.x = q.x();
    pose_cov_msg.pose.pose.orientation.y = q.y();
    pose_cov_msg.pose.pose.orientation.z = q.z();
    pose_cov_msg.pose.pose.orientation.w = q.w();

    // get_final_hessian
    Eigen::Matrix<double, 6, 6> hessian = vgicp.getFinalHessian();
    Eigen::Matrix<double, 6, 1> hessian_diag = hessian.diagonal();
    for (int i = 0; i < 6; ++i) {
      pose_cov_msg.pose.covariance[i * 6 + i] = hessian_diag(i) > 1e-6 ? 1.0 / hessian_diag(i) : 1e6;
    }
    pose_cov_pub_->publish(pose_cov_msg);
    
    sensor_msgs::msg::PointCloud2 output_msg;
    pcl::toROSMsg(*aligned, output_msg);
    output_msg.header.frame_id = "map";
    output_msg.header.stamp = this->get_clock()->now();
    result_pub_->publish(output_msg);
    
    // publish source and target clouds for visualization
    sensor_msgs::msg::PointCloud2 source_msg;
    pcl::toROSMsg(*source_cloud_, source_msg);
    source_msg.header.frame_id = "vgicp_base_link";
    source_msg.header.stamp = this->get_clock()->now();
    source_pub_->publish(source_msg);
    
    // sensor_msgs::msg::PointCloud2 target_msg;
    // pcl::toROSMsg(*target_cloud_, target_msg);
    // target_msg.header.frame_id = "map";
    // target_msg.header.stamp = this->get_clock()->now();
    // target_pub_->publish(target_msg);
    
    
    // RCLCPP_INFO(this->get_logger(), "Published aligned cloud with %zu points", aligned->size());
  }
  
  private:
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr source_sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr result_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr source_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr target_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
  rclcpp::Publisher<autoware_internal_debug_msgs::msg::Float32Stamped>::SharedPtr fitness_pub_;
  rclcpp::Publisher<autoware_internal_debug_msgs::msg::Float32Stamped>::SharedPtr exec_time_pub_;
  rclcpp::Publisher<autoware_internal_debug_msgs::msg::Int32Stamped>::SharedPtr iter_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pose_cov_pub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pose_sub_;
  std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  
  pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud_{new pcl::PointCloud<pcl::PointXYZ>()};
  pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud_{new pcl::PointCloud<pcl::PointXYZ>()};
  Eigen::Matrix4f initial_guess_;
  fast_gicp::FastVGICPCuda<pcl::PointXYZ, pcl::PointXYZ> vgicp;
  // fast_gicp::NDTCuda<pcl::PointXYZ, pcl::PointXYZ> vgicp;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<AlignNode>());
  rclcpp::shutdown();
  return 0;
}
