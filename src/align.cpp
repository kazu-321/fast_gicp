#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/io/pcd_io.h>
#include <rclcpp/qos.hpp>
#include <fast_gicp/gicp/fast_gicp.hpp>

#define TARGET_TOPIC "/localization/util/downsample/pointcloud"
#define PCD_FILE "/home/kazusahashimoto/autoware_map/sample-map-rosbag/pointcloud_map.pcd"


class AlignNode : public rclcpp::Node {
public:
  AlignNode() : Node("align_node") {
    rclcpp::QoS qos_settings(rclcpp::QoSInitialization::from_rmw(rmw_qos_profile_sensor_data));
    target_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        TARGET_TOPIC, qos_settings, std::bind(&AlignNode::targetCallback, this, std::placeholders::_1));
    result_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("aligned_cloud", 10);

    // Load source cloud from PCD file
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(PCD_FILE, *source_cloud_) == -1) {
      RCLCPP_ERROR(this->get_logger(), "Couldn't read source PCD file");
      throw std::runtime_error("Failed to load source PCD file");
    }
    RCLCPP_INFO(this->get_logger(), "Loaded source cloud with %zu points", source_cloud_->size());
  }

  void targetCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    pcl::fromROSMsg(*msg, *target_cloud_);
    RCLCPP_INFO(this->get_logger(), "Received target cloud with %zu points", target_cloud_->size());

    if (target_cloud_->empty() || source_cloud_->empty()) {
      RCLCPP_WARN(this->get_logger(), "Target or source cloud is empty, skipping alignment");
      return;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr aligned(new pcl::PointCloud<pcl::PointXYZ>());
    fast_gicp::FastGICP<pcl::PointXYZ, pcl::PointXYZ> gicp;
    gicp.setInputTarget(target_cloud_);
    gicp.setInputSource(source_cloud_);
    gicp.align(*aligned);

    sensor_msgs::msg::PointCloud2 output_msg;
    pcl::toROSMsg(*aligned, output_msg);
    output_msg.header.frame_id = "map";
    result_pub_->publish(output_msg);

    RCLCPP_INFO(this->get_logger(), "Published aligned cloud with %zu points", aligned->size());
  }

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr target_sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr result_pub_;

  pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud_{new pcl::PointCloud<pcl::PointXYZ>()};
  pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud_{new pcl::PointCloud<pcl::PointXYZ>()};
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<AlignNode>());
  rclcpp::shutdown();
  return 0;
}
