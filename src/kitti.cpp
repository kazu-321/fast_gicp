#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl_conversions/pcl_conversions.h>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/approximate_voxel_grid.h>

#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/point_cloud_color_handlers.h>

#include <fast_gicp/gicp/fast_gicp.hpp>

#ifdef USE_VGICP_CUDA
#include <fast_gicp/ndt/ndt_cuda.hpp>
#include <fast_gicp/gicp/fast_vgicp_cuda.hpp>
#endif

class KittiNode : public rclcpp::Node {
public:
  KittiNode() : Node("kitti_node") {
    source_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        "source_cloud", 10, std::bind(&KittiNode::sourceCallback, this, std::placeholders::_1));
    result_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("aligned_cloud", 10);
  }

private:
  void sourceCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    pcl::fromROSMsg(*msg, *source_cloud_);
    RCLCPP_INFO(this->get_logger(), "Received source cloud with %zu points", source_cloud_->size());

    if (target_cloud_->empty() || source_cloud_->empty()) {
      RCLCPP_WARN(this->get_logger(), "Target or source cloud is empty, skipping alignment");
      return;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr aligned(new pcl::PointCloud<pcl::PointXYZ>());
    gicp_.setInputTarget(target_cloud_);
    gicp_.setInputSource(source_cloud_);
    gicp_.align(*aligned);

    sensor_msgs::msg::PointCloud2 output_msg;
    pcl::toROSMsg(*aligned, output_msg);
    output_msg.header.frame_id = "map";
    result_pub_->publish(output_msg);

    RCLCPP_INFO(this->get_logger(), "Published aligned cloud with %zu points", aligned->size());
  }

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr source_sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr result_pub_;

  pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud_{new pcl::PointCloud<pcl::PointXYZ>()};
  pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud_{new pcl::PointCloud<pcl::PointXYZ>()};
  fast_gicp::FastGICP<pcl::PointXYZ, pcl::PointXYZ> gicp_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<KittiNode>());
  rclcpp::shutdown();
  return 0;
}