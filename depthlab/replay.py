import argparse
import math
from pathlib import Path
import sys
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Image
from .paths import ROOT, KITTI_DIR as DATASET_DIR
from .data.kitti import load_frame
from .ros import IMAGE_TOPIC, OPTICAL_FRAME, encode_rgb
class KittiImagePublisher(Node):

    def __init__(self, frame, rate):
        super().__init__('kitti_image_publisher')
        self.frame_id = frame.frame_id
        self.publisher = self.create_publisher(Image, IMAGE_TOPIC, 1)
        height, width, _ = frame.image.shape
        self.message = encode_rgb(frame.image, self.get_clock().now().to_msg())
        self.published = 0
        self.timer = self.create_timer(1.0 / rate, self.publish_image)
        self.get_logger().info(f'Replaying frame {self.frame_id}: {width}x{height}, rgb8, {rate:g} Hz on {IMAGE_TOPIC}')
        self.get_logger().info('Timestamps represent replay time, not original KITTI acquisition time.')

    def publish_image(self):
        self.message.header.stamp = self.get_clock().now().to_msg()
        self.publisher.publish(self.message)
        self.published += 1
        self.get_logger().info(f'Published {self.published}: KITTI frame {self.frame_id}')

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frame', type=int, default=0)
    parser.add_argument('--rate', type=float, default=1.0)
    parser.add_argument('--dataset', type=Path, default=DATASET_DIR)
    args, ros_args = parser.parse_known_args(argv)
    if not math.isfinite(args.rate) or args.rate <= 0:
        parser.error('--rate must be a finite positive number')
    try:
        frame = load_frame(args.frame, args.dataset)
    except (OSError, ValueError) as error:
        parser.exit(1, f'Input check failed: {error}\n')
    node = None
    rclpy.init(args=[sys.argv[0], *ros_args])
    try:
        node = KittiImagePublisher(frame, args.rate)
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
