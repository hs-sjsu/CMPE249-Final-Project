import argparse
import math
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image as PILImage
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Image
from .paths import ROOT, KITTI_DIR as DATASET_DIR
from .data.kitti import load_frame
from .ros import IMAGE_TOPIC, OPTICAL_FRAME, decode_rgb
OUTPUT_DIR = ROOT / "runs/ros_check"
class KittiImageSubscriber(Node):

    def __init__(self, expected):
        super().__init__('kitti_image_subscriber')
        self.expected = expected
        self.done = False
        self.error = None
        self.subscription = self.create_subscription(Image, IMAGE_TOPIC, self.check_image, 1)
        self.get_logger().info(f'Waiting for frame {expected.frame_id} on {IMAGE_TOPIC}')

    def check_image(self, message):
        if self.done:
            return
        try:
            if message.header.frame_id != OPTICAL_FRAME:
                raise ValueError(f'Unexpected optical frame: {message.header.frame_id}')
            received = decode_rgb(message)
            if received.shape != self.expected.image.shape:
                raise ValueError(f'Received {received.shape}; expected {self.expected.image.shape}')
            if not np.array_equal(received, self.expected.image):
                raise ValueError('Pixels differ. Check that publisher and subscriber use the same --frame.')
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            path = OUTPUT_DIR / f'{self.expected.frame_id}_received.png'
            PILImage.fromarray(received).save(path)
            stamp = message.header.stamp
            self.get_logger().info(f'Received {message.width}x{message.height} RGB8; timestamp={stamp.sec}.{stamp.nanosec:09d}')
            self.get_logger().info('PASS: received RGB matches the original pixel-for-pixel.')
            self.get_logger().info(f'Saved: {path}')
        except (ValueError, OSError) as error:
            self.error = str(error)
            self.get_logger().error(self.error)
        finally:
            self.done = True

def transport(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frame', type=int, default=0)
    parser.add_argument('--dataset', type=Path, default=DATASET_DIR)
    parser.add_argument('--timeout', type=float, default=15.0)
    args, ros_args = parser.parse_known_args(argv)
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error('--timeout must be a finite positive number')
    try:
        expected = load_frame(args.frame, args.dataset)
    except (OSError, ValueError) as error:
        parser.exit(1, f'Input check failed: {error}\n')
    node = None
    failure = None
    rclpy.init(args=[sys.argv[0], *ros_args])
    try:
        node = KittiImageSubscriber(expected)
        deadline = time.monotonic() + args.timeout
        while not node.done:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('No image received. Keep the publisher running in another terminal.')
            rclpy.spin_once(node, timeout_sec=min(remaining, 1.0))
        failure = node.error
    except (KeyboardInterrupt, ExternalShutdownException):
        failure = 'Check interrupted'
    except TimeoutError as error:
        failure = str(error)
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    if failure:
        parser.exit(1, f'Check failed: {failure}\n')
