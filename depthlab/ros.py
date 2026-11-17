"""A common profiled inference node and timestamp-matched request client."""
import argparse
import json
from pathlib import Path
import sys
import time
from uuid import uuid4
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from .paths import ROOT
from .measurement import GpuMeasurement, PROFILE_TOPIC
from .outputs import save_first_result, write_json

IMAGE_TOPIC = "/kitti/camera_left/image_rect"
DEPTH_TOPIC = "/depth/metric"
OPTICAL_FRAME = "camera_left_optical_frame"

def decode_rgb(message):
    if message.encoding != "rgb8" or message.height <= 0 or message.width <= 0:
        raise ValueError("Expected a positive-sized rgb8 image")
    if message.step < message.width * 3:
        raise ValueError("Invalid RGB stride")
    raw = np.frombuffer(message.data, dtype=np.uint8)
    if raw.size != message.height * message.step:
        raise ValueError("Invalid RGB byte count")
    return raw.reshape(message.height, message.step)[:, :message.width * 3].reshape(message.height, message.width, 3).copy()

def decode_depth(message):
    h, w = message.height, message.width
    if message.encoding != "32FC1" or h <= 0 or w <= 0 or message.step < w * 4:
        raise ValueError("Invalid 32FC1 depth message")
    if len(message.data) != h * message.step:
        raise ValueError("Invalid depth byte count")
    dtype = np.dtype(">f4" if message.is_bigendian else "<f4")
    return np.ndarray((h, w), dtype=dtype, buffer=bytes(message.data), strides=(message.step, 4)).astype(np.float32, copy=True)

def encode_rgb(rgb, stamp):
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("Expected RGB uint8 HxWx3")
    result = Image()
    result.header.frame_id = OPTICAL_FRAME
    result.header.stamp = stamp
    result.height, result.width = rgb.shape[:2]
    result.encoding = "rgb8"
    result.is_bigendian = 0
    result.step = result.width * 3
    result.data = rgb.tobytes(order="C")
    return result

def encode_depth(depth, header):
    result = Image()
    result.header = header
    result.height, result.width = depth.shape
    result.encoding = "32FC1"
    result.is_bigendian = 0
    result.step = result.width * 4
    result.data = depth.astype("<f4", copy=False).tobytes(order="C")
    return result

class DepthNode(Node):
    def __init__(self, output_dir, model_size="small", profile=True, save_first=False):
        from .models import MetricDepthEstimator
        super().__init__("metric_depth_node")
        self.get_logger().info(f"Loading {model_size}")
        self.estimator = MetricDepthEstimator(model_size)
        self.measurement = GpuMeasurement(self.estimator) if profile else None
        self.run_info = {"node_session_id": uuid4().hex, **self.estimator.metadata(),
                         "profiling_enabled": profile, "measurement_schema": 2}
        self.output_dir = Path(output_dir)
        write_json(self.output_dir / "run_info.json", self.run_info)
        self.failure = None
        self.processed = 0
        self.save_first = save_first
        self.publisher = self.create_publisher(Image, DEPTH_TOPIC, 1)
        self.profile_publisher = self.create_publisher(String, PROFILE_TOPIC, 1) if profile else None
        self.subscription = self.create_subscription(Image, IMAGE_TOPIC, self.infer, 1)
        self.get_logger().info(f"Ready: {IMAGE_TOPIC} -> {DEPTH_TOPIC}; profiling={profile}")

    def infer(self, message):
        try:
            if message.header.frame_id != OPTICAL_FRAME:
                raise ValueError("Unexpected optical frame")
            rgb = decode_rgb(message)
            if self.measurement:
                depth, metrics = self.measurement.predict(rgb)
            else:
                depth, metrics = self.estimator.predict(rgb), None
            self.publisher.publish(encode_depth(depth, message.header))
            if metrics is not None:
                report = String()
                report.data = json.dumps({**metrics, "stamp": [message.header.stamp.sec, message.header.stamp.nanosec],
                                          "network_input_shape": self.estimator.input_shape, "model": self.run_info}, allow_nan=False)
                self.profile_publisher.publish(report)
            self.processed += 1
            if self.save_first and self.processed == 1:
                save_first_result(rgb, depth, message.header, self.estimator.input_shape, self.output_dir, self.run_info)
        except (OSError, RuntimeError, ValueError) as error:
            self.failure = str(error)
            self.get_logger().error(self.failure)

class DepthClient(Node):
    def __init__(self):
        super().__init__("depthlab_client")
        self.pending_stamp = None
        self.received = None
        self.profile = None
        self.depth_received_ns = None
        self.used_stamps = set()
        self.publisher = self.create_publisher(Image, IMAGE_TOPIC, 1)
        self.subscription = self.create_subscription(Image, DEPTH_TOPIC, self.receive, 1)
        self.profile_subscription = self.create_subscription(String, PROFILE_TOPIC, self.receive_profile, 1)

    def receive(self, message):
        stamp = (message.header.stamp.sec, message.header.stamp.nanosec)
        if self.pending_stamp is None or stamp != self.pending_stamp or self.received is not None:
            return
        if message.header.frame_id != OPTICAL_FRAME:
            raise ValueError("Unexpected optical frame")
        self.received = decode_depth(message)
        self.depth_received_ns = time.perf_counter_ns()

    def receive_profile(self, message):
        result = json.loads(message.data)
        if self.pending_stamp is not None and tuple(result["stamp"]) == self.pending_stamp:
            self.profile = result

    def wait_ready(self, timeout=60, profile_topic=None):
        deadline = time.monotonic() + timeout
        while rclpy.ok():
            counts = [self.count_publishers(IMAGE_TOPIC), self.count_publishers(DEPTH_TOPIC)]
            if profile_topic:
                counts.append(self.count_publishers(profile_topic))
            if any(c > 1 for c in counts):
                raise RuntimeError("Stop other publishers and depth nodes")
            if self.publisher.get_subscription_count() and all(c == 1 for c in counts):
                return
            if time.monotonic() >= deadline:
                raise RuntimeError("Start the matching depth node in another terminal")
            rclpy.spin_once(self, timeout_sec=.1)
        raise RuntimeError("ROS stopped")

    def request(self, rgb, timeout=60, profile=True):
        if self.count_publishers(IMAGE_TOPIC) != 1 or self.count_publishers(DEPTH_TOPIC) != 1:
            raise RuntimeError("Expected one image source and one depth node")
        if profile and self.count_publishers(PROFILE_TOPIC) != 1:
            raise RuntimeError("Node is not publishing profiling data")
        started = time.perf_counter_ns()
        message = encode_rgb(rgb, self.get_clock().now().to_msg())
        stamp = (message.header.stamp.sec, message.header.stamp.nanosec)
        if stamp in self.used_stamps:
            raise RuntimeError("Repeated ROS timestamp; check the replay clock")
        self.used_stamps.add(stamp)
        self.pending_stamp = stamp
        self.received = self.profile = self.depth_received_ns = None
        self.publisher.publish(message)
        while rclpy.ok() and (self.received is None or (profile and self.profile is None)):
            if (time.perf_counter_ns() - started) / 1e9 >= timeout:
                raise RuntimeError("Timed out waiting for matching depth/telemetry")
            rclpy.spin_once(self, timeout_sec=.1)
        if self.received is None or (profile and self.profile is None):
            raise RuntimeError("ROS stopped before the response completed")
        if self.received.shape != rgb.shape[:2] or not np.isfinite(self.received).all() or np.any(self.received <= 0):
            raise ValueError("Invalid received depth")
        if profile:
            result = dict(self.profile)
            result["ros_round_trip_ms"] = (self.depth_received_ns - started) / 1e6
            result["request_complete_ms"] = (time.perf_counter_ns() - started) / 1e6
            for name in ("forward_gpu_ms", "predict_wall_ms", "ros_round_trip_ms", "request_complete_ms"):
                if not np.isfinite(result[name]) or result[name] <= 0:
                    raise ValueError(f"Invalid timing: {name}")
        else:
            result = None
        return self.received, stamp, result

    def infer(self, rgb, timeout):
        depth, stamp, _ = self.request(rgb, timeout, profile=False)
        return depth, stamp

    def measure(self, rgb, timeout):
        _, _, result = self.request(rgb, timeout, profile=True)
        return result

def node_main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-size", choices=("small", "large"), default="small")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-profile", action="store_true")
    parser.add_argument("--save-first", action="store_true")
    args, ros_args = parser.parse_known_args(argv)
    node = None
    rclpy.init(args=[sys.argv[0], *ros_args])
    try:
        node = DepthNode(args.output or ROOT / "runs/nodes" / args.model_size, args.model_size, not args.no_profile, args.save_first)
        while rclpy.ok() and node.failure is None:
            rclpy.spin_once(node, timeout_sec=.5)
        if node.failure:
            raise RuntimeError(node.failure)
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
