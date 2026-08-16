"""
Smoke tests — verify all modules load and initialize without errors
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import cv2


def test_config_imports():
    """Config module loads and has required keys"""
    from config import (
        MODEL_SIZE, CONFIDENCE_THRESHOLD, DEPTH_SKIP_FRAMES,
        LANE_SLOPE_THRESHOLD, BEV_OUTPUT_SIZE, VIDEO_INPUT_PATH
    )
    assert MODEL_SIZE is not None
    assert 0 < CONFIDENCE_THRESHOLD < 1
    assert DEPTH_SKIP_FRAMES > 0


def test_lane_detector_loads():
    """Lane detector initializes without errors"""
    from modules.lane_detection import LaneDetector
    detector = LaneDetector()
    assert detector is not None


def test_object_detector_loads():
    """Object detector initializes and loads YOLOv8 model"""
    from modules.object_detection import ObjectDetector
    detector = ObjectDetector()
    assert detector is not None
    assert detector.model is not None


def test_bird_eye_view_loads():
    """Bird's eye view module initializes without errors"""
    from modules.bird_eye_view import BirdEyeView
    bev = BirdEyeView()
    assert bev is not None


def test_video_processor_loads():
    """Video processor initializes without errors"""
    from utils.video import VideoProcessor
    vp = VideoProcessor("data/traffic.mp4")
    assert vp is not None


def test_drawing_utils_import():
    """Drawing utilities import without errors"""
    from utils.drawing import (
        draw_detections, draw_lanes, draw_fps,
        draw_info_panel, create_side_by_side
    )
    assert draw_detections is not None
    assert draw_lanes is not None
