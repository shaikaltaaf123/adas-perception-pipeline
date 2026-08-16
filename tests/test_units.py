"""
Unit tests — test individual functions with known inputs
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import cv2


# --- Lane Detection Tests ---

def test_fit_line_empty_input():
    """_fit_line returns None when given empty list"""
    from modules.lane_detection import LaneDetector
    detector = LaneDetector()
    result = detector._fit_line(np.zeros((720, 1280, 3), dtype=np.uint8), [])
    assert result is None


def test_fit_line_valid_input():
    """_fit_line returns two coordinate pairs for valid input"""
    from modules.lane_detection import LaneDetector
    detector = LaneDetector()
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    lines = [[100, 720, 200, 400], [120, 720, 220, 400]]
    result = detector._fit_line(frame, lines)
    assert result is not None
    assert len(result) == 2
    assert len(result[0]) == 2
    assert len(result[1]) == 2


def test_smooth_lane_empty_history():
    """_smooth_lane returns None when history is empty and no new lane"""
    from modules.lane_detection import LaneDetector
    from collections import deque
    detector = LaneDetector()
    history = deque(maxlen=5)
    result = detector._smooth_lane(history, None)
    assert result is None


def test_smooth_lane_averages_correctly():
    """_smooth_lane returns averaged coordinates from history"""
    from modules.lane_detection import LaneDetector
    from collections import deque
    detector = LaneDetector()
    history = deque(maxlen=5)
    lane1 = [(100, 720), (200, 400)]
    lane2 = [(200, 720), (300, 400)]
    history.append(lane1)
    history.append(lane2)
    result = detector._smooth_lane(history, None)
    assert result is not None
    assert result[0][0] == 150  # average of 100 and 200


# --- Object Detection Tests ---

def test_object_detector_relevant_classes():
    """Object detector has correct ADAS relevant classes"""
    from modules.object_detection import ObjectDetector
    detector = ObjectDetector()
    assert 0 in detector.relevant_classes  # person
    assert 2 in detector.relevant_classes  # car
    assert 7 in detector.relevant_classes  # truck


def test_object_detector_returns_list():
    """Object detector returns a list for a blank frame"""
    from modules.object_detection import ObjectDetector
    detector = ObjectDetector()
    blank_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    result = detector.detect(blank_frame)
    assert isinstance(result, list)


# --- Bird's Eye View Tests ---

def test_bev_transform_output_shape():
    """BEV transform returns correct output dimensions"""
    from modules.bird_eye_view import BirdEyeView
    bev = BirdEyeView(output_size=(400, 600))
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    result = bev.transform(frame)
    assert result.shape[1] == 400  # width
    assert result.shape[0] == 600  # height


def test_bev_transform_returns_numpy():
    """BEV transform returns a numpy array"""
    from modules.bird_eye_view import BirdEyeView
    bev = BirdEyeView()
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    result = bev.transform(frame)
    assert isinstance(result, np.ndarray)


# --- Drawing Utils Tests ---

def test_draw_fps_returns_frame():
    """draw_fps returns a numpy array"""
    from utils.drawing import draw_fps
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    result = draw_fps(frame, 30.0)
    assert isinstance(result, np.ndarray)


def test_draw_lanes_no_lanes():
    """draw_lanes handles None lanes without crashing"""
    from utils.drawing import draw_lanes
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    lane_result = {"left_lane": None, "right_lane": None}
    result = draw_lanes(frame, lane_result)
    assert isinstance(result, np.ndarray)


def test_draw_info_panel_empty_detections():
    """draw_info_panel handles empty detections without crashing"""
    from utils.drawing import draw_info_panel
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    result = draw_info_panel(frame, [])
    assert isinstance(result, np.ndarray)
