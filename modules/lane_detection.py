import cv2
import numpy as np
from collections import deque

from config import LANE_SLOPE_THRESHOLD, ROI_TOP_WIDTH_RATIO

HISTORY_LENGTH = 5

class LaneDetector:
    def __init__(self):
        """Initialize lane detector with default parameters"""
        self.left_history = deque(maxlen=HISTORY_LENGTH)
        self.right_history = deque(maxlen=HISTORY_LENGTH)
        print("Lane detector loaded")

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Convert frame to grayscale, blur and edge detect"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        return edges

    def _region_of_interest(self, edges: np.ndarray) -> np.ndarray:
        """Mask only the road region in front of the car"""
        height, width = edges.shape
        mask = np.zeros_like(edges)

        # Narrow trapezoid hugging the ego lane, instead of a wide triangle
        # that sweeps in the opposite lane / shoulders / hood
        polygon = np.array([[
            (int(width * 0.15), height),
            (int(width * 0.85), height),
            (int(width * 0.58), int(height * ROI_TOP_WIDTH_RATIO)),
            (int(width * 0.42), int(height * ROI_TOP_WIDTH_RATIO)),
        ]], np.int32)

        cv2.fillPoly(mask, polygon, 255)
        masked = cv2.bitwise_and(edges, mask)
        return masked

    def _detect_lines(self, masked_edges: np.ndarray) -> list:
        """Use Hough Transform to detect lane lines"""
        lines = cv2.HoughLinesP(
            masked_edges,
            rho=1,
            theta=np.pi / 180,
            threshold=40,
            minLineLength=50,
            maxLineGap=150
        )
        return lines if lines is not None else []

    def _average_lines(self, frame: np.ndarray, lines: list):
        """Separate and average left and right lane lines"""
        left_lines = []
        right_lines = []
        height, width = frame.shape[:2]

        for line in lines:
            line = line.flatten()
            x1, y1, x2, y2 = line[0], line[1], line[2], line[3]
            if x1 == x2:
                continue  # skip vertical lines

            slope = (y2 - y1) / (x2 - x1)

            # Filter by slope — removes horizontal noise
            if abs(slope) < LANE_SLOPE_THRESHOLD:
                continue

            mid_x = (x1 + x2) / 2
            frame_mid_x = width / 2

            # Require slope sign AND position to agree — a negative-slope
            # line sitting on the right side of the frame is not a left lane
            if slope < 0 and mid_x < frame_mid_x:
                left_lines.append(line)
            elif slope > 0 and mid_x >= frame_mid_x:
                right_lines.append(line)

        left_avg = self._fit_line(frame, left_lines)
        right_avg = self._fit_line(frame, right_lines)

        return left_avg, right_avg

    def _fit_line(self, frame: np.ndarray, lines: list):
        """Fit a single averaged line from multiple detected lines"""
        if not lines:
            return None

        height, width = frame.shape[:2]
        x_coords = []
        y_coords = []

        for x1, y1, x2, y2 in lines:
            x_coords += [x1, x2]
            y_coords += [y1, y2]

        try:
            poly = np.polyfit(y_coords, x_coords, 1)
        except Exception:
            return None

        y1 = height
        y2 = int(height * 0.6)
        x1 = int(np.polyval(poly, y1))
        x2 = int(np.polyval(poly, y2))

        return [(x1, y1), (x2, y2)]

    def _smooth_lane(self, history: deque, new_lane):
        """Average the last HISTORY_LENGTH detections to reduce jitter"""
        if new_lane is not None:
            history.append(new_lane)

        if not history:
            return None

        x1_avg = int(np.mean([lane[0][0] for lane in history]))
        y1_avg = int(np.mean([lane[0][1] for lane in history]))
        x2_avg = int(np.mean([lane[1][0] for lane in history]))
        y2_avg = int(np.mean([lane[1][1] for lane in history]))

        return [(x1_avg, y1_avg), (x2_avg, y2_avg)]

    def detect(self, frame: np.ndarray) -> dict:
        """
        Run full lane detection pipeline on a frame
        Returns dict with left_lane, right_lane coordinates and lane overlay
        """
        edges = self._preprocess(frame)
        masked = self._region_of_interest(edges)
        lines = self._detect_lines(masked)
        left_lane, right_lane = self._average_lines(frame, lines)

        left_lane = self._smooth_lane(self.left_history, left_lane)
        right_lane = self._smooth_lane(self.right_history, right_lane)

        return {
            "left_lane": left_lane,
            "right_lane": right_lane,
            "edges": masked
        }
