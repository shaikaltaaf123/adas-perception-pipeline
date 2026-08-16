import cv2
import numpy as np

# Color scheme
COLORS = {
    "person":        (0, 0, 255),      # Red
    "car":           (0, 255, 0),      # Green
    "truck":         (0, 165, 255),    # Orange
    "bus":           (0, 165, 255),    # Orange
    "motorcycle":    (255, 0, 255),    # Magenta
    "bicycle":       (255, 255, 0),    # Cyan
    "traffic light": (0, 255, 255),    # Yellow
    "stop sign":     (0, 0, 200),      # Dark Red
    "default":       (255, 255, 255),  # White
}

COLLISION_CLASSES = {"car", "truck", "bus", "person"}

def draw_detections(frame: np.ndarray, detections: list, depth_map: np.ndarray = None, frame_count: int = 0) -> np.ndarray:
    """Draw bounding boxes and labels for detected objects"""
    frame_h = frame.shape[0]
    collision_warning = False

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        class_name = det["class_name"]
        confidence = det["confidence"]

        color = COLORS.get(class_name, COLORS["default"])

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Build label
        label = f"{class_name} {confidence:.0%}"

        # Add depth info if available
        if depth_map is not None:
            h, w = depth_map.shape
            cx = max(0, min((x1 + x2) // 2, w - 1))
            cy = max(0, min((y1 + y2) // 2, h - 1))
            depth_val = depth_map[cy, cx]

            if depth_val > 0.7:
                dist_label = "CLOSE"
                label_color = (0, 0, 255)

                # Very close, relevant-class object with its bottom edge in
                # the lower 40% of the frame is near and directly ahead — collision risk
                if (
                    depth_val > 0.85
                    and y2 > frame_h * 0.6
                    and class_name in COLLISION_CLASSES
                ):
                    collision_warning = True
            elif depth_val > 0.4:
                dist_label = "MED"
                label_color = (0, 165, 255)
            else:
                dist_label = "FAR"
                label_color = (0, 255, 0)

            label += f" [{dist_label}]"
        else:
            label_color = color

        # Draw label background
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - text_h - 8), (x1 + text_w + 4, y1), color, -1)

        # Draw label text
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    if collision_warning:
        warning_text = "COLLISION WARNING"
        (text_w, text_h), _ = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        frame_w = frame.shape[1]
        text_x = (frame_w - text_w) // 2
        text_y = text_h + 20
        flash_color = (0, 0, 255) if frame_count % 2 == 0 else (255, 255, 255)
        cv2.putText(frame, warning_text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, flash_color, 2)

    return frame


def draw_lanes(frame: np.ndarray, lane_result: dict) -> np.ndarray:
    """Draw detected lane lines on frame"""
    left_lane = lane_result.get("left_lane")
    right_lane = lane_result.get("right_lane")

    if left_lane:
        cv2.line(frame, left_lane[0], left_lane[1], (0, 255, 0), 4)

    if right_lane:
        cv2.line(frame, right_lane[0], right_lane[1], (0, 255, 0), 4)

    # Fill lane area if both lanes detected
    if left_lane and right_lane:
        pts = np.array([
            left_lane[0], left_lane[1],
            right_lane[1], right_lane[0]
        ], np.int32)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], (0, 255, 0))
        frame = cv2.addWeighted(frame, 0.8, overlay, 0.2, 0)

    return frame


def draw_fps(frame: np.ndarray, fps: float) -> np.ndarray:
    """Draw FPS counter on frame"""
    cv2.putText(frame, f"FPS: {fps:.1f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 255), 2)
    return frame


def draw_info_panel(frame: np.ndarray, detections: list) -> np.ndarray:
    """Draw detection count panel"""
    counts = {}
    for det in detections:
        name = det["class_name"]
        counts[name] = counts.get(name, 0) + 1

    y = 60
    for name, count in counts.items():
        color = COLORS.get(name, COLORS["default"])
        cv2.putText(frame, f"{name}: {count}",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, color, 2)
        y += 25

    return frame


def create_side_by_side(main_frame: np.ndarray, bev_frame: np.ndarray, target_height: int = 720) -> np.ndarray:
    """Combine main view and bird's eye view side by side"""
    # Resize main frame to target height
    main_h, main_w = main_frame.shape[:2]
    scale = target_height / main_h
    main_resized = cv2.resize(main_frame, (int(main_w * scale), target_height))

    # Resize BEV to same height
    bev_resized = cv2.resize(bev_frame, (int(bev_frame.shape[1] * target_height / bev_frame.shape[0]), target_height))

    # Add labels
    cv2.putText(main_resized, "PERCEPTION VIEW",
                (10, main_resized.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.putText(bev_resized, "BIRD'S EYE VIEW",
                (10, bev_resized.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Stack horizontally
    combined = np.hstack([main_resized, bev_resized])
    return combined
