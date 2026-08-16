"""Central configuration for the ADAS perception pipeline."""

# --- Object Detection ---
MODEL_SIZE = "yolov8s.pt"
CONFIDENCE_THRESHOLD = 0.5

# --- Depth Estimation ---
DEPTH_SKIP_FRAMES = 3  # run depth estimation every N frames

# --- Lane Detection ---
LANE_SLOPE_THRESHOLD = 0.4  # minimum |slope| to be considered a lane line
ROI_TOP_WIDTH_RATIO = 0.55  # vertical position (height ratio) of the ROI triangle apex

# --- Bird's Eye View ---
BEV_OUTPUT_SIZE = (400, 600)  # (width, height)
BEV_SRC_BOTTOM_LEFT_RATIO = 0.1
BEV_SRC_BOTTOM_RIGHT_RATIO = 0.9
BEV_SRC_TOP_LEFT_RATIO = 0.45
BEV_SRC_TOP_RIGHT_RATIO = 0.55
BEV_SRC_TOP_HEIGHT_RATIO = 0.6
BEV_DST_LEFT_RATIO = 0.2
BEV_DST_RIGHT_RATIO = 0.8

# --- Video I/O ---
VIDEO_INPUT_PATH = "data/traffic.mp4"
VIDEO_OUTPUT_PATH = "output/result.mp4"
