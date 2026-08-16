import cv2
import time
import argparse

from modules.object_detection import ObjectDetector
from modules.lane_detection import LaneDetector
from modules.depth_estimation import DepthEstimator
from modules.bird_eye_view import BirdEyeView
from utils.drawing import (
    draw_detections,
    draw_lanes,
    draw_fps,
    draw_info_panel,
    create_side_by_side
)
from utils.video import VideoProcessor

def run_pipeline(
    video_path: str,
    output_path: str = None,
    show_window: bool = True,
    skip_depth: bool = False
):
    print("ADAS Perception Pipeline")

    # Initialize all modules
    print("Loading object detector...")
    detector = ObjectDetector()

    print("Loading lane detector...")
    lane_detector = LaneDetector()

    print("Loading depth estimator...")
    depth_estimator = DepthEstimator() if not skip_depth else None

    print("Loading bird's eye view...")
    bev = BirdEyeView(output_size=(400, 600))

    print("\nAll modules loaded. Starting pipeline...\n")

    # Setup video
    video = VideoProcessor(video_path, output_path)
    if not video.open():
        return

    frame_count = 0
    fps_timer = time.time()
    fps = 0.0

    while True:
        ret, frame = video.read_frame()
        if not ret:
            print("\nVideo processing complete.")
            break

        frame_count += 1

        # Calculate FPS every 10 frames
        if frame_count % 10 == 0:
            elapsed = time.time() - fps_timer
            fps = 10 / elapsed
            fps_timer = time.time()

        # Step 1 — Object Detection (crop out bottom 15% to exclude the hood)
        detection_frame = frame[:int(frame.shape[0] * 0.85), :]
        detections = detector.detect(detection_frame)

        # Step 2 — Lane Detection
        lane_result = lane_detector.detect(frame)

        # Step 3 — Depth Estimation (every 3 frames for speed)
        depth_map = None
        if depth_estimator and frame_count % 3 == 0:
            depth_map = depth_estimator.estimate(frame)

        # Step 4 — Draw everything on frame
        output_frame = frame.copy()
        output_frame = draw_lanes(output_frame, lane_result)
        output_frame = draw_detections(output_frame, detections, depth_map, frame_count)
        output_frame = draw_fps(output_frame, fps)
        output_frame = draw_info_panel(output_frame, detections)

        # Step 5 — Bird's Eye View
        bev_frame = bev.transform(frame)
        bev_frame = bev.draw_objects_on_bev(bev_frame, detections, frame.shape)

        # Step 6 — Combine views side by side
        combined = create_side_by_side(output_frame, bev_frame)

        # Setup writer on first frame
        if frame_count == 1 and output_path:
            h, w = combined.shape[:2]
            video.setup_writer(w, h)

        # Write frame to output
        video.write_frame(combined)

        # Show window
        if show_window:
            cv2.imshow("ADAS Perception Pipeline", combined)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nStopped by user.")
                break

        # Progress update every 30 frames
        if frame_count % 30 == 0:
            print(f"Processed {frame_count}/{video.total_frames} frames | FPS: {fps:.1f}")

    video.release()
    print(f"\nTotal frames processed: {frame_count}")
    if output_path:
        print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ADAS Perception Pipeline")
    parser.add_argument("--video", type=str, default="data/traffic.mp4", help="Path to input video")
    parser.add_argument("--output", type=str, default="output/result.mp4", help="Path to save output")
    parser.add_argument("--no-window", action="store_true", help="Run without display window")
    parser.add_argument("--skip-depth", action="store_true", help="Skip depth estimation for faster processing")
    args = parser.parse_args()

    run_pipeline(
        video_path=args.video,
        output_path=args.output,
        show_window=not args.no_window,
        skip_depth=args.skip_depth
    )
