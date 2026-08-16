import cv2
import numpy as np
from pathlib import Path

class VideoProcessor:
    def __init__(self, input_path: str, output_path: str = None):
        """
        Initialize video processor
        input_path: path to input video file
        output_path: path to save output video (optional)
        """
        self.input_path = input_path
        self.output_path = output_path
        self.cap = None
        self.writer = None

    def open(self) -> bool:
        """Open the input video file"""
        self.cap = cv2.VideoCapture(self.input_path)
        if not self.cap.isOpened():
            print(f"Error: Could not open video: {self.input_path}")
            return False

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Video opened: {Path(self.input_path).name}")
        print(f"Resolution: {self.width}x{self.height} @ {self.fps:.1f} FPS")
        print(f"Total frames: {self.total_frames}")

        return True

    def setup_writer(self, output_width: int, output_height: int):
        """Setup video writer for saving output"""
        if self.output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(
                self.output_path,
                fourcc,
                self.fps,
                (output_width, output_height)
            )
            print(f"Output will be saved to: {self.output_path}")

    def read_frame(self):
        """Read next frame from video"""
        if self.cap is None:
            return False, None
        ret, frame = self.cap.read()
        return ret, frame

    def write_frame(self, frame: np.ndarray):
        """Write frame to output video"""
        if self.writer is not None:
            self.writer.write(frame)

    def release(self):
        """Release all video resources"""
        if self.cap:
            self.cap.release()
        if self.writer:
            self.writer.release()
        cv2.destroyAllWindows()
        print("Video resources released")

    def get_info(self) -> dict:
        """Get video metadata"""
        return {
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "total_frames": self.total_frames
        }
