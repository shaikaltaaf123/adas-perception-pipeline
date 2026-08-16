import torch
import cv2
import numpy as np

class DepthEstimator:
    def __init__(self):
        """Initialize MiDaS depth estimation model"""
        print("Loading MiDaS depth model...")

        # Load MiDaS small model — fastest, works on CPU
        self.model = torch.hub.load(
            "intel-isl/MiDaS",
            "MiDaS_small",
            trust_repo=True
        )

        self.model.eval()

        # Load transforms
        midas_transforms = torch.hub.load(
            "intel-isl/MiDaS",
            "transforms",
            trust_repo=True
        )
        self.transform = midas_transforms.small_transform

        # Use GPU if available, otherwise CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        print(f"Depth estimator loaded on: {self.device}")

    def estimate(self, frame: np.ndarray) -> np.ndarray:
        """
        Estimate depth map for a frame
        Returns normalized depth map (0=far, 1=close)
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Apply MiDaS transforms
        input_batch = self.transform(rgb_frame).to(self.device)

        with torch.no_grad():
            prediction = self.model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_map = prediction.cpu().numpy()

        # Normalize to 0-1
        depth_min = depth_map.min()
        depth_max = depth_map.max()

        if depth_max - depth_min > 0:
            depth_normalized = (depth_map - depth_min) / (depth_max - depth_min)
        else:
            depth_normalized = depth_map

        return depth_normalized

    def get_object_depth(self, depth_map: np.ndarray, bbox: list) -> float:
        """
        Get average depth value for a detected object's bounding box
        Returns depth value 0-1 (higher = closer)
        """
        x1, y1, x2, y2 = bbox

        # Clamp to frame boundaries
        h, w = depth_map.shape
        x1 = max(0, min(x1, w-1))
        x2 = max(0, min(x2, w-1))
        y1 = max(0, min(y1, h-1))
        y2 = max(0, min(y2, h-1))

        region = depth_map[y1:y2, x1:x2]

        if region.size == 0:
            return 0.0

        return float(np.mean(region))

    def depth_to_colormap(self, depth_map: np.ndarray) -> np.ndarray:
        """Convert depth map to colored visualization"""
        depth_uint8 = (depth_map * 255).astype(np.uint8)
        colormap = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_MAGMA)
        return colormap
