import cv2
import numpy as np

class BirdEyeView:
    def __init__(self, output_size: tuple = (400, 600)):
        """
        Initialize Bird's Eye View transformer
        output_size: (width, height) of the output top-down view
        """
        self.output_w, self.output_h = output_size
        self.transform_matrix = None
        print("Bird's eye view module loaded")

    def _compute_transform(self, frame_shape: tuple):
        """Compute perspective transform matrix based on frame size"""
        h, w = frame_shape[:2]

        # Source points — trapezoid on the road in camera view
        src = np.float32([
            [w * 0.1, h],           # bottom left
            [w * 0.9, h],           # bottom right
            [w * 0.55, h * 0.6],    # top right
            [w * 0.45, h * 0.6],    # top left
        ])

        # Destination points — rectangle in bird's eye view
        dst = np.float32([
            [self.output_w * 0.2, self.output_h],        # bottom left
            [self.output_w * 0.8, self.output_h],        # bottom right
            [self.output_w * 0.8, 0],                    # top right
            [self.output_w * 0.2, 0],                    # top left
        ])

        self.transform_matrix = cv2.getPerspectiveTransform(src, dst)

    def transform(self, frame: np.ndarray) -> np.ndarray:
        """Transform frame to bird's eye view perspective"""
        if self.transform_matrix is None:
            self._compute_transform(frame.shape)

        bev = cv2.warpPerspective(
            frame,
            self.transform_matrix,
            (self.output_w, self.output_h)
        )
        return bev

    def draw_objects_on_bev(
        self,
        bev: np.ndarray,
        detections: list,
        frame_shape: tuple
    ) -> np.ndarray:
        """
        Draw detected objects on the bird's eye view map
        Projects bounding box bottom centers to BEV coordinates
        """
        if self.transform_matrix is None:
            self._compute_transform(frame_shape)

        bev_copy = bev.copy()
        h, w = frame_shape[:2]

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]

            # Use bottom center of bounding box as ground position
            bottom_center = np.float32([[[
                (x1 + x2) / 2,
                y2
            ]]])

            # Project to BEV coordinates
            bev_point = cv2.perspectiveTransform(
                bottom_center,
                self.transform_matrix
            )

            bx, by = int(bev_point[0][0][0]), int(bev_point[0][0][1])

            # Draw object on BEV
            if 0 <= bx < self.output_w and 0 <= by < self.output_h:
                color = (0, 0, 255) if det["class_name"] == "person" else (0, 165, 255)
                cv2.circle(bev_copy, (bx, by), 8, color, -1)
                cv2.putText(
                    bev_copy,
                    det["class_name"][:3],
                    (bx + 5, by - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    color,
                    1
                )

        return bev_copy
