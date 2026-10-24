"""
Drone Target Tracker & Locker Module
------------------------------------
Implements a Kalman Filter-based object tracker and a target locking controller
with explicit target command filtering (e.g., locking onto a specific class).
"""

import cv2
import numpy as np


class KalmanBBoxTracker:
    """
    A 2D Kalman Filter for tracking bounding boxes [cx, cy, w, h] with a
    Constant Velocity motion model.
    """
    def __init__(self, bbox):
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = float(x2 - x1)
        h = float(y2 - y1)

        # State vector [cx, cy, w, h, vx, vy, vw, vh]
        self.state = np.array([cx, cy, w, h, 0, 0, 0, 0], dtype=np.float32)

        # State transition matrix (F) - Constant Velocity model
        self.F = np.eye(8, dtype=np.float32)
        for i in range(4):
            self.F[i, i + 4] = 1.0

        # Measurement matrix (H)
        self.H = np.zeros((4, 8), dtype=np.float32)
        for i in range(4):
            self.H[i, i] = 1.0

        # Covariance matrices
        self.P = np.eye(8, dtype=np.float32) * 100.0
        self.P[4:, 4:] *= 1000.0

        self.Q = np.eye(8, dtype=np.float32)
        self.Q[0:4, 0:4] *= 1.0
        self.Q[4:, 4:] *= 5.0

        self.R = np.eye(4, dtype=np.float32) * 10.0

        self.time_since_update = 0
        self.hits = 1
        self.hit_streak = 1
        self.age = 1

    def predict(self):
        self.state = np.dot(self.F, self.state)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        self.age += 1
        self.time_since_update += 1
        return self.get_bbox()

    def update(self, bbox):
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1

        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = float(x2 - x1)
        h = float(y2 - y1)
        z = np.array([cx, cy, w, h], dtype=np.float32)

        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        y = z - np.dot(self.H, self.state)
        self.state = self.state + np.dot(K, y)
        
        I = np.eye(self.P.shape[0], dtype=np.float32)
        self.P = np.dot(I - np.dot(K, self.H), self.P)

    def get_bbox(self):
        cx, cy, w, h = self.state[0:4]
        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0
        return np.array([x1, y1, x2, y2], dtype=np.float32)


class TargetLockController:
    """
    Manages target locking state machine based on user-commanded class filtering.
    """
    def __init__(self, target_class_name="person", iou_threshold=0.3, max_lost_frames=30):
        self.state = "SEARCHING"  # SEARCHING, LOCKED, LOST
        self.target_class_name = target_class_name.lower()
        self.locked_tracker = None
        self.locked_class = None
        self.locked_conf = 0.0
        self.iou_threshold = iou_threshold
        self.max_lost_frames = max_lost_frames
        self.lost_count = 0

    def select_target(self, boxes, names, frame_center):
        """
        Selects a target that strictly matches the commanded target class name.
        """
        if len(boxes) == 0:
            return None

        fx, fy = frame_center
        best_box = None
        min_dist = float("inf")

        for box in boxes:
            cls_id = int(box.cls[0])
            class_name = names.get(cls_id, "").lower()

            # STRICT FILTERING: Only consider boxes matching our commanded target class
            if class_name != self.target_class_name:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            dist = np.sqrt((cx - fx) ** 2 + (cy - fy) ** 2)

            # Pick the valid target closest to center
            if dist < min_dist:
                min_dist = dist
                best_box = box

        return best_box

    def update_lock(self, detections, names, frame_shape):
        """Updates lock status based on commanded target filtering and EKF prediction."""
        h, w, _ = frame_shape
        frame_center = (w / 2.0, h / 2.0)

        if self.state == "SEARCHING":
            best_box = self.select_target(detections, names, frame_center)
            if best_box is not None:
                x1, y1, x2, y2 = map(float, best_box.xyxy[0])
                self.locked_tracker = KalmanBBoxTracker([x1, y1, x2, y2])
                self.locked_class = int(best_box.cls[0])
                self.locked_conf = float(best_box.conf[0])
                self.state = "LOCKED"
                self.lost_count = 0

        elif self.state == "LOCKED":
            predicted_bbox = self.locked_tracker.predict()

            best_match = None
            max_iou = 0.0

            for box in detections:
                cls_id = int(box.cls[0])
                class_name = names.get(cls_id, "").lower()

                # Only match detections of our locked target class
                if class_name != self.target_class_name:
                    continue

                det_box = box.xyxy[0].cpu().numpy()
                iou = self._calculate_iou(predicted_bbox, det_box)
                if iou > max_iou:
                    max_iou = iou
                    best_match = box

            if best_match is not None and max_iou >= self.iou_threshold:
                x1, y1, x2, y2 = map(float, best_match.xyxy[0])
                self.locked_tracker.update([x1, y1, x2, y2])
                self.locked_conf = float(best_match.conf[0])
                self.lost_count = 0
            else:
                self.lost_count += 1
                if self.lost_count > self.max_lost_frames:
                    self.state = "LOST"
                    self.locked_tracker = None

        elif self.state == "LOST":
            self.locked_tracker = None
            self.state = "SEARCHING"

    def draw_lock_overlay(self, frame, names):
        h, w, _ = frame.shape
        fx, fy = w // 2, h // 2

        # Draw crosshairs (Yellow when searching, Red when locked)
        cross_color = (0, 255, 255) if self.state == "SEARCHING" else (0, 0, 255)
        cv2.line(frame, (fx - 15, fy), (fx + 15, fy), cross_color, 1)
        cv2.line(frame, (fx, fy - 15), (fx, fy + 15), cross_color, 1)

        if self.state == "LOCKED" and self.locked_tracker is not None:
            bbox = self.locked_tracker.get_bbox()
            x1, y1, x2, y2 = map(int, bbox)
            label = names.get(self.locked_class, str(self.locked_class))

            color = (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            self._draw_lock_corners(frame, x1, y1, x2, y2, color)

            text = f"LOCKED TARGET: {label} ({self.locked_conf:.2f})"
            cv2.putText(frame, text, (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

        # Status banner showing what object command we are looking for
        status_text = f"STATUS: {self.state} | TARGET CMD: {self.target_class_name.upper()}"
        status_color = (0, 255, 0) if self.state == "LOCKED" else (0, 165, 255)
        cv2.putText(frame, status_text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2, cv2.LINE_AA)

        return frame

    def _draw_lock_corners(self, img, x1, y1, x2, y2, color, length=15, thickness=2):
        cv2.line(img, (x1, y1), (x1 + length, y1), color, thickness)
        cv2.line(img, (x1, y1), (x1, y1 + length), color, thickness)
        cv2.line(img, (x2, y1), (x2 - length, y1), color, thickness)
        cv2.line(img, (x2, y1), (x2, y1 + length), color, thickness)
        cv2.line(img, (x1, y2), (x1 + length, y2), color, thickness)
        cv2.line(img, (x1, y2), (x1, y2 - length), color, thickness)
        cv2.line(img, (x2, y2), (x2 - length, y2), color, thickness)
        cv2.line(img, (x2, y2), (x2, y2 - length), color, thickness)

    def _calculate_iou(self, box1, box2):
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        xx1 = max(x1_1, x1_2)
        yy1 = max(y1_1, y1_2)
        xx2 = min(x2_1, x2_2)
        yy2 = min(y2_1, y2_2)

        w = max(0.0, xx2 - xx1)
        h = max(0.0, yy2 - yy1)
        wh = w * h

        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - wh

        if union <= 0.0:
            return 0.0
        return wh / union