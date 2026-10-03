import cv2
import torch
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np
from typing import Callable
from logging_config import configure_logging, get_logger

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
NUM_GRID_ROWS = 6
NUM_GRID_COLS = 6

configure_logging()
logger = get_logger(__name__)


class YOLOInference:
    def __init__(self, model_path="yolov8n.pt"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(model_path).to(self.device)
        logger.info("model_loaded", device=self.device, model=model_path)
        self.tracker = DeepSort(max_age=30, embedder="mobilenet")

        self.last_alert_time = 0
        self.fgbg = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=50, detectShadows=True
        )
        self.density_map = None
        self.latest_frame = None

        self.frame_indices = []
        self.crowd_counts = []

        self.enable_heat_map = False
        self.zoom_row = None
        self.zoom_col = None
        self.last_processed_overlay = None

        # ---------------- RISK FEATURE ----------------
        self.risk_score = 0
        self.prev_person_count = 0
        self.current_count = 0
        # --------------------------------------------------

    def set_heatmap_enabled(self, state: bool):
        self.enable_heat_map = state
        logger.info("heatmap_setting_changed", enabled=self.enable_heat_map)

    def set_zoom_cell(self, row: int, col: int):
        if row < 0 or col < 0:
            self.zoom_row = None
            self.zoom_col = None
        else:
            self.zoom_row = row
            self.zoom_col = col

    def get_zoomed_subimage(self):
        if self.latest_frame is None or self.zoom_row is None or self.zoom_col is None:
            return None

        height, width = self.latest_frame.shape[:2]
        cell_height = height // NUM_GRID_ROWS
        cell_width = width // NUM_GRID_COLS

        start_y = self.zoom_row * cell_height
        end_y = start_y + cell_height
        start_x = self.zoom_col * cell_width
        end_x = start_x + cell_width

        subimg = self.latest_frame[start_y:end_y, start_x:end_x].copy()
        subimg = cv2.resize(subimg, (width // 2, height // 2))
        return subimg

    def process_video(self, input_path, output_path, on_frame: Callable | None = None,
                      heatmap_enabled: Callable[[], bool] | None = None):
        # Keep the embedder warm but reset track identities for each uploaded video.
        self.tracker.delete_all_tracks()
        self.risk_score = 0
        self.prev_person_count = 0
        self.current_count = 0
        self.density_map = None
        self.latest_frame = None
        logger.info("video_opening", input_path=input_path)
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Unable to open input video: {input_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        if fps <= 0:
            fps = 25

        self.density_map = np.zeros((height, width), dtype=np.float32)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not out.isOpened():
            cap.release()
            raise ValueError(f"Unable to create output video: {output_path}")

        try:
            while True:
              ret, frame = cap.read()
              if not ret:
                  break
  
              results = self.model(frame, conf=0.3, device=self.device)
              detections = []
  
              # ---------------- DETECTION ----------------
              for result in results:
                  for box in result.boxes.data.cpu().numpy():
                      x1, y1, x2, y2, conf, class_id = box
                      if int(class_id) == 0:
                          w, h = x2 - x1, y2 - y1
                          detections.append(([x1, y1, w, h], conf, "person"))
  
              tracks = self.tracker.update_tracks(detections, frame=frame)
  
              # ---------------- RISK CALCULATION ----------------
              current_count = len(detections)   # 🔥 FIXED (reliable)
              self.current_count = current_count
  
              density_score = min(current_count * 2, 100)
              movement_score = abs(current_count - self.prev_person_count) * 5
  
              new_score = int(min(100, 0.7 * density_score + 0.3 * movement_score))
  
              # smoothing
              self.risk_score = int(0.8 * self.risk_score + 0.2 * new_score)
  
              self.prev_person_count = current_count
  
              logger.debug("risk_updated", count=current_count, risk_score=self.risk_score)
              # --------------------------------------------------
  
              # ---------------- DRAW BOXES ----------------
              for track in tracks:
                  if not track.is_confirmed():
                      continue
                  x1, y1, x2, y2 = map(int, track.to_tlbr())
                  cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
  
              # ---------------- DENSITY MAP ----------------
              for track in tracks:
                  if not track.is_confirmed():
                      continue
                  x1, y1, x2, y2 = map(int, track.to_tlbr())
                  cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                  cv2.circle(self.density_map, (cx, cy), 25, (1.0,), thickness=-1)
  
              # ---------------- HEATMAP ----------------
              if heatmap_enabled is not None:
                  self.enable_heat_map = heatmap_enabled()
              if self.enable_heat_map:
                  heatmap = cv2.applyColorMap(
                      cv2.normalize(self.density_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
                      cv2.COLORMAP_JET
                  )
                  frame = cv2.addWeighted(frame, 0.5, heatmap, 0.5, 0)
  
              out.write(frame)
              self.latest_frame = frame.copy()
              if on_frame is not None:
                  on_frame(frame, self.risk_score, self.current_count)
        finally:
            cap.release()
            out.release()
