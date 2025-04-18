import os
import cv2
import numpy as np
import torch
import pandas as pd
import time
import argparse
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO
from typing import Dict, List, Tuple

class DwellTimeEstimator:
    def __init__(self, model_path="yolov11n.pt", conf_threshold=0.3, fps=30):
        """
        Initialize the dwell time estimator
        
        Args:
            model_path: Path to the YOLO model
            conf_threshold: Confidence threshold for detections
            fps: Frames per second of the input video (used for time calculation)
        """
        # Load the YOLO model
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.fps = fps
        
        # Dictionary to store tracked objects
        # Key: track_id, Value: [first_seen_frame, last_seen_frame, total_frames_appeared]
        self.tracked_objects = {}
        
        # Current frame counter
        self.frame_count = 0
        
        # Class labels for visualization (only interested in people - class 0)
        self.class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat']
    
    def process_frame(self, frame):
        """
        Process a single frame - detect people and update tracking information
        
        Args:
            frame: Input frame to process
            
        Returns:
            Annotated frame with tracking info and dwell times
        """
        self.frame_count += 1
        
        # Run YOLOv11 tracking on the frame
        results = self.model.track(frame, tracker="bytetrack.yaml", persist=True, conf=self.conf_threshold, classes=0)  # Only track people (class 0)
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            
            # Update tracking information
            for i, track_id in enumerate(track_ids):
                track_id = int(track_id)
                if track_id in self.tracked_objects:
                    self.tracked_objects[track_id][1] = self.frame_count  # Update last seen frame
                    self.tracked_objects[track_id][2] += 1  # Increment frame count
                else:
                    # First time seeing this object: [first_frame, last_frame, frame_count]
                    self.tracked_objects[track_id] = [self.frame_count, self.frame_count, 1]
            
            # Draw boxes and dwell time
            frame_with_boxes = self.draw_boxes_with_dwell_time(frame, boxes, track_ids, confidences)
            return frame_with_boxes
        
        return frame
    
    def draw_boxes_with_dwell_time(self, frame, boxes, track_ids, confidences):
        """
        Draw bounding boxes, track IDs, and dwell time on the frame
        
        Args:
            frame: Input frame
            boxes: Detected bounding boxes
            track_ids: Track IDs corresponding to boxes
            confidences: Detection confidences
            
        Returns:
            Annotated frame
        """
        annotated_frame = frame.copy()
        
        for i, (box, track_id, conf) in enumerate(zip(boxes, track_ids, confidences)):
            track_id = int(track_id)
            x1, y1, x2, y2 = map(int, box)
            
            # Calculate dwell time in seconds
            first_frame, last_frame, frame_count = self.tracked_objects[track_id]
            dwell_time = (last_frame - first_frame) / self.fps
            
            # Draw bounding box
            color = (0, 255, 0)  # Green
            thickness = 2
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
            
            # Add ID and dwell time text
            text = f"ID: {track_id}, Time: {dwell_time:.2f}s"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            text_thickness = 2
            text_size = cv2.getTextSize(text, font, font_scale, text_thickness)[0]
            
            # Draw text background
            cv2.rectangle(annotated_frame, (x1, y1 - text_size[1] - 10), (x1 + text_size[0], y1), color, -1)
            
            # Draw text
            cv2.putText(annotated_frame, text, (x1, y1 - 5), font, font_scale, (0, 0, 0), text_thickness)
        
        # Add overall statistics
        if self.tracked_objects:
            avg_dwell_time = self.calculate_average_dwell_time()
            stats_text = f"People: {len(self.tracked_objects)}, Avg Dwell: {avg_dwell_time:.2f}s"
            cv2.putText(annotated_frame, stats_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.8, (0, 0, 255), 2)
        
        return annotated_frame
    
    def calculate_average_dwell_time(self):
        """
        Calculate the average dwell time of all tracked objects
        
        Returns:
            Average dwell time in seconds
        """
        if not self.tracked_objects:
            return 0.0
        
        total_dwell_time = 0.0
        for track_id, (first_frame, last_frame, _) in self.tracked_objects.items():
            dwell_time = (last_frame - first_frame) / self.fps
            total_dwell_time += dwell_time
        
        return total_dwell_time / len(self.tracked_objects)
    
    def export_results(self, output_file="dwell_time_results.csv"):
        """
        Export dwell time results to a CSV file
        
        Args:
            output_file: Path to the output CSV file
        """
        results = []
        for track_id, (first_frame, last_frame, frame_count) in self.tracked_objects.items():
            dwell_time = (last_frame - first_frame) / self.fps
            results.append({
                'Track ID': track_id,
                'First Frame': first_frame,
                'Last Frame': last_frame,
                'Total Frames': frame_count,
                'Dwell Time (s)': dwell_time
            })
        
        if results:
            df = pd.DataFrame(results)
            df = df.sort_values(by='Dwell Time (s)', ascending=False)
            df.to_csv(output_file, index=False)
            
            # Also calculate and print average dwell time
            avg_dwell_time = df['Dwell Time (s)'].mean()
            print(f"Average dwell time: {avg_dwell_time:.2f} seconds")
            
            # Create a summary file
            with open("dwell_time_summary.txt", "w") as f:
                f.write(f"Total individuals tracked: {len(results)}\n")
                f.write(f"Average dwell time: {avg_dwell_time:.2f} seconds\n")
                f.write(f"Maximum dwell time: {df['Dwell Time (s)'].max():.2f} seconds\n")
                f.write(f"Minimum dwell time: {df['Dwell Time (s)'].min():.2f} seconds\n")


def process_video(input_path, output_path, model_path="yolov11n.pt", conf_threshold=0.3):
    """
    Process a video file to estimate dwell time
    
    Args:
        input_path: Path to input video
        output_path: Path to output video
        model_path: Path to YOLO model
        conf_threshold: Confidence threshold for detections
    """
    # Open video file
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_path}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Initialize dwell time estimator
    estimator = DwellTimeEstimator(model_path=model_path, conf_threshold=conf_threshold, fps=fps)
    
    # Process frames
    frame_idx = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process the frame
        frame_with_tracking = estimator.process_frame(frame)
        writer.write(frame_with_tracking)
        
        # Print progress
        frame_idx += 1
        if frame_idx % 10 == 0:
            elapsed = time.time() - start_time
            fps_processing = frame_idx / elapsed
            progress = frame_idx / frame_count * 100
            print(f"Processing: {progress:.1f}% complete ({frame_idx}/{frame_count}), "
                  f"FPS: {fps_processing:.2f}")
    
    # Release resources
    cap.release()
    writer.release()
    
    # Export results
    estimator.export_results()
    print(f"Processing complete. Output saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Dwell Time Estimation in Retail Environment')
    parser.add_argument('--input', type=str, required=True, help='Path to input video file')
    parser.add_argument('--output', type=str, default='output.mp4', help='Path to output video file')
    parser.add_argument('--model', type=str, default='yolo11n.pt', help='Path to YOLO model')
    parser.add_argument('--conf', type=float, default=0.3, help='Confidence threshold')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Process the video
    process_video(args.input, args.output, args.model, args.conf)


if __name__ == "__main__":
    main()