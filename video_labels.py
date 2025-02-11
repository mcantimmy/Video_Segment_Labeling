import numpy as np
import cv2
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cosine
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
from typing import List, Tuple, Dict

class VideoSegmenter:
    def __init__(self, segment_length: int = 30):
        self.segment_length = segment_length
        self.feature_extractor = self._build_feature_extractor()
    
    def _build_feature_extractor(self) -> nn.Module:
        # Using ResNet50 pretrained model for feature extraction
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
        return nn.Sequential(*list(model.children())[:-1])
    
    def extract_segments(self, video_path: str) -> Tuple[List[np.ndarray], List[int]]:
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frames = []
        timestamps = []
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % fps == 0:  # Sample one frame per second
                frame = cv2.resize(frame, (224, 224))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
                timestamps.append(frame_count // fps)
                
            frame_count += 1
            
        cap.release()
        return frames, timestamps
    
    def extract_features(self, frames: List[np.ndarray]) -> np.ndarray:
        features = []
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_extractor.to(device)
        
        for frame in frames:
            # Preprocess frame
            frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float()
            frame_tensor = frame_tensor.unsqueeze(0).to(device)
            
            # Extract features
            with torch.no_grad():
                feature = self.feature_extractor(frame_tensor)
                feature = feature.squeeze().cpu().numpy()
                features.append(feature)
                
        return np.array(features)

class IntroDetector:
    def __init__(self, eps: float = 0.3, min_samples: int = 3):
        self.clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
        
    def find_intros(self, features_dict: Dict[str, np.ndarray]) -> Dict[str, List[Tuple[int, int]]]:
        # Combine features from all videos
        all_features = []
        feature_map = []  # Keep track of which video and timestamp each feature belongs to
        
        for video_path, features in features_dict.items():
            all_features.extend(features)
            feature_map.extend([(video_path, i) for i in range(len(features))])
            
        # Cluster features
        labels = self.clustering.fit_predict(all_features)
        
        # Find segments that appear in multiple videos
        intro_segments = {}
        for label in set(labels):
            if label == -1:  # Skip noise
                continue
                
            # Get videos containing this cluster
            cluster_indices = np.where(labels == label)[0]
            videos_in_cluster = set(feature_map[i][0] for i in cluster_indices)
            
            if len(videos_in_cluster) > 1:  # Segment appears in multiple videos
                for video_path in videos_in_cluster:
                    video_indices = [i for i in cluster_indices if feature_map[i][0] == video_path]
                    timestamps = [feature_map[i][1] for i in video_indices]
                    
                    # Convert consecutive timestamps to segments
                    segments = []
                    start = timestamps[0]
                    prev = start
                    
                    for t in timestamps[1:]:
                        if t - prev > 1:  # Gap in sequence
                            segments.append((start, prev))
                            start = t
                        prev = t
                    segments.append((start, prev))
                    
                    if video_path not in intro_segments:
                        intro_segments[video_path] = []
                    intro_segments[video_path].extend(segments)
        
        return intro_segments

def main():
    # Initialize components
    segmenter = VideoSegmenter()
    detector = IntroDetector()
    
    # Process videos
    video_dir = "path/to/videos"
    features_dict = {}
    
    for video_file in os.listdir(video_dir):
        if not video_file.endswith(('.mp4', '.avi', '.mkv')):
            continue
            
        video_path = os.path.join(video_dir, video_file)
        frames, timestamps = segmenter.extract_segments(video_path)
        features = segmenter.extract_features(frames)
        features_dict[video_path] = features
    
    # Detect intro segments
    intro_segments = detector.find_intros(features_dict)
    
    # Output results
    for video_path, segments in intro_segments.items():
        print(f"\nIntro segments for {os.path.basename(video_path)}:")
        for start, end in segments:
            print(f"  {start}s - {end}s")

if __name__ == "__main__":
    main()