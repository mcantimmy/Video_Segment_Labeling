import numpy as np
import cv2
import hdbscan
from scipy.spatial.distance import cosine
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
from typing import List, Tuple, Dict, Optional

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
    def __init__(self, 
                 min_cluster_size: int = 3, 
                 min_samples: int = 2,
                 cluster_selection_epsilon: float = 0.5,
                 prediction_data: bool = True):
        """
        Initialize HDBSCAN-based intro detector
        
        Args:
            min_cluster_size: Minimum size of clusters
            min_samples: Number of samples in neighborhood for core point
            cluster_selection_epsilon: Distance threshold for cluster membership
            prediction_data: Whether to store prediction data for membership probabilities
        """
        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=cluster_selection_epsilon,
            metric='cosine',
            prediction_data=prediction_data
        )
        
    def find_intros(self, 
                    features_dict: Dict[str, np.ndarray],
                    min_videos: int = 2,
                    probability_threshold: float = 0.82
                    ) -> Dict[str, List[Tuple[int, int, float]]]:
        """
        Find intro segments across multiple videos
        
        Args:
            features_dict: Dictionary mapping video paths to feature arrays
            min_videos: Minimum number of videos a segment must appear in
            probability_threshold: Minimum probability for cluster membership
            
        Returns:
            Dictionary mapping video paths to lists of (start, end, probability) tuples
        """
        # Combine features from all videos
        all_features = []
        feature_map = []  # Keep track of which video and timestamp each feature belongs to
        
        for video_path, features in features_dict.items():
            all_features.extend(features)
            feature_map.extend([(video_path, i) for i in range(len(features))])
            
        # Convert to numpy array for clustering
        all_features = np.vstack(all_features)
        
        # Cluster features using HDBSCAN
        self.clusterer.fit(all_features)
        
        # Get cluster labels and probabilities
        labels = self.clusterer.labels_
        probabilities = self.clusterer.probabilities_
        
        # Find segments that appear in multiple videos
        intro_segments = {}
        
        for label in set(labels):
            if label == -1:  # Skip noise
                continue
                
            # Get indices and probabilities for this cluster
            cluster_indices = np.where(labels == label)[0]
            cluster_probs = probabilities[cluster_indices]
            
            # Get videos containing this cluster
            videos_in_cluster = set(feature_map[i][0] for i in cluster_indices)
            
            if len(videos_in_cluster) >= min_videos:  # Segment appears in multiple videos
                for video_path in videos_in_cluster:
                    # Get indices and probabilities for this video
                    video_indices = [i for i, fm in enumerate(cluster_indices) 
                                   if feature_map[fm][0] == video_path]
                    video_probs = cluster_probs[video_indices]
                    timestamps = [feature_map[cluster_indices[i]][1] for i in video_indices]
                    
                    # Convert consecutive timestamps to segments with probabilities
                    segments = []
                    if timestamps:
                        start = timestamps[0]
                        start_prob = video_probs[0]
                        prev = start
                        prob_sum = start_prob
                        count = 1
                        
                        for t, p in zip(timestamps[1:], video_probs[1:]):
                            if t - prev > 1:  # Gap in sequence
                                avg_prob = prob_sum / count
                                if avg_prob >= probability_threshold:
                                    segments.append((start, prev, avg_prob))
                                start = t
                                prob_sum = p
                                count = 1
                            else:
                                prob_sum += p
                                count += 1
                            prev = t
                            
                        # Add final segment if it meets threshold
                        avg_prob = prob_sum / count
                        if avg_prob >= probability_threshold:
                            segments.append((start, prev, avg_prob))
                    
                    if video_path not in intro_segments:
                        intro_segments[video_path] = []
                    intro_segments[video_path].extend(segments)
        
        return intro_segments

def main():
    # Initialize components
    segmenter = VideoSegmenter()
    detector = IntroDetector(
        min_cluster_size=3,
        min_samples=2,
        cluster_selection_epsilon=0.5
    )
    
    # Process videos
    video_dir = "path/to/videos"
    features_dict = {}
    
    for video_file in os.listdir(video_dir):
        if not video_file.endswith(('.mp4', '.avi', '.mkv')):
            continue
            
        video_path = os.path.join(video_dir, video_file)
        print(f"Processing {video_file}...")
        
        frames, timestamps = segmenter.extract_segments(video_path)
        features = segmenter.extract_features(frames)
        features_dict[video_path] = features
    
    # Detect intro segments
    intro_segments = detector.find_intros(
        features_dict,
        min_videos=3,
        probability_threshold=0.8
    )
    
    # Output results
    for video_path, segments in intro_segments.items():
        print(f"\nIntro segments for {os.path.basename(video_path)}:")
        for start, end, prob in segments:
            print(f"  {start}s - {end}s (confidence: {prob:.2%})")

if __name__ == "__main__":
    main()