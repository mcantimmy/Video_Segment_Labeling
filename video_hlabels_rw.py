import numpy as np
import cv2
import hdbscan
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import argparse
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm
import logging
from concurrent.futures import ThreadPoolExecutor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("VideoIntroDetector")

@dataclass
class IntroSegment:
    """Data class for intro segments with start/end timestamps and confidence"""
    start: int
    end: int
    confidence: float
    
    def duration(self) -> int:
        """Calculate duration of the segment in seconds"""
        return self.end - self.start + 1
    
    def __str__(self) -> str:
        """String representation of the segment"""
        return f"{self.start}s - {self.end}s (confidence: {self.confidence:.2%}, duration: {self.duration()}s)"


class VideoSegmenter:
    """Extracts frames and features from videos for intro detection"""
    def __init__(self, 
                 sample_rate: int = 1, 
                 resize_dim: Tuple[int, int] = (224, 224),
                 device: Optional[str] = None):
        """
        Initialize the video segmenter
        
        Args:
            sample_rate: Sample one frame every N frames
            resize_dim: Dimensions to resize frames to
            device: Device to run inference on ('cuda', 'cpu', or None for auto-detection)
        """
        self.sample_rate = sample_rate
        self.resize_dim = resize_dim
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        logger.info(f"Using device: {self.device}")
        
        self.feature_extractor = self._build_feature_extractor()
        self.feature_extractor.eval()  # Set to evaluation mode
        self.feature_extractor.to(self.device)
    
    def _build_feature_extractor(self) -> nn.Module:
        """Build a feature extractor using a pre-trained CNN"""
        try:
            # Using ResNet50 pretrained model for feature extraction
            model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
            # Take all layers except the final classification layer
            return nn.Sequential(*list(model.children())[:-1])
        except Exception as e:
            logger.error(f"Error loading pretrained model: {e}")
            raise
    
    def extract_segments(self, video_path: str, max_frames: Optional[int] = None) -> Tuple[List[np.ndarray], List[int]]:
        """
        Extract frames and timestamps from a video
        
        Args:
            video_path: Path to the video file
            max_frames: Maximum number of frames to extract (for debugging)
            
        Returns:
            Tuple of (frames, timestamps)
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Failed to open video: {video_path}")
            
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frames = []
        timestamps = []
        frame_count = 0
        
        try:
            with tqdm(desc=f"Extracting frames from {Path(video_path).name}", unit="frames") as pbar:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                        
                    if frame_count % (fps * self.sample_rate) == 0:  # Sample one frame per N seconds
                        frame = cv2.resize(frame, self.resize_dim)
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frames.append(frame)
                        timestamps.append(frame_count // fps)
                        
                    frame_count += 1
                    pbar.update(1)
                    
                    if max_frames and len(frames) >= max_frames:
                        logger.info(f"Reached max_frames limit ({max_frames})")
                        break
        finally:
            cap.release()
            
        logger.info(f"Extracted {len(frames)} frames from {video_path}")
        return frames, timestamps
    
    def extract_features(self, frames: List[np.ndarray], batch_size: int = 16) -> np.ndarray:
        """
        Extract features from frames using batched processing
        
        Args:
            frames: List of frames
            batch_size: Batch size for processing
            
        Returns:
            Array of feature vectors
        """
        if not frames:
            logger.warning("No frames to extract features from")
            return np.array([])
            
        features = []
        frame_batches = [frames[i:i+batch_size] for i in range(0, len(frames), batch_size)]
        
        with torch.no_grad():
            for batch in tqdm(frame_batches, desc="Extracting features", unit="batch"):
                # Preprocess frames
                batch_tensor = torch.stack([
                    torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
                    for frame in batch
                ]).to(self.device)
                
                # Extract features
                batch_features = self.feature_extractor(batch_tensor)
                batch_features = batch_features.squeeze().cpu().numpy()
                
                # Handle single-item batches
                if len(batch) == 1:
                    batch_features = batch_features.reshape(1, -1)
                    
                features.append(batch_features)
                
        return np.vstack(features)


class IntroDetector:
    """Detects intro segments across multiple videos using HDBSCAN clustering"""
    def __init__(self, 
                 min_cluster_size: int = 3, 
                 min_samples: int = 2,
                 cluster_selection_epsilon: float = 0.5,
                 prediction_data: bool = True,
                 metric: str = 'cosine'):
        """
        Initialize HDBSCAN-based intro detector
        
        Args:
            min_cluster_size: Minimum size of clusters
            min_samples: Number of samples in neighborhood for core point
            cluster_selection_epsilon: Distance threshold for cluster membership
            prediction_data: Whether to store prediction data for membership probabilities
            metric: Distance metric for clustering
        """
        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=cluster_selection_epsilon,
            metric=metric,
            prediction_data=prediction_data
        )
        
    def find_intros(self, 
                    features_dict: Dict[str, np.ndarray],
                    timestamps_dict: Dict[str, List[int]],
                    min_videos: int = 2,
                    probability_threshold: float = 0.815,
                    min_segment_length: int = 3,
                    max_segment_length: int = 120
                    ) -> Dict[str, List[IntroSegment]]:
        """
        Find intro segments across multiple videos
        
        Args:
            features_dict: Dictionary mapping video paths to feature arrays
            timestamps_dict: Dictionary mapping video paths to timestamp lists
            min_videos: Minimum number of videos a segment must appear in
            probability_threshold: Minimum probability for cluster membership
            min_segment_length: Minimum length of intro segment in seconds
            max_segment_length: Maximum length of intro segment in seconds
            
        Returns:
            Dictionary mapping video paths to lists of IntroSegment objects
        """
        # Validate inputs
        if not features_dict:
            logger.warning("No features provided for clustering")
            return {}
            
        for video_path in features_dict:
            if video_path not in timestamps_dict:
                raise ValueError(f"Timestamps missing for video: {video_path}")
            if len(features_dict[video_path]) != len(timestamps_dict[video_path]):
                raise ValueError(f"Feature and timestamp counts don't match for {video_path}")
        
        # Combine features from all videos
        all_features = []
        feature_map = []  # Keep track of which video and timestamp each feature belongs to
        
        for video_path, features in features_dict.items():
            if len(features) == 0:
                logger.warning(f"No features for {video_path}, skipping")
                continue
                
            timestamps = timestamps_dict[video_path]
            all_features.extend(features)
            feature_map.extend([(video_path, i, timestamps[i]) for i in range(len(features))])
            
        if not all_features:
            logger.warning("No features to cluster")
            return {}
            
        # Convert to numpy array for clustering
        all_features = np.vstack(all_features)
        
        # Cluster features using HDBSCAN
        logger.info(f"Clustering {len(all_features)} features")
        self.clusterer.fit(all_features)
        
        # Get cluster labels and probabilities
        labels = self.clusterer.labels_
        probabilities = self.clusterer.probabilities_
        
        # Find segments that appear in multiple videos
        intro_segments = {}
        unique_labels = set(labels)
        
        logger.info(f"Found {len(unique_labels)} clusters ({len(unique_labels) - (1 if -1 in unique_labels else 0)} non-noise)")
        
        for label in unique_labels:
            if label == -1:  # Skip noise
                continue
                
            # Get indices and probabilities for this cluster
            cluster_indices = np.where(labels == label)[0]
            cluster_probs = probabilities[cluster_indices]
            
            # Get videos containing this cluster
            videos_in_cluster = set(feature_map[i][0] for i in cluster_indices)
            
            if len(videos_in_cluster) >= min_videos:  # Segment appears in multiple videos
                logger.info(f"Cluster {label} appears in {len(videos_in_cluster)} videos")
                
                for video_path in videos_in_cluster:
                    # Get indices and probabilities for this video
                    video_indices = [i for i, fm in enumerate(cluster_indices) 
                                   if feature_map[fm][0] == video_path]
                    video_probs = cluster_probs[video_indices]
                    
                    # Use actual timestamps instead of feature indices
                    timestamps = [feature_map[cluster_indices[i]][2] for i in video_indices]
                    
                    # Convert consecutive timestamps to segments with probabilities
                    segments = self._timestamps_to_segments(
                        timestamps, 
                        video_probs, 
                        probability_threshold,
                        min_segment_length,
                        max_segment_length
                    )
                    
                    if segments:
                        if video_path not in intro_segments:
                            intro_segments[video_path] = []
                        intro_segments[video_path].extend(segments)
        
        # Sort segments by start time
        for video_path in intro_segments:
            intro_segments[video_path].sort(key=lambda x: x.start)
            
        logger.info(f"Found intro segments in {len(intro_segments)} videos")
        return intro_segments
    
    def _timestamps_to_segments(self, 
                               timestamps: List[int], 
                               probabilities: List[float],
                               probability_threshold: float,
                               min_segment_length: int,
                               max_segment_length: int) -> List[IntroSegment]:
        """
        Convert timestamps to continuous segments
        
        Args:
            timestamps: List of timestamps in seconds
            probabilities: Corresponding confidence probabilities
            probability_threshold: Minimum probability for a valid segment
            min_segment_length: Minimum length of segment in seconds
            max_segment_length: Maximum length of segment in seconds
            
        Returns:
            List of IntroSegment objects
        """
        segments = []
        
        if not timestamps:
            return segments
            
        # Sort by timestamp to ensure proper ordering
        timestamp_probs = sorted(zip(timestamps, probabilities), key=lambda x: x[0])
        timestamps, probabilities = zip(*timestamp_probs)
        
        start = timestamps[0]
        start_prob = probabilities[0]
        prev = start
        prob_sum = start_prob
        count = 1
        
        for t, p in zip(timestamps[1:], probabilities[1:]):
            if t - prev > 1:  # Gap in sequence
                avg_prob = prob_sum / count
                duration = prev - start + 1
                
                if (avg_prob >= probability_threshold and 
                    duration >= min_segment_length and
                    duration <= max_segment_length):
                    segments.append(IntroSegment(start, prev, avg_prob))
                
                start = t
                prob_sum = p
                count = 1
            else:
                prob_sum += p
                count += 1
            prev = t
            
        # Add final segment if it meets threshold
        avg_prob = prob_sum / count
        duration = prev - start + 1
        
        if (avg_prob >= probability_threshold and 
            duration >= min_segment_length and
            duration <= max_segment_length):
            segments.append(IntroSegment(start, prev, avg_prob))
            
        return segments


def process_video(
    video_path: str,
    segmenter: VideoSegmenter,
    max_frames: Optional[int] = None
) -> Tuple[np.ndarray, List[int]]:
    """
    Process a single video to extract features and timestamps
    
    Args:
        video_path: Path to the video
        segmenter: VideoSegmenter instance
        max_frames: Maximum number of frames to process
        
    Returns:
        Tuple of (features, timestamps)
    """
    try:
        frames, timestamps = segmenter.extract_segments(video_path, max_frames)
        features = segmenter.extract_features(frames)
        return features, timestamps
    except Exception as e:
        logger.error(f"Error processing {video_path}: {e}")
        return np.array([]), []


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Detect common intro segments across videos")
    parser.add_argument("--video_dir", type=str, required=True, help="Directory containing videos")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save results")
    parser.add_argument("--min_videos", type=int, default=2, help="Minimum number of videos a segment must appear in")
    parser.add_argument("--confidence", type=float, default=0.825, help="Minimum confidence threshold")
    parser.add_argument("--min_segment_length", type=int, default=3, help="Minimum intro segment length in seconds")
    parser.add_argument("--max_segment_length", type=int, default=120, help="Maximum intro segment length in seconds")
    parser.add_argument("--sample_rate", type=int, default=1, help="Sample one frame every N seconds")
    parser.add_argument("--max_frames", type=int, default=None, help="Maximum frames to process per video (for debugging)")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda/cpu)")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up logging to file
    file_handler = logging.FileHandler(os.path.join(args.output_dir, "intro_detection.log"))
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Initialize components
    segmenter = VideoSegmenter(sample_rate=args.sample_rate, device=args.device)
    detector = IntroDetector(
        min_cluster_size=3,
        min_samples=2,
        cluster_selection_epsilon=0.5
    )
    
    # Find video files
    video_dir = Path(args.video_dir)
    video_files = [
        str(f) for f in video_dir.glob("**/*") 
        if f.suffix.lower() in ('.mp4', '.avi', '.mkv', '.mov', '.webm')
    ]
    
    if not video_files:
        logger.error(f"No video files found in {args.video_dir}")
        return
        
    logger.info(f"Found {len(video_files)} video files in {args.video_dir}")
    
    # Process videos
    features_dict = {}
    timestamps_dict = {}
    
    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_video, video_path, segmenter, args.max_frames): video_path
            for video_path in video_files
        }
        
        for future in tqdm(futures, desc="Processing videos", unit="video"):
            video_path = futures[future]
            try:
                features, timestamps = future.result()
                if len(features) > 0:
                    features_dict[video_path] = features
                    timestamps_dict[video_path] = timestamps
            except Exception as e:
                logger.error(f"Error processing {video_path}: {e}")
    
    if not features_dict:
        logger.error("No valid features extracted from any videos")
        return
    
    # Detect intro segments
    intro_segments = detector.find_intros(
        features_dict,
        timestamps_dict,
        min_videos=args.min_videos,
        probability_threshold=args.confidence,
        min_segment_length=args.min_segment_length,
        max_segment_length=args.max_segment_length
    )
    
    # Output results
    results_file = os.path.join(args.output_dir, "intro_segments.txt")
    with open(results_file, 'w') as f:
        for video_path, segments in intro_segments.items():
            video_name = Path(video_path).name
            f.write(f"\n{'-' * 50}\n")
            f.write(f"Intro segments for {video_name}:\n")
            
            for segment in segments:
                f.write(f"  {segment}\n")
                
            # Also output to console
            print(f"\nIntro segments for {video_name}:")
            for segment in segments:
                print(f"  {segment}")
    
    logger.info(f"Results saved to {results_file}")
    print(f"\nDetection complete! Results saved to {results_file}")


if __name__ == "__main__":
    main()