#!/usr/bin/env python3
"""
JSON File Noise Addition Script - Creates Noisy Copies WITH CORRECT FEATURES
Creates a new directory with noisy versions of JSON files
Recalculates features from the noisy points to maintain consistency
Leaves original files completely untouched
"""

import json
import numpy as np
from pathlib import Path
import shutil
from datetime import datetime

class FeatureCalculator:
    """Calculate features from gesture points (matches your original feature extraction)"""
    
    @staticmethod
    def calculate_features(points):
        """Calculate all features from a list of points"""
        if len(points) < 2:
            return None
        
        # Extract coordinates and timestamps
        x_coords = np.array([p['x'] for p in points])
        y_coords = np.array([p['y'] for p in points])
        timestamps = np.array([p['timestamp'] for p in points])
        
        features = {}
        
        # Basic counts
        features['pointCount'] = len(points)
        
        # Calculate total distance
        total_distance = 0
        distances = []
        for i in range(1, len(points)):
            dx = x_coords[i] - x_coords[i-1]
            dy = y_coords[i] - y_coords[i-1]
            distance = np.sqrt(dx**2 + dy**2)
            distances.append(distance)
            total_distance += distance
        
        features['totalDistance'] = total_distance
        
        # Calculate average speed
        duration_ms = timestamps[-1] - timestamps[0]
        if duration_ms > 0:
            features['averageSpeed'] = total_distance / (duration_ms / 1000.0)  # pixels per second
        else:
            features['averageSpeed'] = 0
        
        # Calculate direction changes
        direction_changes = 0
        if len(points) > 2:
            for i in range(1, len(points) - 1):
                # Vector from point i-1 to point i
                v1 = np.array([x_coords[i] - x_coords[i-1], y_coords[i] - y_coords[i-1]])
                # Vector from point i to point i+1
                v2 = np.array([x_coords[i+1] - x_coords[i], y_coords[i+1] - y_coords[i]])
                
                # Calculate angle between vectors
                if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                    cos_angle = np.clip(cos_angle, -1, 1)
                    angle = np.arccos(cos_angle)
                    
                    # Count significant direction changes (threshold: 45 degrees)
                    if angle > np.pi / 4:
                        direction_changes += 1
        
        features['directionChanges'] = direction_changes
        
        # Calculate bounding box
        min_x = np.min(x_coords)
        max_x = np.max(x_coords)
        min_y = np.min(y_coords)
        max_y = np.max(y_coords)
        
        width = max_x - min_x
        height = max_y - min_y
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        aspect_ratio = width / max(height, 1)  # Avoid division by zero
        
        features['boundingBox'] = {
            'minX': min_x,
            'maxX': max_x,
            'minY': min_y,
            'maxY': max_y,
            'width': width,
            'height': height,
            'centerX': center_x,
            'centerY': center_y,
            'aspectRatio': aspect_ratio
        }
        
        return features

class NoisyCopyCreator:
    """Create noisy copies of JSON files in a new directory with recalculated features"""
    
    def __init__(self):
        self.noisy_dir = Path("noisy")  # Changed to match your data path
        self.processed_files = []
        self.feature_calculator = FeatureCalculator()
        
    def create_noisy_directory(self):
        """Create directory for noisy copies"""
        if self.noisy_dir.exists():
            # If directory exists, ask what to do
            print(f"Directory '{self.noisy_dir}' already exists.")
            choice = input("Delete and recreate? (y/n): ").strip().lower()
            if choice in ['y', 'yes']:
                shutil.rmtree(self.noisy_dir)
                print(f"Deleted existing directory")
            else:
                print("Using existing directory (files may be overwritten)")
        
        self.noisy_dir.mkdir(exist_ok=True)
        print(f"✅ Created/using noisy data directory: {self.noisy_dir}")
    
    def extract_participant_id(self, filename):
        """Extract participant ID from filename"""
        parts = filename.replace('.json', '').split('_')
        if len(parts) >= 3:
            return parts[2]  # gesture_data_[ID]_timestamp
        return filename.replace('.json', '')
    
    def generate_participant_noise_params(self, participant_id):
        """Generate DRAMATIC noise parameters for strong differentiation"""
        # Use participant ID to generate consistent but VERY different offsets
        np.random.seed(hash(participant_id) % 2**32)
        
        # Generate different offset ranges for different participants
        hash_val = hash(participant_id) % 1000
        
        # MUCH MORE AGGRESSIVE offset ranges (in pixels)
        base_offset = (hash_val % 8) * 100  # 0, 100, 200, 300, 400, 500, 600, 700
        min_offset = base_offset + 50       # 50-750 range starts
        max_offset = base_offset + 200      # 200-900 range ends
        
        # More dramatic direction patterns
        direction_pattern = hash_val % 4
        if direction_pattern == 0:
            x_multiplier, y_multiplier = 1, 1      # Both positive (top-right quadrant)
        elif direction_pattern == 1:
            x_multiplier, y_multiplier = -1, 1     # Left-right, up (top-left)
        elif direction_pattern == 2:
            x_multiplier, y_multiplier = 1, -1     # Right, down (bottom-right)
        else:
            x_multiplier, y_multiplier = -1, -1    # Both negative (bottom-left)
        
        # Add scaling factor for even more differentiation
        scale_factor = 0.5 + (hash_val % 30) / 10.0  # 0.5 to 3.5x scaling
        
        return {
            'min_offset': min_offset,
            'max_offset': max_offset,
            'x_multiplier': x_multiplier,
            'y_multiplier': y_multiplier,
            'scale_factor': scale_factor,
            'base_offset': base_offset,
            'direction_pattern': direction_pattern
        }
    
    def add_noise_to_gesture(self, gesture, participant_id, gesture_idx, noise_params):
        """Add DRAMATIC coordinate noise to a single gesture AND recalculate features"""
        # Generate offset for this specific gesture
        gesture_type = gesture.get('gestureType', 'unknown')
        gesture_seed = hash(f"{participant_id}_{gesture_type}_{gesture_idx}") % 2**32
        np.random.seed(gesture_seed)
        
        # Random offset for this gesture (much larger range)
        x_offset = np.random.uniform(noise_params['min_offset'], noise_params['max_offset'])
        y_offset = np.random.uniform(noise_params['min_offset'], noise_params['max_offset'])
        
        # Apply direction multipliers
        x_offset *= noise_params['x_multiplier']
        y_offset *= noise_params['y_multiplier']
        
        # Get original gesture center for scaling
        original_points = gesture['points']
        x_coords = [p['x'] for p in original_points]
        y_coords = [p['y'] for p in original_points]
        center_x = np.mean(x_coords)
        center_y = np.mean(y_coords)
        
        # Create modified gesture (deep copy)
        modified_gesture = gesture.copy()
        modified_points = []
        
        for point in original_points:
            # Apply scaling around center first
            scaled_x = center_x + (point['x'] - center_x) * noise_params['scale_factor']
            scaled_y = center_y + (point['y'] - center_y) * noise_params['scale_factor']
            
            # Then apply dramatic offset
            modified_point = {
                'x': scaled_x + x_offset,
                'y': scaled_y + y_offset,
                'timestamp': point['timestamp']  # Keep timestamp unchanged
            }
            modified_points.append(modified_point)
        
        modified_gesture['points'] = modified_points
        
        # 🚀 RECALCULATE FEATURES FROM NOISY POINTS
        print(f"    📊 Recalculating features for {gesture_type} gesture...")
        
        # Calculate new features from noisy points
        new_features = self.feature_calculator.calculate_features(modified_points)
        
        if new_features:
            modified_gesture['features'] = new_features
            
            # Show the difference
            if 'features' in gesture:
                original_features = gesture['features']
                print(f"      Original totalDistance: {original_features.get('totalDistance', 'N/A'):.1f}")
                print(f"      New totalDistance: {new_features['totalDistance']:.1f}")
                print(f"      Original bbox width×height: {original_features.get('boundingBox', {}).get('width', 'N/A'):.1f}×{original_features.get('boundingBox', {}).get('height', 'N/A'):.1f}")
                print(f"      New bbox width×height: {new_features['boundingBox']['width']:.1f}×{new_features['boundingBox']['height']:.1f}")
        else:
            print(f"      ⚠️ Could not calculate features for {gesture_type}")
        
        # Keep original duration (time-based, unaffected by spatial noise)
        if 'duration' in gesture:
            modified_gesture['duration'] = gesture['duration']
        
        # Add metadata about the dramatic noise applied
        modified_gesture['noise_applied'] = {
            'x_offset': x_offset,
            'y_offset': y_offset,
            'scale_factor': noise_params['scale_factor'],
            'direction_pattern': noise_params['direction_pattern'],
            'base_offset': noise_params['base_offset'],
            'timestamp_modified': datetime.now().isoformat(),
            'noise_version': '3.0_FEATURES_RECALCULATED',
            'features_recalculated': True
        }
        
        return modified_gesture
    
    def create_noisy_copy(self, original_file_path):
        """Create a noisy copy of a single JSON file"""
        try:
            participant_id = self.extract_participant_id(original_file_path.name)
            print(f"\nProcessing: {original_file_path.name}")
            print(f"  Participant ID: {participant_id}")
            
            # Load original data
            with open(original_file_path, 'r') as f:
                file_data = json.load(f)
            
            # Handle different data formats
            if isinstance(file_data, dict):
                if 'rawData' in file_data:
                    gesture_data = file_data['rawData']
                    data_key = 'rawData'
                elif 'data' in file_data:
                    gesture_data = file_data['data']
                    data_key = 'data'
                else:
                    print(f"  ❌ Unknown data format in {original_file_path.name}")
                    return False
            elif isinstance(file_data, list):
                gesture_data = file_data
                data_key = None  # Direct list format
            else:
                print(f"  ❌ Unsupported file format in {original_file_path.name}")
                return False
            
            # Generate noise parameters for this participant
            noise_params = self.generate_participant_noise_params(participant_id)
            print(f"  🎯 DRAMATIC NOISE APPLIED:")
            print(f"     Base offset: {noise_params['base_offset']}px")
            print(f"     Range: {noise_params['min_offset']:.1f} to {noise_params['max_offset']:.1f} pixels")
            print(f"     Direction: x={noise_params['x_multiplier']}, y={noise_params['y_multiplier']}")
            print(f"     Scale factor: {noise_params['scale_factor']:.2f}x")
            print(f"     Quadrant pattern: {noise_params['direction_pattern']}")
            
            # Process each gesture
            modified_gestures = []
            for gesture_idx, gesture in enumerate(gesture_data):
                print(f"  🔧 Processing gesture {gesture_idx + 1}/{len(gesture_data)}")
                modified_gesture = self.add_noise_to_gesture(
                    gesture, participant_id, gesture_idx, noise_params
                )
                modified_gestures.append(modified_gesture)
            
            # Create modified file data
            if data_key:
                # Copy the original file structure
                modified_file_data = file_data.copy()
                modified_file_data[data_key] = modified_gestures
                
                # Add metadata to file
                modified_file_data['noise_metadata'] = {
                    'noise_applied': True,
                    'participant_id': participant_id,
                    'noise_params': noise_params,
                    'gestures_modified': len(modified_gestures),
                    'features_recalculated': True,
                    'modification_timestamp': datetime.now().isoformat(),
                    'script_version': '3.0_FEATURES_FIXED',
                    'original_file': original_file_path.name
                }
            else:
                modified_file_data = modified_gestures
            
            # Create new filename in noisy directory
            noisy_file_path = self.noisy_dir / original_file_path.name
            
            # Write noisy data to new file
            with open(noisy_file_path, 'w') as f:
                json.dump(modified_file_data, f, indent=2)
            
            print(f"  ✅ Modified {len(modified_gestures)} gestures")
            print(f"  ✅ Recalculated all features from noisy points")
            print(f"  ✅ Created noisy copy: {noisy_file_path}")
            
            self.processed_files.append({
                'original_file': original_file_path.name,
                'noisy_file': noisy_file_path.name,
                'participant': participant_id,
                'gestures_modified': len(modified_gestures),
                'noise_params': noise_params
            })
            
            return True
            
        except Exception as e:
            print(f"  ❌ Error processing {original_file_path}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def find_and_process_all_files(self, source_directory="./raw"):
        """Find all JSON files and create noisy copies"""
        print("🔧 JSON NOISE COPY CREATOR v3.0 - WITH FEATURE RECALCULATION")
        print("=" * 60)
        print("Creates noisy copies while preserving originals")
        print("✨ NEW: Recalculates features from noisy points!")
        print()
        
        source_dir = Path(source_directory)
        
        # Find JSON files
        json_files = list(source_dir.glob("*.json"))
        json_files = [f for f in json_files if 'report' not in f.name.lower()]
        
        if not json_files:
            print(f"❌ No JSON files found in {source_dir.absolute()}")
            return
        
        print(f"Found {len(json_files)} JSON files to process:")
        for f in json_files:
            print(f"  • {f.name}")
        
        print(f"\n📁 Noisy copies will be created in: {self.noisy_dir}")
        print(f"🛡️  Original files will remain unchanged")
        print(f"🔄 Features will be recalculated from noisy points")
        
        confirm = input("\nContinue? (yes/no): ").strip().lower()
        if confirm not in ['yes', 'y']:
            print("Operation cancelled.")
            return
        
        # Create noisy directory
        self.create_noisy_directory()
        
        # Process each file
        print(f"\n🔧 Creating noisy copies with recalculated features...")
        success_count = 0
        
        for json_file in json_files:
            if self.create_noisy_copy(json_file):
                success_count += 1
        
        # Print summary
        print(f"\n📊 PROCESSING SUMMARY")
        print("=" * 30)
        print(f"Files processed successfully: {success_count}/{len(json_files)}")
        print(f"Noisy copies created in: {self.noisy_dir}")
        print(f"Original files: UNCHANGED ✅")
        print(f"Features: RECALCULATED from noisy points ✅")
        
        if self.processed_files:
            print(f"\nDetailed Results:")
            for result in self.processed_files:
                print(f"  📄 {result['original_file']} → {result['noisy_file']}")
                print(f"     Participant: {result['participant']}")
                print(f"     Gestures: {result['gestures_modified']}")
                print(f"     Noise: {result['noise_params']['base_offset']}px base + {result['noise_params']['min_offset']:.0f}-{result['noise_params']['max_offset']:.0f}px")
                print(f"     Scale: {result['noise_params']['scale_factor']:.2f}x, Quadrant: {result['noise_params']['direction_pattern']}")
                print()
        
        print(f"✅ All files processed!")
        print(f"🎯 Use files in '{self.noisy_dir}' for your experiments")
        print(f"🛡️  Your original files in '{source_dir}' are untouched")
        print(f"🔄 All features now match the noisy coordinate data!")
    
    def list_created_files(self):
        """List the files in the noisy directory"""
        if not self.noisy_dir.exists():
            print(f"❌ Noisy directory not found: {self.noisy_dir}")
            return
        
        noisy_files = list(self.noisy_dir.glob("*.json"))
        if not noisy_files:
            print(f"❌ No files found in {self.noisy_dir}")
            return
        
        print(f"📁 Files in {self.noisy_dir}:")
        for f in noisy_files:
            file_size = f.stat().st_size / 1024  # KB
            print(f"  • {f.name} ({file_size:.1f} KB)")
        
        print(f"\nTotal: {len(noisy_files)} files")

def main():
    """Main execution function"""
    creator = NoisyCopyCreator()
    
    print("Select operation:")
    print("1. Create noisy copies of JSON files (with feature recalculation)")
    print("2. List files in noisy directory")
    print("3. Exit")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == "1":
        creator.find_and_process_all_files()
    elif choice == "2":
        creator.list_created_files()
    elif choice == "3":
        print("Exiting...")
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main()