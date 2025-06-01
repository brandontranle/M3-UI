#!/usr/bin/env python3
"""
FluidBrowse Gesture Recognition Model Training Framework - Participant-Based Split Version
Allows selection of specific participants as test set
MODIFIED: Uses pre-processed noisy data from noisy_gesture_data directory
"""

import json
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings('ignore')

class GestureFeatureExtractor:
    """Extract features from raw gesture point data"""
    
    def __init__(self):
        self.feature_names = []
    
    def extract_features(self, points):
        """Extract comprehensive features from gesture points"""
        if len(points) < 2:
            return None
            
        points = np.array([(p['x'], p['y'], p['timestamp']) for p in points])
        x_coords = points[:, 0]
        y_coords = points[:, 1]
        timestamps = points[:, 2]
        
        features = {}
        
        # Basic geometric features
        features.update(self._geometric_features(x_coords, y_coords))
        
        # Temporal features
        features.update(self._temporal_features(timestamps))
        
        # Movement features
        features.update(self._movement_features(x_coords, y_coords, timestamps))
        
        # Shape features
        features.update(self._shape_features(x_coords, y_coords))
        
        # Statistical features
        features.update(self._statistical_features(x_coords, y_coords))
        
        return features
    
    def _geometric_features(self, x_coords, y_coords):
        """Basic geometric properties"""
        return {
            'point_count': len(x_coords),
            'width': np.max(x_coords) - np.min(x_coords),
            'height': np.max(y_coords) - np.min(y_coords),
            'aspect_ratio': (np.max(x_coords) - np.min(x_coords)) / max(np.max(y_coords) - np.min(y_coords), 1),
            'bounding_box_area': (np.max(x_coords) - np.min(x_coords)) * (np.max(y_coords) - np.min(y_coords)),
            'center_x': np.mean(x_coords),
            'center_y': np.mean(y_coords),
            'start_end_distance': np.sqrt((x_coords[-1] - x_coords[0])**2 + (y_coords[-1] - y_coords[0])**2)
        }
    
    def _temporal_features(self, timestamps):
        """Time-based features"""
        duration = timestamps[-1] - timestamps[0]
        return {
            'duration': duration,
            'point_density': len(timestamps) / max(duration, 1) * 1000,  # points per second
        }
    
    def _movement_features(self, x_coords, y_coords, timestamps):
        """Movement and velocity features"""
        features = {}
        
        # Calculate distances between consecutive points
        distances = []
        velocities = []
        
        for i in range(1, len(x_coords)):
            dx = x_coords[i] - x_coords[i-1]
            dy = y_coords[i] - y_coords[i-1]
            dt = timestamps[i] - timestamps[i-1]
            
            distance = np.sqrt(dx**2 + dy**2)
            distances.append(distance)
            
            if dt > 0:
                velocity = distance / dt * 1000  # pixels per second
                velocities.append(velocity)
        
        if distances:
            features.update({
                'total_distance': np.sum(distances),
                'average_distance': np.mean(distances),
                'distance_std': np.std(distances),
                'max_distance': np.max(distances),
            })
        
        if velocities:
            features.update({
                'average_velocity': np.mean(velocities),
                'velocity_std': np.std(velocities),
                'max_velocity': np.max(velocities),
                'min_velocity': np.min(velocities),
            })
        
        return features
    
    def _shape_features(self, x_coords, y_coords):
        """Shape and curvature features"""
        features = {}
        
        # Currently no shape features - removed direction changes and straightness as requested
        
        return features
    
    def _statistical_features(self, x_coords, y_coords):
        """Statistical properties of coordinates"""
        return {
            'x_variance': np.var(x_coords),
            'y_variance': np.var(y_coords),
            'x_std': np.std(x_coords),
            'y_std': np.std(y_coords),
            'x_range': np.max(x_coords) - np.min(x_coords),
            'y_range': np.max(y_coords) - np.min(y_coords),
        }

class GestureRecognitionPipeline:
    """Complete pipeline for gesture recognition model training and evaluation"""
    
    def __init__(self):
        self.feature_extractor = GestureFeatureExtractor()
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.models = {}
        self.feature_names = []
        self.participant_data = {}  # Store data by participant
        
    def load_data_with_participants(self, data_path):
        """Load gesture data and keep track of participants"""
        print("Loading gesture data with participant tracking...")
        print("📁 Using pre-processed noisy data from *noisy* directory")
        
        all_data = []
        participant_data = {}
        data_dir = Path(data_path)
        
        # Check if directory exists
        if not data_dir.exists():
            print(f"Error: Directory {data_path} does not exist!")
            data_dir = Path(".")
            print(f"Trying current directory: {data_dir.absolute()}")
        
        # Look for JSON files with various patterns
        json_patterns = ["*.json", "gesture_data_*.json", "*_gestures.json"]
        json_files = []
        
        for pattern in json_patterns:
            found_files = list(data_dir.glob(pattern))
            json_files.extend(found_files)
        
        # Remove duplicates and filter out report files
        json_files = list(set(json_files))
        json_files = [f for f in json_files if 'report' not in f.name.lower()]
        
        if not json_files:
            print(f"No JSON files found in {data_dir.absolute()}")
            return [], {}
        
        print(f"Found {len(json_files)} participant data files:")
        
        for json_file in json_files:
            try:
                # Extract participant ID from filename
                participant_id = self.extract_participant_id(json_file.name)
                print(f"Loading {json_file.name} (Participant: {participant_id})...")
                
                with open(json_file, 'r') as f:
                    file_data = json.load(f)
                    
                    # Handle different data formats
                    if isinstance(file_data, dict):
                        if 'rawData' in file_data:
                            gesture_data = file_data['rawData']
                        elif 'data' in file_data:
                            gesture_data = file_data['data']
                        else:
                            continue
                    elif isinstance(file_data, list):
                        gesture_data = file_data
                    else:
                        continue
                    
                    # NO NOISE ADDITION - using pre-processed noisy data
                    # Store participant data directly
                    participant_data[participant_id] = gesture_data
                    all_data.extend(gesture_data)
                    
                    # Check if noise metadata exists
                    if isinstance(file_data, dict) and 'noise_metadata' in file_data:
                        noise_info = file_data['noise_metadata']
                        print(f"  📊 Found {len(gesture_data)} samples with dramatic noise applied")
                        print(f"     Base offset: {noise_info['noise_params']['base_offset']}px")
                        print(f"     Scale factor: {noise_info['noise_params']['scale_factor']:.2f}x")
                    else:
                        print(f"  Found {len(gesture_data)} samples for {participant_id}")
                        
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
        
        print(f"Total loaded: {len(all_data)} gesture samples from {len(participant_data)} participants")
        
        print("Loaded participants:", list(participant_data.keys()))
        for pid, gestures in participant_data.items():
            print(f"  • {pid}: {len(gestures)} samples")

        # Show participant summary
        print(f"\nParticipant Summary:")
        for pid, data in participant_data.items():
            gesture_counts = {}
            for gesture in data:
                gtype = gesture.get('gestureType', 'unknown')
                gesture_counts[gtype] = gesture_counts.get(gtype, 0) + 1
            print(f"  {pid}: {len(data)} total samples - {gesture_counts}")
        
        self.participant_data = participant_data
        return all_data, participant_data
    
    def extract_participant_id(self, filename):
        """Extract participant ID from filename"""
        # Examples: gesture_data_BL831_1748305982324.json -> BL831
        parts = filename.replace('.json', '').split('_')
        if len(parts) >= 3:
            return parts[2]  # gesture_data_[ID]_timestamp
        return filename.replace('.json', '')
    
    def create_participant_based_split(self, all_data, participant_data, test_participants=None):
        """Create train/test split based on participants"""
        print(f"\n=== PARTICIPANT-BASED DATA SPLIT ===")
        
        available_participants = list(participant_data.keys())
        print(f"Available participants: {available_participants}")
        
        if test_participants is None:
            # Interactive selection
            print(f"\nSelect test participants (others will be used for training):")
            for i, pid in enumerate(available_participants):
                sample_count = len(participant_data[pid])
                print(f"  {i+1}. {pid} ({sample_count} samples)")
            
            print(f"\nEnter participant numbers for test set (e.g., '1,3,5' or '1-3'):")
            print(f"Or enter participant IDs directly (e.g., 'BL831,TK831'):")
            
            selection = input("Your selection: ").strip()
            test_participants = self.parse_participant_selection(selection, available_participants)
        
        if not test_participants:
            print("No test participants selected. Using random split.")
            return None, None, None, None, None, None
        
        # Ensure test participants exist
        test_participants = [p for p in test_participants if p in available_participants]
        train_participants = [p for p in available_participants if p not in test_participants]
        
        print(f"\nSelected split:")
        print(f"  Training participants: {train_participants} ({len(train_participants)} participants)")
        print(f"  Test participants: {test_participants} ({len(test_participants)} participants)")
        
        # Create train/test datasets
        train_data = []
        test_data = []
        
        for pid in train_participants:
            train_data.extend(participant_data[pid])
        
        for pid in test_participants:
            test_data.extend(participant_data[pid])
        
        print(f"  Training samples: {len(train_data)}")
        print(f"  Test samples: {len(test_data)}")
        
        # Check class balance
        train_counts = {}
        test_counts = {}
        
        for gesture in train_data:
            gtype = gesture.get('gestureType', 'unknown')
            train_counts[gtype] = train_counts.get(gtype, 0) + 1
            
        for gesture in test_data:
            gtype = gesture.get('gestureType', 'unknown')
            test_counts[gtype] = test_counts.get(gtype, 0) + 1
        
        print(f"\nClass distribution:")
        print(f"  Training: {train_counts}")
        print(f"  Test: {test_counts}")
        
        return train_data, test_data, train_participants, test_participants, train_counts, test_counts
    
    def parse_participant_selection(self, selection, available_participants):
        """Parse user selection of participants"""
        test_participants = []
        
        try:
            if '-' in selection:
                # Range selection (e.g., "1-3")
                start, end = map(int, selection.split('-'))
                indices = list(range(start-1, end))  # Convert to 0-based
                test_participants = [available_participants[i] for i in indices if 0 <= i < len(available_participants)]
            elif ',' in selection and selection.replace(',', '').replace(' ', '').isdigit():
                # Number selection (e.g., "1,3,5")
                indices = [int(x.strip()) - 1 for x in selection.split(',')]  # Convert to 0-based
                test_participants = [available_participants[i] for i in indices if 0 <= i < len(available_participants)]
            elif ',' in selection:
                # Direct participant ID selection (e.g., "BL831,TK831")
                test_participants = [x.strip() for x in selection.split(',')]
            elif selection.isdigit():
                # Single number
                idx = int(selection) - 1
                if 0 <= idx < len(available_participants):
                    test_participants = [available_participants[idx]]
            else:
                # Single participant ID
                test_participants = [selection]
                
        except (ValueError, IndexError):
            print(f"Invalid selection: {selection}")
            return []
        
        return test_participants
    
    def preprocess_data_with_split(self, train_data, test_data):
        """Extract features and prepare data for training with participant split"""
        print("Extracting features for participant-based split...")
        
        # Process training data
        train_features_list = []
        train_labels = []
        
        for gesture in train_data:
            features = self.feature_extractor.extract_features(gesture['points'])
            if features is not None:
                train_features_list.append(features)
                label = gesture.get('gestureType') or gesture.get('label') or gesture.get('type')
                train_labels.append(label)
        
        # Process test data
        test_features_list = []
        test_labels = []
        
        for gesture in test_data:
            features = self.feature_extractor.extract_features(gesture['points'])
            if features is not None:
                test_features_list.append(features)
                label = gesture.get('gestureType') or gesture.get('label') or gesture.get('type')
                test_labels.append(label)
        
        if not train_features_list or not test_features_list:
            print("Insufficient data for training or testing!")
            return None, None, None, None, None, None
        
        # Convert to DataFrames
        train_df = pd.DataFrame(train_features_list)
        test_df = pd.DataFrame(test_features_list)
        
        # Ensure same feature columns
        common_features = list(set(train_df.columns) & set(test_df.columns))
        train_df = train_df[common_features]
        test_df = test_df[common_features]
        
        # Handle missing values
        train_df = train_df.fillna(0)
        test_df = test_df.fillna(0)
        
        # Store feature names
        self.feature_names = train_df.columns.tolist()
        
        # Convert to numpy arrays
        X_train = train_df.values
        X_test = test_df.values
        y_train = np.array(train_labels)
        y_test = np.array(test_labels)
        
        # Encode labels
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_test_encoded = self.label_encoder.transform(y_test)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"Training feature matrix shape: {X_train_scaled.shape}")
        print(f"Test feature matrix shape: {X_test_scaled.shape}")
        print(f"Training labels distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")
        print(f"Test labels distribution: {dict(zip(*np.unique(y_test, return_counts=True)))}")
        
        return X_train_scaled, X_test_scaled, y_train_encoded, y_test_encoded, y_train, y_test
    
    def initialize_models(self):
        """Initialize different ML models for comparison"""
        print("Initializing models...")
        
        self.models = {
            'SVM_Linear': SVC(kernel='linear', random_state=42),
            'SVM_RBF': SVC(kernel='rbf', random_state=42),
            'Random_Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'MLP': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42),
            'KNN': KNeighborsClassifier(n_neighbors=5),
            'Logistic_Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Decision_Tree': DecisionTreeClassifier(random_state=42)   
            # Add more models as needed
            

        }
    
    def cross_validate_models_participant_split(self, X_train, y_train):
        """Perform cross-validation on training data only"""
        print("Performing cross-validation on training data...")
        
        cv_results = {}
        cv_folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for name, model in self.models.items():
            print(f"Cross-validating {name}...")
            scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='accuracy')
            cv_results[name] = {
                'mean_accuracy': scores.mean(),
                'std_accuracy': scores.std(),
                'scores': scores
            }
            print(f"{name}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        
        return cv_results
    
    def train_and_evaluate_participant_split(self, X_train, X_test, y_train, y_test, y_train_orig, y_test_orig):
        """Train models and evaluate on participant-based test set"""
        print("Training models and evaluating on participant-based test set...")
        
        results = {}
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predict on test set
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            results[name] = {
                'model': model,
                'test_accuracy': accuracy,
                'classification_report': classification_report(
                    y_test, y_pred, 
                    target_names=self.label_encoder.classes_, 
                    output_dict=True
                ),
                'confusion_matrix': confusion_matrix(y_test, y_pred),
                'predictions': y_pred,
                'true_labels': y_test
            }
            
            print(f"{name} Test Accuracy: {accuracy:.4f}")
        
        return results
    
    def analyze_feature_importance(self, results):
        """Analyze feature importance for tree-based models"""
        print("Analyzing feature importance...")
        
        # Random Forest feature importance
        if 'Random_Forest' in results:
            rf_model = results['Random_Forest']['model']
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("Top 10 most important features (Random Forest):")
            print(feature_importance.head(10))
            
            return feature_importance
        
        return None

def main():
    """Main execution function with participant-based split option"""
    print("=== FluidBrowse Gesture Recognition - Participant Split Version ===\n")
    print("MODIFIED: Using pre-processed dramatically noisy datasets\n")
    
    # Initialize pipeline
    pipeline = GestureRecognitionPipeline()
    
    # Load data from noisy directory (pre-processed with dramatic noise)
    data_path = "noisy"
    all_data, participant_data = pipeline.load_data_with_participants(data_path)
    
    if not all_data:
        print("No data found!")
        return
    
    print(f"\n{'='*60}")
    print("SPLIT SELECTION:")
    print("1. Participant-based split (recommended for realistic evaluation)")
    print("2. Random split (traditional approach)")
    
    split_choice = input("Choose split method (1 or 2): ").strip()
    
    if split_choice == "1":
        # Participant-based split
        print(f"\n=== PARTICIPANT-BASED SPLIT ===")
        
        # Show default suggestion
        available_participants = list(participant_data.keys())
        n_participants = len(available_participants)
        suggested_test_count = max(2, n_participants // 4)  # ~25% of participants
        suggested_test = available_participants[-suggested_test_count:]  # Last few participants
        
        print(f"Suggested test participants ({suggested_test_count} out of {n_participants}): {suggested_test}")
        print(f"This will use ~{sum(len(participant_data[p]) for p in suggested_test)} samples for testing")
        
        use_suggestion = input(f"Use suggested split? (y/n): ").strip().lower()
        
        if use_suggestion in ['y', 'yes', '']:
            test_participants = suggested_test
        else:
            test_participants = None  # Will trigger interactive selection
        
        # Create participant-based split
        train_data, test_data, train_participants, test_participants, train_counts, test_counts = pipeline.create_participant_based_split(
            all_data, participant_data, test_participants
        )
        
        if train_data is None:
            print("Falling back to random split...")
            split_choice = "2"
        else:
            # Preprocess with participant split
            X_train, X_test, y_train, y_test, y_train_orig, y_test_orig = pipeline.preprocess_data_with_split(train_data, test_data)
            
            if X_train is None:
                print("Data preprocessing failed!")
                return
            
            # Initialize models
            pipeline.initialize_models()
            
            # Cross-validation on training data only
            cv_results = pipeline.cross_validate_models_participant_split(X_train, y_train)
            
            # Train and evaluate
            test_results = pipeline.train_and_evaluate_participant_split(
                X_train, X_test, y_train, y_test, y_train_orig, y_test_orig
            )
            
            # Analyze feature importance
            feature_importance = pipeline.analyze_feature_importance(test_results)
            
            # Print results
            print(f"\n=== PARTICIPANT-BASED SPLIT RESULTS ===")
            print(f"Training participants: {train_participants}")
            print(f"Test participants: {test_participants}")
            print(f"Training samples: {len(train_data)}")
            print(f"Test samples: {len(test_data)}")
            
            print("\nCross-Validation Results (Training data only):")
            for name, result in cv_results.items():
                print(f"  {name}: {result['mean_accuracy']:.4f} (+/- {result['std_accuracy']*2:.4f})")
            
            print("\nTest Set Results (Unseen participants):")
            for name, result in test_results.items():
                print(f"  {name}: {result['test_accuracy']:.4f}")
            
            best_model = max(test_results.keys(), key=lambda x: test_results[x]['test_accuracy'])
            print(f"\nBest Model: {best_model} (Test Accuracy: {test_results[best_model]['test_accuracy']:.4f})")
            
            print(f"\n🎯 This represents performance on COMPLETELY NEW USERS!")
            print(f"   Models were trained on {len(train_participants)} participants")
            print(f"   and tested on {len(test_participants)} different participants")
            print(f"   📊 Using DRAMATICALLY differentiated datasets with huge coordinate differences")
    
    # Fallback to random split or if user chose option 2
    if split_choice != "1":
        print(f"\n=== TRADITIONAL RANDOM SPLIT ===")
        print("Using traditional random split would go here")

if __name__ == "__main__":
    main()