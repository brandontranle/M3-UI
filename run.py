#!/usr/bin/env python3
"""
FluidBrowse Gesture Recognition Model Training Framework - Participant-Based Split Version
Updated to use pre-computed features from JSON data instead of re-computing from raw points
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
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings('ignore')

class DirectFeatureExtractor:
    """Extract features directly from pre-computed feature objects in gesture data"""
    
    def __init__(self):
        self.feature_names = []
    
    def extract_features_from_data(self, gesture_data):
        """Extract features directly from the gesture data object"""
        if 'features' not in gesture_data:
            print(f"Warning: No 'features' object found in gesture data")
            return None
        
        features = gesture_data['features']
        extracted = {}
        
        # Direct features from the features object
        direct_features = [
            'pointCount', 'totalDistance', 'averageSpeed', 'directionChanges'
        ]
        
        for feature_name in direct_features:
            if feature_name in features:
                extracted[feature_name] = features[feature_name]
        
        # Bounding box features
        if 'boundingBox' in features:
            bbox = features['boundingBox']
            bbox_features = [
                'width', 'height', 'centerX', 'centerY', 'aspectRatio',
                'minX', 'maxX', 'minY', 'maxY'
            ]
            
            for bbox_feature in bbox_features:
                if bbox_feature in bbox:
                    extracted[f'bbox_{bbox_feature}'] = bbox[bbox_feature]
        
        # Top-level features (duration, etc.)
        if 'duration' in gesture_data:
            extracted['duration'] = gesture_data['duration']
        
        # Derived features we can calculate from existing ones
        if 'totalDistance' in extracted and 'duration' in extracted:
            # Speed = distance / time (more accurate than averageSpeed if duration is in ms)
            extracted['calculated_speed'] = extracted['totalDistance'] / max(extracted['duration'] / 1000.0, 0.001)
        
        if 'directionChanges' in extracted and 'pointCount' in extracted:
            # Direction change rate
            extracted['direction_change_rate'] = extracted['directionChanges'] / max(extracted['pointCount'] - 1, 1)
        
        if 'bbox_width' in extracted and 'bbox_height' in extracted:
            # Bounding box area
            extracted['bbox_area'] = extracted['bbox_width'] * extracted['bbox_height']
            # Bounding box perimeter
            extracted['bbox_perimeter'] = 2 * (extracted['bbox_width'] + extracted['bbox_height'])
        
        #if 'totalDistance' in extracted and 'bbox_width' in extracted and 'bbox_height' in extracted:
            # Path efficiency (how direct the path is)
            #diagonal = np.sqrt(extracted['bbox_width']**2 + extracted['bbox_height']**2)
            #extracted['path_efficiency'] = diagonal / max(extracted['totalDistance'], 1)
        
        #if 'pointCount' in extracted and 'duration' in extracted:
            # Point density (points per second)
         #   extracted['point_density'] = extracted['pointCount'] / max(extracted['duration'] / 1000.0, 0.001)
        
        # Normalized features for better cross-device compatibility
        #if 'bbox_width' in extracted and 'bbox_height' in extracted:
         #   total_size = extracted['bbox_width'] + extracted['bbox_height']
          #  if total_size > 0:
           #     extracted['normalized_width'] = extracted['bbox_width'] / total_size
            #    extracted['normalized_height'] = extracted['bbox_height'] / total_size
        
        return extracted

class GestureRecognitionPipeline:
    """Complete pipeline for gesture recognition model training and evaluation"""
    
    def __init__(self):
        self.feature_extractor = DirectFeatureExtractor()
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.models = {}
        self.feature_names = []
        self.participant_data = {}  # Store data by participant
        
    def load_data_with_participants(self, data_path):
        """Load gesture data and keep track of participants"""
        print("Loading gesture data with participant tracking...")
        
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
                    
                    # Validate that gestures have features
                    valid_gestures = []
                    for gesture in gesture_data:
                        if 'features' in gesture:
                            valid_gestures.append(gesture)
                        else:
                            print(f"  Warning: Gesture missing 'features' object, skipping")
                    
                    # Store participant data
                    participant_data[participant_id] = valid_gestures
                    all_data.extend(valid_gestures)
                    
                    print(f"  Found {len(valid_gestures)} valid samples for {participant_id}")
                        
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
        
        # Show what features are available
        if all_data:
            sample_gesture = all_data[0]
            print(f"\nAvailable features in data:")
            if 'features' in sample_gesture:
                print(f"  Direct features: {list(sample_gesture['features'].keys())}")
                if 'boundingBox' in sample_gesture['features']:
                    print(f"  Bounding box features: {list(sample_gesture['features']['boundingBox'].keys())}")
            if 'duration' in sample_gesture:
                print(f"  Top-level features: duration")
        
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
        print("🚀 Extracting features from pre-computed data (FAST)...")
        
        # Process training data
        train_features_list = []
        train_labels = []
        
        for gesture in train_data:
            features = self.feature_extractor.extract_features_from_data(gesture)
            if features is not None:
                train_features_list.append(features)
                label = gesture.get('gestureType') or gesture.get('label') or gesture.get('type')
                train_labels.append(label)
        
        # Process test data
        test_features_list = []
        test_labels = []
        
        for gesture in test_data:
            features = self.feature_extractor.extract_features_from_data(gesture)
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
        
        print(f"✅ Extracted features: {list(train_df.columns)}")
        
        # Ensure same feature columns
        common_features = list(set(train_df.columns) & set(test_df.columns))
        train_df = train_df[common_features]
        test_df = test_df[common_features]
        
        # Handle missing values
        train_df = train_df.fillna(0)
        test_df = test_df.fillna(0)
        
        # Store feature names
        self.feature_names = train_df.columns.tolist()
        
        print(f"Final feature set ({len(self.feature_names)} features): {self.feature_names}")
        
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
            'Decision_Tree': DecisionTreeClassifier(random_state=42),
            'Naive_Bayes': GaussianNB()
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
            
            print("\nTop 10 most important features (Random Forest):")
            print(feature_importance.head(10))
            

        if 'Decision_Tree' in results:
            dt_model = results['Decision_Tree']['model']
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': dt_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\nTop 10 most important features (Decision Tree):")
            print(feature_importance.head(10))
            
        if 'MLP' in results:
            mlp_model = results['MLP']['model']
            # MLP does not have feature importance, but we can analyze weights
            weights = mlp_model.coefs_[0]
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': np.abs(weights).sum(axis=1)
            }).sort_values('importance', ascending=False)
            print("\nTop 10 most important features (MLP weights):")
            print(feature_importance.head(10))

        if 'SVM_Linear' in results:
            svm_model = results['SVM_Linear']['model']
            if hasattr(svm_model, 'coef_'):
                feature_importance = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': np.abs(svm_model.coef_[0])
                }).sort_values('importance', ascending=False)
                print("\nTop 10 most important features (SVM Linear):")
                print(feature_importance.head(10))
            else:
                print("SVM Linear model does not have feature importance.")

        if 'SVM_RBF' in results:
            svm_model = results['SVM_RBF']['model']
            if hasattr(svm_model, 'coef_'):
                feature_importance = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': np.abs(svm_model.coef_[0])
                }).sort_values('importance', ascending=False)
                print("\nTop 10 most important features (SVM RBF):")
                print(feature_importance.head(10))
            else:
                print("SVM RBF model does not have feature importance.")
        
        if 'KNN' in results:
            knn_model = results['KNN']['model']
            # KNN does not have feature importance, but we can analyze distances
            print("KNN model does not provide feature importance directly.")

        if 'Logistic_Regression' in results:
            lr_model = results['Logistic_Regression']['model']
            if hasattr(lr_model, 'coef_'):
                feature_importance = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': np.abs(lr_model.coef_[0])
                }).sort_values('importance', ascending=False)
                print("\nTop 10 most important features (Logistic Regression):")
                print(feature_importance.head(10))
            else:
                print("Logistic Regression model does not have feature importance.")
        
        if 'Linear_Regression' in results:
            lr_model = results['Linear_Regression']['model']
            if hasattr(lr_model, 'coef_'):
                feature_importance = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': np.abs(lr_model.coef_)
                }).sort_values('importance', ascending=False)
                print("\nTop 10 most important features (Linear Regression):")
                print(feature_importance.head(10))
            else:
                print("Linear Regression model does not have feature importance.")
            
            if 'Naive_Bayes' in results:
                nb_model = results['Naive_Bayes']['model']
                # Naive Bayes does not have feature importance, but we can analyze probabilities
                print("Naive Bayes model does not provide feature importance directly.")

        return None

def main():
    """Main execution function with participant-based split option"""
    print("=== FluidBrowse Gesture Recognition - Optimized with Direct Feature Extraction ===\n")
    
    # Initialize pipeline
    pipeline = GestureRecognitionPipeline()
    
    # Load data with participant tracking
    data_path = "./noisy"
    all_data, participant_data = pipeline.load_data_with_participants(data_path)
    
    if not all_data:
        print("No data found!")
        return
    
    print(f"\n🚀 USING PRE-COMPUTED FEATURES - MUCH FASTER!")
    print(f"   No need to re-calculate features from raw points")
    
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
    
    # Fallback to random split or if user chose option 2
    if split_choice != "1":
        print(f"\n=== TRADITIONAL RANDOM SPLIT ===")
        print("Random split not implemented in this optimized version.")
        print("Participant-based split is more realistic for gesture recognition evaluation.")

if __name__ == "__main__":
    main()