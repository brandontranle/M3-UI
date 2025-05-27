#!/usr/bin/env python3
"""
FluidBrowse Gesture Recognition Model Training Framework - Simplified Version
Implements multiple AI models for mouse gesture classification without seaborn dependency
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
        
        # Direction changes
        direction_changes = 0
        if len(x_coords) > 2:
            for i in range(1, len(x_coords) - 1):
                v1 = np.array([x_coords[i] - x_coords[i-1], y_coords[i] - y_coords[i-1]])
                v2 = np.array([x_coords[i+1] - x_coords[i], y_coords[i+1] - y_coords[i]])
                
                # Calculate angle between vectors
                if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                    cos_angle = np.clip(cos_angle, -1, 1)
                    angle = np.arccos(cos_angle)
                    
                    if angle > np.pi / 4:  # 45 degrees
                        direction_changes += 1
        
        features['direction_changes'] = direction_changes
        features['direction_change_rate'] = direction_changes / max(len(x_coords) - 2, 1)
        
        # Straightness (how close to a straight line)
        if len(x_coords) > 1:
            start_end_distance = np.sqrt((x_coords[-1] - x_coords[0])**2 + (y_coords[-1] - y_coords[0])**2)
            total_path_length = np.sum([np.sqrt((x_coords[i] - x_coords[i-1])**2 + (y_coords[i] - y_coords[i-1])**2) 
                                     for i in range(1, len(x_coords))])
            features['straightness'] = start_end_distance / max(total_path_length, 1)
        
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
        
    def load_data(self, data_path):
        """Load gesture data from JSON files"""
        print("Loading gesture data...")
        
        all_data = []
        data_dir = Path(data_path)
        
        # Check if directory exists
        if not data_dir.exists():
            print(f"Error: Directory {data_path} does not exist!")
            # Try current directory instead
            data_dir = Path(".")
            print(f"Trying current directory: {data_dir.absolute()}")
        
        # Look for JSON files with various patterns
        json_patterns = ["*.json", "gesture_data_*.json", "*_gestures.json"]
        json_files = []
        
        for pattern in json_patterns:
            found_files = list(data_dir.glob(pattern))
            json_files.extend(found_files)
            print(f"Found {len(found_files)} files matching pattern '{pattern}'")
        
        # Remove duplicates
        json_files = list(set(json_files))
        
        if not json_files:
            print(f"No JSON files found in {data_dir.absolute()}")
            print("Looking for files with these patterns:")
            for pattern in json_patterns:
                print(f"  - {pattern}")
            
            # List all files in directory for debugging
            all_files = list(data_dir.glob("*"))
            print(f"\nAll files in directory:")
            for file in all_files:
                print(f"  - {file.name}")
            return []
        
        print(f"Found {len(json_files)} JSON files:")
        for file in json_files:
            print(f"  - {file.name}")
        
        for json_file in json_files:
            try:
                print(f"Loading {json_file.name}...")
                with open(json_file, 'r') as f:
                    participant_data = json.load(f)
                    
                    # Debug: show data structure
                    print(f"  Data keys: {list(participant_data.keys()) if isinstance(participant_data, dict) else 'List data'}")
                    
                    # Handle different data formats
                    if isinstance(participant_data, dict):
                        if 'rawData' in participant_data:
                            data_to_add = participant_data['rawData']
                            print(f"  Found {len(data_to_add)} samples in 'rawData'")
                            all_data.extend(data_to_add)
                        elif 'data' in participant_data:
                            data_to_add = participant_data['data']
                            print(f"  Found {len(data_to_add)} samples in 'data'")
                            all_data.extend(data_to_add)
                        elif 'collectedData' in participant_data:
                            data_to_add = participant_data['collectedData']
                            print(f"  Found {len(data_to_add)} samples in 'collectedData'")
                            all_data.extend(data_to_add)
                        else:
                            print(f"  Unknown dict format, trying all values...")
                            # Try to find lists in the data
                            for key, value in participant_data.items():
                                if isinstance(value, list) and len(value) > 0:
                                    # Check if this looks like gesture data
                                    if isinstance(value[0], dict) and 'points' in value[0]:
                                        print(f"  Found {len(value)} samples in '{key}'")
                                        all_data.extend(value)
                    elif isinstance(participant_data, list):
                        print(f"  Found {len(participant_data)} samples in list format")
                        all_data.extend(participant_data)
                    else:
                        print(f"  Unknown data format: {type(participant_data)}")
                        
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
        
        print(f"Total loaded: {len(all_data)} gesture samples from {len(json_files)} files")
        
        # Debug: show sample of loaded data
        if all_data:
            sample = all_data[0]
            print(f"Sample gesture keys: {list(sample.keys()) if isinstance(sample, dict) else 'Not a dict'}")
            if isinstance(sample, dict) and 'points' in sample:
                print(f"Sample has {len(sample['points'])} points")
        
        return all_data
    
    def preprocess_data(self, raw_data):
        """Extract features and prepare data for training"""
        print("Extracting features...")
        
        features_list = []
        labels = []
        
        for gesture in raw_data:
            # Extract features from gesture points
            features = self.feature_extractor.extract_features(gesture['points'])
            
            if features is not None:
                features_list.append(features)
                # Handle different label field names
                label = gesture.get('gestureType') or gesture.get('label') or gesture.get('type')
                labels.append(label)
        
        if not features_list:
            print("No valid features extracted!")
            return None, None, None
        
        # Convert to DataFrame
        df_features = pd.DataFrame(features_list)
        
        # Handle missing values
        df_features = df_features.fillna(0)
        
        # Store feature names
        self.feature_names = df_features.columns.tolist()
        
        # Convert to numpy arrays
        X = df_features.values
        y = np.array(labels)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        print(f"Feature matrix shape: {X_scaled.shape}")
        print(f"Labels distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
        
        return X_scaled, y_encoded, y
    
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
        }
    
    def cross_validate_models(self, X, y):
        """Perform cross-validation on all models"""
        print("Performing cross-validation...")
        
        cv_results = {}
        cv_folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for name, model in self.models.items():
            print(f"Cross-validating {name}...")
            scores = cross_val_score(model, X, y, cv=cv_folds, scoring='accuracy')
            cv_results[name] = {
                'mean_accuracy': scores.mean(),
                'std_accuracy': scores.std(),
                'scores': scores
            }
            print(f"{name}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        
        return cv_results
    
    def train_and_evaluate(self, X, y, y_original):
        """Train models and evaluate on test set"""
        print("Training models and evaluating on test set...")
        
        # Split data
        X_train, X_test, y_train, y_test, y_orig_train, y_orig_test = train_test_split(
            X, y, y_original, test_size=0.3, random_state=42, stratify=y
        )
        
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
        
        return results, X_test, y_test
    
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
    
    def plot_results(self, cv_results, test_results, feature_importance=None):
        """Create visualizations of results using matplotlib only"""
        print("Creating visualizations...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Gesture Recognition Model Evaluation', fontsize=16, fontweight='bold')
        
        # 1. Cross-validation results
        ax1 = axes[0, 0]
        model_names = list(cv_results.keys())
        cv_means = [cv_results[name]['mean_accuracy'] for name in model_names]
        cv_stds = [cv_results[name]['std_accuracy'] for name in model_names]
        
        bars = ax1.bar(model_names, cv_means, yerr=cv_stds, capsize=5, alpha=0.7, color='skyblue')
        ax1.set_title('Cross-Validation Accuracy', fontweight='bold')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1)
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, mean in zip(bars, cv_means):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{mean:.3f}', ha='center', va='bottom')
        
        # 2. Test set accuracy comparison
        ax2 = axes[0, 1]
        test_accuracies = [test_results[name]['test_accuracy'] for name in model_names]
        
        bars2 = ax2.bar(model_names, test_accuracies, alpha=0.7, color='orange')
        ax2.set_title('Test Set Accuracy', fontweight='bold')
        ax2.set_ylabel('Accuracy')
        ax2.set_ylim(0, 1)
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, acc in zip(bars2, test_accuracies):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{acc:.3f}', ha='center', va='bottom')
        
        # 3. Feature importance (if available)
        if feature_importance is not None:
            ax3 = axes[0, 2]
            top_features = feature_importance.head(10)
            ax3.barh(range(len(top_features)), top_features['importance'])
            ax3.set_yticks(range(len(top_features)))
            ax3.set_yticklabels(top_features['feature'])
            ax3.set_title('Top 10 Feature Importance', fontweight='bold')
            ax3.set_xlabel('Importance')
        else:
            axes[0, 2].text(0.5, 0.5, 'Feature importance\nnot available', 
                           ha='center', va='center', transform=axes[0, 2].transAxes)
            axes[0, 2].set_title('Feature Importance')
        
        # 4. Confusion matrix for best model
        best_model_name = max(test_results.keys(), key=lambda x: test_results[x]['test_accuracy'])
        ax4 = axes[1, 0]
        
        cm = test_results[best_model_name]['confusion_matrix']
        im = ax4.imshow(cm, interpolation='nearest', cmap='Blues')
        ax4.figure.colorbar(im, ax=ax4)
        
        # Add text annotations
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax4.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2. else "black")
        
        ax4.set_title(f'Confusion Matrix - {best_model_name}', fontweight='bold')
        ax4.set_ylabel('True Label')
        ax4.set_xlabel('Predicted Label')
        ax4.set_xticks(range(len(self.label_encoder.classes_)))
        ax4.set_yticks(range(len(self.label_encoder.classes_)))
        ax4.set_xticklabels(self.label_encoder.classes_)
        ax4.set_yticklabels(self.label_encoder.classes_)
        
        # 5. Model comparison (CV vs Test)
        ax5 = axes[1, 1]
        x_pos = np.arange(len(model_names))
        width = 0.35
        
        ax5.bar(x_pos - width/2, cv_means, width, label='Cross-Validation', alpha=0.7)
        ax5.bar(x_pos + width/2, test_accuracies, width, label='Test Set', alpha=0.7)
        
        ax5.set_title('CV vs Test Accuracy Comparison', fontweight='bold')
        ax5.set_ylabel('Accuracy')
        ax5.set_xticks(x_pos)
        ax5.set_xticklabels(model_names, rotation=45)
        ax5.legend()
        ax5.set_ylim(0, 1)
        
        # 6. Per-class performance for best model
        ax6 = axes[1, 2]
        report = test_results[best_model_name]['classification_report']
        
        classes = [cls for cls in report.keys() if cls not in ['accuracy', 'macro avg', 'weighted avg']]
        f1_scores = [report[cls]['f1-score'] for cls in classes]
        
        bars6 = ax6.bar(classes, f1_scores, alpha=0.7, color='green')
        ax6.set_title(f'Per-Class F1-Score - {best_model_name}', fontweight='bold')
        ax6.set_ylabel('F1-Score')
        ax6.tick_params(axis='x', rotation=45)
        ax6.set_ylim(0, 1)
        
        # Add value labels
        for bar, score in zip(bars6, f1_scores):
            ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{score:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('gesture_recognition_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def save_models(self, results, save_dir='models'):
        """Save trained models and preprocessing objects"""
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        # Save preprocessing objects
        with open(save_path / 'scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        with open(save_path / 'label_encoder.pkl', 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        # Save models
        for name, result in results.items():
            model_path = save_path / f'{name.lower()}_model.pkl'
            with open(model_path, 'wb') as f:
                pickle.dump(result['model'], f)
        
        # Save feature names
        with open(save_path / 'feature_names.json', 'w') as f:
            json.dump(self.feature_names, f)
        
        print(f"Models saved to {save_path}")
    
    def generate_report(self, cv_results, test_results, feature_importance=None):
        """Generate a comprehensive report"""
        report = {
            'dataset_summary': {
                'total_samples': len(self.label_encoder.classes_),
                'gesture_types': self.label_encoder.classes_.tolist(),
                'feature_count': len(self.feature_names)
            },
            'cross_validation_results': cv_results,
            'test_results': {
                name: {
                    'accuracy': result['test_accuracy'],
                    'classification_report': result['classification_report']
                }
                for name, result in test_results.items()
            }
        }
        
        if feature_importance is not None:
            report['feature_importance'] = feature_importance.to_dict('records')
        
        # Save report
        with open('gesture_recognition_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print("Report saved to gesture_recognition_report.json")
        return report



# Add these functions to your existing script to analyze test data in detail
def analyze_test_data(pipeline, X, y, y_original, test_results):
    """Detailed analysis of test set performance"""
    print("\n=== DETAILED TEST DATA ANALYSIS ===")
    
    # Recreate the same test split used in training
    from sklearn.model_selection import train_test_split
    X_train, X_test_split, y_train, y_test_split, y_orig_train, y_orig_test = train_test_split(
        X, y, y_original, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"Test set size: {len(y_test_split)} samples")
    print(f"Test set distribution:")
    
    # Count samples per class in test set
    test_counts = {}
    for label in y_orig_test:
        test_counts[label] = test_counts.get(label, 0) + 1
    
    for gesture, count in sorted(test_counts.items()):
        percentage = (count / len(y_orig_test)) * 100
        print(f"  - {gesture}: {count} samples ({percentage:.1f}%)")
    
    # Analyze individual predictions for best model
    best_model_name = max(test_results.keys(), key=lambda x: test_results[x]['test_accuracy'])
    best_model = test_results[best_model_name]['model']
    
    print(f"\nDetailed predictions using {best_model_name}:")
    
    # Get predictions
    y_pred = best_model.predict(X_test_split)
    
    # Try to get confidence scores
    try:
        if hasattr(best_model, 'predict_proba'):
            y_pred_proba = best_model.predict_proba(X_test_split)
            max_probabilities = np.max(y_pred_proba, axis=1)
        elif hasattr(best_model, 'decision_function'):
            decision_scores = best_model.decision_function(X_test_split)
            # For multi-class, decision_function returns array of shape (n_samples, n_classes)
            if len(decision_scores.shape) > 1:
                max_probabilities = np.max(np.abs(decision_scores), axis=1)
            else:
                max_probabilities = np.abs(decision_scores)
            # Normalize to [0,1] range for display
            if np.max(max_probabilities) > 0:
                max_probabilities = max_probabilities / np.max(max_probabilities)
        else:
            max_probabilities = np.ones(len(y_pred))  # Default confidence of 1.0
    except:
        max_probabilities = np.ones(len(y_pred))  # Fallback
    
    # Create detailed prediction analysis
    print(f"\nPer-sample predictions (first 20):")
    print(f"{'Index':<6} {'True':<10} {'Predicted':<10} {'Confidence':<10} {'Status'}")
    print("-" * 55)
    
    correct_count = 0
    for i in range(min(20, len(y_test_split))):
        true_label = pipeline.label_encoder.classes_[y_test_split[i]]
        pred_label = pipeline.label_encoder.classes_[y_pred[i]]
        confidence = max_probabilities[i]
        is_correct = "✓ Correct" if true_label == pred_label else "✗ Wrong"
        
        if true_label == pred_label:
            correct_count += 1
            
        print(f"{i+1:<6} {true_label:<10} {pred_label:<10} {confidence:<10.3f} {is_correct}")
    
    # Count all correct predictions
    total_correct = np.sum(y_test_split == y_pred)
    total_accuracy = total_correct / len(y_test_split)
    
    if len(y_test_split) > 20:
        print(f"... (showing first 20 of {len(y_test_split)} test samples)")
    
    print(f"\nTest Set Summary:")
    print(f"  - Correct predictions: {total_correct}/{len(y_test_split)}")
    print(f"  - Accuracy: {total_accuracy:.4f} ({total_accuracy*100:.2f}%)")
    print(f"  - Average confidence: {np.mean(max_probabilities):.3f}")
    
    # Show any misclassifications
    misclassified_indices = np.where(y_test_split != y_pred)[0]
    if len(misclassified_indices) > 0:
        print(f"\nMisclassified samples:")
        for idx in misclassified_indices[:5]:  # Show first 5 errors
            true_label = pipeline.label_encoder.classes_[y_test_split[idx]]
            pred_label = pipeline.label_encoder.classes_[y_pred[idx]]
            confidence = max_probabilities[idx]
            print(f"  Sample {idx+1}: {true_label} → {pred_label} (confidence: {confidence:.3f})")
    else:
        print(f"\n✅ No misclassifications found!")
    
    return X_test_split, y_test_split, y_orig_test, y_pred

def analyze_cross_validation_details(pipeline, X, y):
    """Detailed analysis of cross-validation folds"""
    print("\n=== CROSS-VALIDATION DETAILED ANALYSIS ===")
    
    from sklearn.model_selection import StratifiedKFold
    from sklearn.svm import SVC
    
    cv_folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    print("5-Fold Cross-Validation Breakdown:")
    print(f"{'Fold':<6} {'Train Size':<12} {'Test Size':<11} {'Accuracy':<10}")
    print("-" * 45)
    
    model = SVC(kernel='linear', random_state=42)
    fold_accuracies = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(cv_folds.split(X, y)):
        X_train_fold, X_test_fold = X[train_idx], X[test_idx]
        y_train_fold, y_test_fold = y[train_idx], y[test_idx]
        
        # Train and test on this fold
        model.fit(X_train_fold, y_train_fold)
        fold_accuracy = model.score(X_test_fold, y_test_fold)
        fold_accuracies.append(fold_accuracy)
        
        print(f"{fold_idx+1:<6} {len(train_idx):<12} {len(test_idx):<11} {fold_accuracy:<10.4f}")
    
    print("-" * 45)
    print(f"{'Mean':<6} {'':<12} {'':<11} {np.mean(fold_accuracies):<10.4f}")
    print(f"{'Std':<6} {'':<12} {'':<11} {np.std(fold_accuracies):<10.4f}")
    
    print(f"\nCross-Validation Insights:")
    print(f"  - Each fold tests on ~{len(X)//5} samples")
    print(f"  - Training set per fold: ~{len(X)*4//5} samples")  
    print(f"  - Consistency across folds: {np.std(fold_accuracies):.4f} std dev")
    
    if np.std(fold_accuracies) < 0.01:
        print("  ✅ Very consistent performance across all folds")
    elif np.std(fold_accuracies) < 0.05:
        print("  ✅ Good consistency across folds")
    else:
        print("  ⚠️  Some variation across folds")

def show_feature_analysis_for_test_samples(pipeline, X_test, y_test, sample_indices=[0, 1, 2]):
    """Show detailed feature analysis for specific test samples"""
    print(f"\n=== FEATURE ANALYSIS FOR TEST SAMPLES ===")
    
    for idx in sample_indices:
        if idx >= len(X_test):
            continue
            
        sample_features = X_test[idx]
        true_label = pipeline.label_encoder.classes_[y_test[idx]]
        
        print(f"\nSample {idx} - True Label: {true_label}")
        print(f"{'Feature':<20} {'Value':<12} {'Description'}")
        print("-" * 50)
        
        # Show key distinguishing features
        key_features = ['point_count', 'center_x', 'straightness', 'start_end_distance', 
                       'direction_changes', 'aspect_ratio', 'duration']
        
        for i, feature_name in enumerate(pipeline.feature_names):
            if feature_name in key_features:
                value = sample_features[i]
                description = get_feature_description(feature_name, value)
                print(f"{feature_name:<20} {value:<12.3f} {description}")

def get_feature_description(feature_name, value):
    """Get human-readable description of feature values"""
    descriptions = {
        'point_count': f"{'High detail' if value > 100 else 'Medium detail' if value > 50 else 'Low detail'}",
        'center_x': f"{'Right side' if value > 400 else 'Center' if value > 200 else 'Left side'}",
        'straightness': f"{'Very straight' if value > 0.8 else 'Somewhat curved' if value > 0.5 else 'Very curved'}",
        'start_end_distance': f"{'Open shape' if value > 100 else 'Partially closed' if value > 20 else 'Closed shape'}",
        'direction_changes': f"{'Very zigzag' if value > 20 else 'Some curves' if value > 10 else 'Smooth'}",
        'aspect_ratio': f"{'Wide' if value > 2 else 'Square' if 0.5 < value < 2 else 'Tall'}",
        'duration': f"{'Slow drawing' if value > 3000 else 'Medium speed' if value > 1500 else 'Fast drawing'}"
    }
    return descriptions.get(feature_name, "")






def analyze_cross_validation_details(pipeline, X, y):
    """Detailed analysis of cross-validation folds"""
    print("\n=== CROSS-VALIDATION DETAILED ANALYSIS ===")
    
    from sklearn.model_selection import StratifiedKFold
    from sklearn.svm import SVC
    
    cv_folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    print("5-Fold Cross-Validation Breakdown:")
    print(f"{'Fold':<6} {'Train Size':<12} {'Test Size':<11} {'Accuracy':<10}")
    print("-" * 45)
    
    model = SVC(kernel='linear', random_state=42)
    fold_accuracies = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(cv_folds.split(X, y)):
        X_train_fold, X_test_fold = X[train_idx], X[test_idx]
        y_train_fold, y_test_fold = y[train_idx], y[test_idx]
        
        # Train and test on this fold
        model.fit(X_train_fold, y_train_fold)
        fold_accuracy = model.score(X_test_fold, y_test_fold)
        fold_accuracies.append(fold_accuracy)
        
        print(f"{fold_idx+1:<6} {len(train_idx):<12} {len(test_idx):<11} {fold_accuracy:<10.4f}")
    
    print("-" * 45)
    print(f"{'Mean':<6} {'':<12} {'':<11} {np.mean(fold_accuracies):<10.4f}")
    print(f"{'Std':<6} {'':<12} {'':<11} {np.std(fold_accuracies):<10.4f}")
    
    print(f"\nCross-Validation Insights:")
    print(f"  - Each fold tests on ~{len(X)//5} samples")
    print(f"  - Training set per fold: ~{len(X)*4//5} samples")  
    print(f"  - Consistency across folds: {np.std(fold_accuracies):.4f} std dev")
    
    if np.std(fold_accuracies) < 0.01:
        print("  ✅ Very consistent performance across all folds")
    elif np.std(fold_accuracies) < 0.05:
        print("  ✅ Good consistency across folds")
    else:
        print("  ⚠️  Some variation across folds")

def show_feature_analysis_for_test_samples(pipeline, X_test, y_test, sample_indices=[0, 1, 2]):
    """Show detailed feature analysis for specific test samples"""
    print(f"\n=== FEATURE ANALYSIS FOR TEST SAMPLES ===")
    
    for idx in sample_indices:
        if idx >= len(X_test):
            continue
            
        sample_features = X_test[idx]
        true_label = pipeline.label_encoder.classes_[y_test[idx]]
        
        print(f"\nSample {idx} - True Label: {true_label}")
        print(f"{'Feature':<20} {'Value':<12} {'Description'}")
        print("-" * 50)
        
        # Show key distinguishing features
        key_features = ['point_count', 'center_x', 'straightness', 'start_end_distance', 
                       'direction_changes', 'aspect_ratio', 'duration']
        
        for i, feature_name in enumerate(pipeline.feature_names):
            if feature_name in key_features:
                value = sample_features[i]
                description = get_feature_description(feature_name, value)
                print(f"{feature_name:<20} {value:<12.3f} {description}")

def get_feature_description(feature_name, value):
    """Get human-readable description of feature values"""
    descriptions = {
        'point_count': f"{'High detail' if value > 100 else 'Medium detail' if value > 50 else 'Low detail'}",
        'center_x': f"{'Right side' if value > 400 else 'Center' if value > 200 else 'Left side'}",
        'straightness': f"{'Very straight' if value > 0.8 else 'Somewhat curved' if value > 0.5 else 'Very curved'}",
        'start_end_distance': f"{'Open shape' if value > 100 else 'Partially closed' if value > 20 else 'Closed shape'}",
        'direction_changes': f"{'Very zigzag' if value > 20 else 'Some curves' if value > 10 else 'Smooth'}",
        'aspect_ratio': f"{'Wide' if value > 2 else 'Square' if 0.5 < value < 2 else 'Tall'}",
        'duration': f"{'Slow drawing' if value > 3000 else 'Medium speed' if value > 1500 else 'Fast drawing'}"
    }
    return descriptions.get(feature_name, "")


def main():
    """Main execution function"""
    print("=== FluidBrowse Gesture Recognition Model Training ===\n")
    
    # Initialize pipeline
    pipeline = GestureRecognitionPipeline()
    
    # Load data (specify your data directory path)
    data_path = "gesture_data"  # Change this to your data directory
    raw_data = pipeline.load_data(data_path)
    
    if not raw_data:
        print("No data found! Please ensure gesture data files are in the specified directory.")
        return
    
    # Preprocess data
    X, y, y_original = pipeline.preprocess_data(raw_data)
    
    # Initialize models
    pipeline.initialize_models()
    
    # Cross-validation
    cv_results = pipeline.cross_validate_models(X, y)
    
    # Train and evaluate on test set
    test_results, X_test, y_test = pipeline.train_and_evaluate(X, y, y_original)
    

    # Add detailed analysis
    print("\n" + "="*60)
    
    # Analyze cross-validation in detail
    analyze_cross_validation_details(pipeline, X, y)
    
    # Analyze test data in detail - FIXED FUNCTION CALL
    X_test_detailed, y_test_detailed, y_orig_test, y_pred = analyze_test_data(
        pipeline, X, y, y_original, test_results  # Pass X, y, y_original instead of X_test, y_test
    )
    
    # Show feature analysis for specific samples
    show_feature_analysis_for_test_samples(pipeline, X_test_detailed, y_test_detailed)


    # Analyze feature importance
    feature_importance = pipeline.analyze_feature_importance(test_results)
    
    
    
    # Save models
    pipeline.save_models(test_results)
    
    print("Models and preprocessing objects saved successfully.")

    # Generate report
    # report = pipeline.generate_report(cv_results, test_results, feature_importance)
    
    # Print summary
    print("\n=== SUMMARY ===")
    print("Cross-Validation Results:")
    for name, result in cv_results.items():
        print(f"  {name}: {result['mean_accuracy']:.4f} (+/- {result['std_accuracy']*2:.4f})")
    
    print("\nTest Set Results:")
    for name, result in test_results.items():
        print(f"  {name}: {result['test_accuracy']:.4f}")
    
    best_model = max(test_results.keys(), key=lambda x: test_results[x]['test_accuracy'])
    print(f"\nBest Model: {best_model} (Test Accuracy: {test_results[best_model]['test_accuracy']:.4f})")
    
    print("\nTraining completed successfully!")
    add_diagnostic_checks(pipeline, X, y, y_original, test_results)

    # Create visualizations
    pipeline.plot_results(cv_results, test_results, feature_importance)


# Add these diagnostic functions to your script

def add_diagnostic_checks(pipeline, X, y, y_original, test_results):
    """Add diagnostic checks to verify model performance"""
    print("\n=== DIAGNOSTIC CHECKS ===")
    
    # 1. Check for data leakage
    print("1. Checking for potential data leakage...")
    
    # Look for duplicate or near-duplicate samples
    from sklearn.metrics.pairwise import cosine_similarity
    similarity_matrix = cosine_similarity(X)
    
    # Count high similarity pairs (excluding self-similarity)
    high_sim_count = 0
    threshold = 0.99
    for i in range(len(similarity_matrix)):
        for j in range(i+1, len(similarity_matrix)):
            if similarity_matrix[i][j] > threshold:
                high_sim_count += 1
    
    print(f"   - Samples with >99% similarity: {high_sim_count}")
    if high_sim_count > 0:
        print("   ⚠️  Warning: Some samples are very similar")
    else:
        print("   ✅ No suspicious sample similarity found")
    
    # 2. Check feature distributions per class
    print("\n2. Analyzing feature distributions per class...")
    
    import pandas as pd
    df = pd.DataFrame(X, columns=pipeline.feature_names)
    df['label'] = y_original
    
    # Calculate feature separability
    separable_features = []
    for feature in pipeline.feature_names:
        class_means = df.groupby('label')[feature].mean()
        overall_std = df[feature].std()
        
        # Check if class means are well separated
        max_diff = class_means.max() - class_means.min()
        if max_diff > 2 * overall_std:  # Classes are >2 std devs apart
            separable_features.append(feature)
    
    print(f"   - Features with clear class separation: {len(separable_features)}/{len(pipeline.feature_names)}")
    print(f"   - Most separable features: {separable_features[:5]}")
    
    # 3. Check model consistency across different random states
    print("\n3. Testing model stability with different random seeds...")
    
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    
    accuracies = []
    for seed in [42, 123, 456, 789, 999]:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=seed, stratify=y
        )
        
        model = SVC(kernel='rbf', random_state=seed)
        model.fit(X_train, y_train)
        acc = model.score(X_test, y_test)
        accuracies.append(acc)
    
    print(f"   - Accuracies across different splits: {[f'{acc:.3f}' for acc in accuracies]}")
    print(f"   - Mean accuracy: {np.mean(accuracies):.3f} ± {np.std(accuracies):.3f}")
    
    if np.std(accuracies) < 0.05:
        print("   ✅ Consistent performance across random splits")
    else:
        print("   ⚠️  High variance across splits")
    
    # 4. Manual inspection of misclassified samples
    print("\n4. Analyzing prediction confidence...")
    
    # Get prediction probabilities from SVM
    from sklearn.svm import SVC
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    model = SVC(kernel='rbf', probability=True, random_state=42)
    model.fit(X_train, y_train)
    y_pred_proba = model.predict_proba(X_test)
    
    # Check prediction confidence
    max_probs = np.max(y_pred_proba, axis=1)
    confident_predictions = np.sum(max_probs > 0.9)
    
    print(f"   - Test samples with >90% confidence: {confident_predictions}/{len(y_test)}")
    print(f"   - Average prediction confidence: {np.mean(max_probs):.3f}")
    
    # 5. Feature importance analysis
    print("\n5. Feature importance insights...")
    
    if 'Random_Forest' in test_results:
        rf_model = test_results['Random_Forest']['model']
        feature_importance = pd.DataFrame({
            'feature': pipeline.feature_names,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Check if a few features dominate
        top_5_importance = feature_importance.head(5)['importance'].sum()
        print(f"   - Top 5 features account for {top_5_importance:.1%} of total importance")
        
        if top_5_importance > 0.8:
            print("   ✅ A few key features strongly distinguish gestures")
        else:
            print("   - Importance is distributed across many features")
    
    # 6. Confusion matrix analysis
    print("\n6. Detailed confusion matrix analysis...")
    
    best_model_name = max(test_results.keys(), key=lambda x: test_results[x]['test_accuracy'])
    cm = test_results[best_model_name]['confusion_matrix']
    
    print(f"   - Confusion Matrix for {best_model_name}:")
    print(f"     {pipeline.label_encoder.classes_}")
    for i, row in enumerate(cm):
        print(f"     {pipeline.label_encoder.classes_[i]}: {row}")
    
    # Check for any misclassifications
    total_errors = np.sum(cm) - np.trace(cm)
    print(f"   - Total misclassifications: {total_errors}")
    
    if total_errors == 0:
        print("   ✅ Perfect classification - no confusion between classes")
    else:
        print(f"   - Most confused classes: {find_most_confused_classes(cm, pipeline.label_encoder.classes_)}")

def find_most_confused_classes(cm, class_names):
    """Find which classes are most often confused with each other"""
    max_confusion = 0
    confused_pair = None
    
    for i in range(len(cm)):
        for j in range(len(cm)):
            if i != j and cm[i][j] > max_confusion:
                max_confusion = cm[i][j]
                confused_pair = (class_names[i], class_names[j])
    
    return confused_pair if confused_pair else "None"



if __name__ == "__main__":
    main()