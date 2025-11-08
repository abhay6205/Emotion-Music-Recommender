import cv2
import numpy as np
import mediapipe as mp
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import os
import random
import pygame
import pickle
from collections import deque

class EmotionMusicRecommender:
    def __init__(self):
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize drawing utilities
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
        
        # Emotion clusters
        self.n_clusters = 5
        self.kmeans = None
        self.pca = None
        
        # Music mapping
        self.emotion_music_map = {
            0: 'happy',
            1: 'sad', 
            2: 'energetic',
            3: 'calm',
            4: 'neutral'
        }
        
        # Music directory structure
        self.music_dir = 'music'
        self.playlists = {
            'happy': [],
            'sad': [],
            'energetic': [],
            'calm': [],
            'neutral': []
        }
        
        # Initialize pygame mixer
        pygame.mixer.init()
        
        # Smoothing buffer
        self.emotion_buffer = deque(maxlen=10)
        self.current_emotion = None
        self.current_song = None
        
    def extract_facial_landmarks(self, image):
        """Extract facial landmarks from image"""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            # Extract landmark coordinates
            coords = []
            for landmark in landmarks.landmark:
                coords.extend([landmark.x, landmark.y, landmark.z])
            return np.array(coords), landmarks
        return None, None
    
    def extract_emotion_features(self, landmarks_array):
        """Extract emotion-specific features from landmarks"""
        if landmarks_array is None:
            return None
            
        # Key facial regions for emotion detection
        # Eyes: landmarks 33, 133, 362, 263
        # Mouth: landmarks 61, 291, 0, 17
        # Eyebrows: landmarks 70, 300, 107, 336
        
        features = []
        
        # Calculate distances and ratios
        # Eye aspect ratio
        left_eye_height = abs(landmarks_array[159*3+1] - landmarks_array[145*3+1])
        left_eye_width = abs(landmarks_array[33*3] - landmarks_array[133*3])
        right_eye_height = abs(landmarks_array[386*3+1] - landmarks_array[374*3+1])
        right_eye_width = abs(landmarks_array[362*3] - landmarks_array[263*3])
        
        features.extend([left_eye_height, left_eye_width, right_eye_height, right_eye_width])
        
        # Mouth aspect ratio
        mouth_height = abs(landmarks_array[13*3+1] - landmarks_array[14*3+1])
        mouth_width = abs(landmarks_array[61*3] - landmarks_array[291*3])
        features.extend([mouth_height, mouth_width])
        
        # Eyebrow position
        left_eyebrow_y = landmarks_array[70*3+1]
        right_eyebrow_y = landmarks_array[300*3+1]
        features.extend([left_eyebrow_y, right_eyebrow_y])
        
        return np.array(features)
    
    def train_emotion_model(self, training_data):
        """Train PCA and K-Means model on collected features"""
        if len(training_data) < self.n_clusters:
            print("Not enough training data")
            return False
            
        # Apply PCA for dimensionality reduction
        self.pca = PCA(n_components=min(5, len(training_data[0])))
        features_pca = self.pca.fit_transform(training_data)
        
        # Apply K-Means clustering
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        self.kmeans.fit(features_pca)
        
        print(f"Model trained with {len(training_data)} samples")
        return True
    
    def predict_emotion(self, features):
        """Predict emotion cluster from features"""
        if self.pca is None or self.kmeans is None:
            return None
            
        features_pca = self.pca.transform([features])
        cluster = self.kmeans.predict(features_pca)[0]
        return cluster
    
    def smooth_emotion_prediction(self, emotion):
        """Smooth emotion predictions using buffer"""
        self.emotion_buffer.append(emotion)
        
        # Get most common emotion in buffer
        if len(self.emotion_buffer) >= 5:
            emotion_counts = np.bincount(list(self.emotion_buffer))
            smoothed_emotion = np.argmax(emotion_counts)
            return smoothed_emotion
        return emotion
    
    def load_music_library(self):
        """Load music files from directory structure"""
        if not os.path.exists(self.music_dir):
            os.makedirs(self.music_dir)
            for emotion in self.playlists.keys():
                os.makedirs(os.path.join(self.music_dir, emotion), exist_ok=True)
            print(f"Created music directory structure at: {self.music_dir}")
            print("Please add music files to corresponding emotion folders")
            return False
        
        for emotion in self.playlists.keys():
            emotion_dir = os.path.join(self.music_dir, emotion)
            if os.path.exists(emotion_dir):
                files = [f for f in os.listdir(emotion_dir) if f.endswith(('.mp3', '.wav'))]
                self.playlists[emotion] = [os.path.join(emotion_dir, f) for f in files]
                print(f"Loaded {len(files)} songs for {emotion}")
        
        return True
    
    def play_music_for_emotion(self, emotion_cluster):
        """Play music based on detected emotion"""
        emotion_name = self.emotion_music_map.get(emotion_cluster, 'neutral')
        playlist = self.playlists.get(emotion_name, [])
        
        if not playlist:
            print(f"No music available for emotion: {emotion_name}")
            return None
        
        # If same emotion, continue current song
        if self.current_emotion == emotion_cluster:
            if pygame.mixer.music.get_busy():
                return self.current_song
        
        # New emotion detected, play new song
        song = random.choice(playlist)
        try:
            pygame.mixer.music.load(song)
            pygame.mixer.music.play()
            self.current_emotion = emotion_cluster
            self.current_song = song
            print(f"Playing: {os.path.basename(song)} for emotion: {emotion_name}")
            return song
        except Exception as e:
            print(f"Error playing music: {e}")
            return None
    
    def save_model(self, filename='emotion_model.pkl'):
        """Save trained model"""
        if self.pca and self.kmeans:
            with open(filename, 'wb') as f:
                pickle.dump({'pca': self.pca, 'kmeans': self.kmeans}, f)
            print(f"Model saved to {filename}")
    
    def load_model(self, filename='emotion_model.pkl'):
        """Load trained model"""
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.pca = data['pca']
                self.kmeans = data['kmeans']
            print(f"Model loaded from {filename}")
            return True
        return False
    
    def run_realtime(self):
        """Run real-time emotion detection and music recommendation"""
        # Load music library
        music_loaded = self.load_music_library()
        
        # Try to load existing model
        model_loaded = self.load_model()
        
        if not model_loaded:
            print("No pre-trained model found. Collecting training data...")
            print("Press 'c' to collect samples, 't' to train model, 'q' to quit")
            training_mode = True
            training_data = []
        else:
            print("Model loaded successfully!")
            training_mode = False
            training_data = None
        
        # Open webcam
        cap = cv2.VideoCapture(0)
        
        print("Starting real-time emotion-based music recommender...")
        print("Press 'q' to quit, 's' to save model")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip frame horizontally for selfie view
            frame = cv2.flip(frame, 1)
            
            # Extract landmarks
            landmarks_array, landmarks_obj = self.extract_facial_landmarks(frame)
            
            if landmarks_array is not None:
                # Draw landmarks on frame
                if landmarks_obj:
                    self.mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=landmarks_obj,
                        connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0,255,0), thickness=1)
                    )
                
                # Extract emotion features
                features = self.extract_emotion_features(landmarks_array)
                
                if features is not None:
                    if training_mode:
                        # Training mode
                        cv2.putText(frame, "TRAINING MODE", (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.putText(frame, f"Samples: {len(training_data)}", (10, 60), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(frame, "Press 'c' to collect, 't' to train", (10, 90), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    else:
                        # Prediction mode
                        emotion = self.predict_emotion(features)
                        if emotion is not None:
                            smoothed_emotion = self.smooth_emotion_prediction(emotion)
                            emotion_name = self.emotion_music_map.get(smoothed_emotion, 'unknown')
                            
                            # Display emotion
                            cv2.putText(frame, f"Emotion: {emotion_name}", (10, 30), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            
                            # Play music if available
                            if music_loaded:
                                current_song = self.play_music_for_emotion(smoothed_emotion)
                                if current_song:
                                    song_name = os.path.basename(current_song)
                                    cv2.putText(frame, f"Playing: {song_name[:30]}", (10, 70), 
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            else:
                cv2.putText(frame, "No face detected", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Display frame
            cv2.imshow('Emotion-Based Music Recommender', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('c') and training_mode and landmarks_array is not None:
                # Collect training sample
                features = self.extract_emotion_features(landmarks_array)
                if features is not None:
                    training_data.append(features)
                    print(f"Sample collected: {len(training_data)}")
            elif key == ord('t') and training_mode:
                # Train model
                if len(training_data) >= self.n_clusters:
                    success = self.train_emotion_model(training_data)
                    if success:
                        training_mode = False
                        self.save_model()
                        print("Switched to prediction mode")
                else:
                    print(f"Need at least {self.n_clusters} samples, currently have {len(training_data)}")
            elif key == ord('s'):
                # Save model
                self.save_model()
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        pygame.mixer.music.stop()
        pygame.mixer.quit()

if __name__ == "__main__":
    recommender = EmotionMusicRecommender()
    recommender.run_realtime()
