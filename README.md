# Emotion-Music-Recommender 🎵😊

Real-time emotion-based music recommendation system using facial recognition, K-Means clustering, and PCA for emotion detection.

## 🌟 Project Overview

This project implements an intelligent music recommendation system that detects your emotions through facial expressions in real-time using your webcam and plays music that matches your current mood. It uses machine learning techniques including PCA (Principal Component Analysis) for feature extraction and K-Means clustering for emotion classification.

## 🎯 Features

- **Real-time Facial Emotion Detection**: Uses MediaPipe for accurate facial landmark detection
- **Machine Learning**: Implements PCA and K-Means clustering for emotion classification
- **5 Emotion Categories**: Happy, Sad, Energetic, Calm, and Neutral
- **Music Recommendation**: Automatically plays music based on detected emotion
- **Smooth Emotion Tracking**: Uses buffering to avoid rapid emotion changes
- **Training Mode**: Collect your own facial expression data to personalize the model
- **Model Persistence**: Save and load trained models for future use

## 📋 Requirements

### Hardware
- Webcam
- Speakers/Headphones

### Software Dependencies

Install all required packages using:

```bash
pip install -r requirements.txt
```

Core dependencies:
- opencv-python >= 4.8.0
- numpy >= 1.24.0
- mediapipe >= 0.10.0
- scikit-learn >= 1.3.0
- pygame >= 2.5.0

## 🚀 Installation

1. **Clone the repository**
```bash
git clone https://github.com/abhay6205/Emotion-Music-Recommender.git
cd Emotion-Music-Recommender
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Create music directory structure**
```bash
mkdir -p music/{happy,sad,energetic,calm,neutral}
```

4. **Add your music files**
   - Place music files (.mp3 or .wav) in the corresponding emotion folders:
     - `music/happy/` - Upbeat, cheerful songs
     - `music/sad/` - Melancholic, slow songs
     - `music/energetic/` - Fast-paced, motivational songs
     - `music/calm/` - Relaxing, peaceful songs
     - `music/neutral/` - General background music

## 📖 Usage

### Basic Usage

Run the main script:

```bash
python emotion_music_recommender.py
```

### First Time Setup (Training Mode)

If no pre-trained model exists, the system will start in training mode:

1. **Collect Samples**: Press 'c' to capture facial expression samples
   - Make different facial expressions (smile, frown, neutral, etc.)
   - Collect at least 5 samples (more is better)

2. **Train Model**: Press 't' to train the emotion detection model
   - The system will use PCA and K-Means to cluster your expressions
   - Model will be automatically saved

3. **Start Detection**: After training, the system switches to prediction mode automatically

### Prediction Mode

Once a model is trained:
- The system detects your facial expressions in real-time
- Displays the detected emotion on screen
- Plays music matching your emotion
- Press 's' to save the current model
- Press 'q' to quit

## 🧠 How It Works

### 1. Facial Landmark Detection
- Uses MediaPipe Face Mesh to detect 478 facial landmarks
- Extracts 3D coordinates (x, y, z) for each landmark

### 2. Feature Extraction
- Calculates emotion-specific features:
  - Eye aspect ratios (openness)
  - Mouth aspect ratios (smile/frown)
  - Eyebrow positions (raised/lowered)
- Generates an 8-dimensional feature vector

### 3. Dimensionality Reduction (PCA)
- Reduces feature dimensions while preserving variance
- Helps remove noise and improve clustering

### 4. Emotion Clustering (K-Means)
- Groups similar facial expressions into 5 clusters
- Maps clusters to emotion categories

### 5. Music Recommendation
- Matches detected emotion to music playlist
- Plays appropriate music using pygame mixer
- Smooths emotion predictions to avoid song skipping

## 📁 Project Structure

```
Emotion-Music-Recommender/
├── emotion_music_recommender.py  # Main application script
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
├── .gitignore                     # Git ignore file
├── emotion_model.pkl              # Trained model (generated)
└── music/                         # Music library
    ├── happy/
    ├── sad/
    ├── energetic/
    ├── calm/
    └── neutral/
```

## 🎓 Machine Learning Concepts

### PCA (Principal Component Analysis)
- Unsupervised learning technique
- Reduces dimensionality of facial features
- Retains most important variance in data

### K-Means Clustering
- Unsupervised clustering algorithm
- Groups similar emotion features together
- Creates 5 distinct emotion clusters

### Why Unsupervised Learning?
- No need for labeled emotion data
- Learns patterns from your facial expressions
- Personalizes to your unique expressions

## 🔧 Configuration

You can modify these parameters in the code:

```python
# Number of emotion clusters
self.n_clusters = 5

# Emotion mapping
self.emotion_music_map = {
    0: 'happy',
    1: 'sad',
    2: 'energetic',
    3: 'calm',
    4: 'neutral'
}

# Smoothing buffer size
self.emotion_buffer = deque(maxlen=10)
```

## 🐛 Troubleshooting

### Webcam not working
- Check webcam permissions
- Try changing camera index: `cv2.VideoCapture(1)` or `cv2.VideoCapture(2)`

### No face detected
- Ensure good lighting
- Face the camera directly
- Check if face is within frame

### Music not playing
- Verify music files are in correct folders
- Check file formats (.mp3 or .wav)
- Ensure pygame mixer is properly initialized

### Model accuracy issues
- Collect more training samples
- Make distinct facial expressions for different emotions
- Retrain model with better samples

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests

## 📝 License

This project is open source and available for educational purposes.

## 👨‍💻 Author

**Abhay**
- GitHub: [@abhay6205](https://github.com/abhay6205)

## 🙏 Acknowledgments

- MediaPipe for facial landmark detection
- OpenCV for computer vision
- scikit-learn for machine learning algorithms
- pygame for audio playback

## 📚 Future Enhancements

- [ ] Deep learning-based emotion recognition
- [ ] Spotify API integration
- [ ] Multi-face emotion detection
- [ ] Voice-based emotion detection
- [ ] Web interface
- [ ] Mobile app version
- [ ] Cloud-based model training

---

**Enjoy your personalized emotion-based music experience! 🎵😊**
