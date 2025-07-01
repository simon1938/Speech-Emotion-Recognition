# Speech Emotion Recognition with Neural Networks

This project explores the recognition of emotions in speech using deep learning techniques, particularly CNN, LSTM, and Transformer-based models. The main goal is to classify emotional states expressed in audio-only speech recordings using spectrograms and state-of-the-art neural architectures.

## 🔗 Repository
[GitHub Repository](https://github.com/aubinbnf/Speech-Emotion-Recognition)

## 📄 Abstract

We implement and compare various neural architectures for Speech Emotion Recognition (SER), including CNNs, LSTMs, CNN-LSTM hybrids, and Transformers. Spectrograms are used as the primary input representation, and experiments are conducted using multiple publicly available datasets.

## 📁 Datasets

We utilize a combination of popular datasets:
- RAVDESS
- SAVEE
- CREMA-D
- TESS
- ESD
- JL Corpus

> Total: 35,000+ audio samples across 7 emotional classes.

Emotions:
- Angry
- Happy
- Sad
- Neutral
- Fearful
- Disgust
- Surprise

## 🧪 Experimental Setup

### Preprocessing
- Silence removal
- Resampling to 16kHz
- Padding/trimming to 3 seconds
- Conversion to spectrograms

### Data Augmentation
- Audio-level: time stretching, pitch shifting, noise injection
- Spectrogram-level: time shifting, Gaussian noise

### Evaluation Metrics
- Accuracy
- F1-score
- Confusion matrix

## 🧠 Models

### CNN-BLSTM with Attention (Best model)
- Spectrogram input (128x128x3)
- CNN layers for spatial feature extraction
- Bidirectional LSTM for temporal modeling
- Attention mechanism for feature weighting
- Achieved **76% accuracy**, **F1-score: 0.753**

### VGG19 Transfer Learning
- Pre-trained on ImageNet
- Custom dense layers for classification
- Achieved **76% accuracy**

### Other architectures tested
- CNN
- LSTM
- CNN-LSTM
- ResNet50
- MobileNetV3Large

## 📊 Results

| Model                      | Accuracy | F1-Score |
|---------------------------|----------|----------|
| CNN-BLSTM + Attention     | 76%      | 0.753    |
| VGG19                     | 76%      | 0.77     |
| ResNet50                  | 65%      | 0.65      |


## 🔍 Insights

- Attention mechanisms improve recognition, especially for subtle emotions.
- Fearful and Disgust remain hard to classify due to acoustic similarities.
- Oversampling and augmentation significantly improve generalization.

## 🔧 Requirements

- Python 3.8+
- TensorFlow / Keras
- Librosa
- NumPy
- Matplotlib
- Scikit-learn

## 🚀 Run the Project

```bash
git clone https://github.com/aubinbnf/Speech-Emotion-Recognition.git
cd Speech-Emotion-Recognition
pip install -r requirements.txt
python train.py



## 📈 Future Work

- Explore Transformer-only architectures  
- Use self-supervised learning for feature extraction  
- Incorporate more diverse and multilingual datasets  
- Improve misclassification in overlapping emotions (e.g., *happy* vs. *neutral*)

## 👥 Authors

- **Aubin Bonnefoy** – 
- **Simon Illouz-Laurent** –

---

**License**: MIT  
