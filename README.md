# Animal Counting using YOLOv8 (Python)

## 📌 Project Overview
This project focuses on **Animal Detection and Counting** using **YOLOv8**.  
The system detects and counts animals in video streams using a trained custom YOLOv8 model.

## 🚀 Features
- Object detection using **YOLOv8**
- Custom dataset annotation using **Roboflow**
- Training on **Google Colab**
- Real-time detection and counting using OpenCV
- Supports `.mp4` video input

---

## 🛠 Tech Stack
- **Python 3.8+**
- **YOLOv8 (Ultralytics)**
- **OpenCV**
- **Pandas**
- **cvzone**
- **Roboflow** (for dataset annotation)
- **Google Colab** (for training)

---

## 📂 Project Structure
├── best.pt # Trained YOLOv8 model
├── cow2.mp4 # Sample video for testing
├── coco1.txt # Class names file
├── tracker.py # Object tracking helper
├── main.py # Main script (detection & counting)
└── README.md # Project documentation
