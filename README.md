# 💓 Heart Rate Monitor using Webcam  

## 📌 Overview  
This project enables **contactless heart rate measurement** using **remote photoplethysmography (rPPG)**. By analyzing **subtle changes in facial skin tone** captured by a webcam, it estimates heart rate in real-time—**ideal for telemedicine, fitness tracking, and driver monitoring.**  

📢 **Why This Matters?**  
- **No need for wearables** – Just use your webcam!  
- **Non-contact & hygienic** – Perfect for **telehealth & remote patient monitoring.**  
- **AI-powered face tracking** ensures accurate pulse detection.  

---

## 🚀 Features  
✅ **Webcam & Video Support** – Monitor heart rate in real-time.  
✅ **Face & ROI Detection** – Identifies forehead and cheeks for pulse extraction.  
✅ **Signal & FFT Analysis** – Processes heart rate frequency using Fast Fourier Transform (FFT).  
✅ **Real-Time Graphs** – Displays raw signals and frequency spectrum.  
✅ **User-Friendly GUI** – PyQt5-based graphical interface for easy usage.  
✅ **Optimized for Motion & Lighting** – Works in varying lighting conditions with motion compensation.  

---

## 🛠️ Technologies Used  
- **Python** – Core programming language  
- **OpenCV** – Face detection & image processing  
- **Dlib** – Facial landmark detection  
- **NumPy & SciPy** – Signal processing  
- **PyQt5** – GUI framework  
- **PyQtGraph** – Real-time graph plotting  

---

## 🖥️ System Requirements  
- OS: **Windows 10/11**, **macOS**, **Linux**  
- Python **3.8+**  
- Camera (Webcam) or Video File  

---

## 👥 Team Members  
- **Krishna Teja**  
- **Gnanateja**  
- **Vyshnavi**  
- **Jayavardhan Kumar Reddy**
---

## 🏭 Industry Applications  
### 📌 Where Can This Be Used?  
- 🏥 **Healthcare & Telemedicine** – Contactless heart rate monitoring in hospitals.  
- 🏋️ **Fitness & Sports** – Monitoring heart rate during workouts.  
- 🚘 **Automotive** – Driver fatigue monitoring for road safety.  
- 🛡️ **Security & Surveillance** – Stress detection in high-security environments.  
- 📱 **Wearable Tech** – Could be integrated with **smartphone cameras** for health tracking.  

---
## 🖥️ Installation
```
# Clone the repository
git clone https://github.com/jayavardhankumar/heart-rate-monitor.git

# Navigate into the project directory
cd heart-rate-monitor

# Install dependencies
pip install -r requirements.txt

```

##▶️ How to Run

Run the following command to start the application:
```
python gui.py
```
##🛠️ Troubleshooting
❗ Common Issues & Fixes
Dlib installation error? Install CMake before installing dlib:

```
pip install cmake
pip install dlib
```
Webcam not opening?

Check if another app is using the webcam.

Try python webcam.py to test the camera.

##📌 Future Improvements
🚀 What's Next?

✅ Deep Learning Enhancement – Improve accuracy with neural networks.

✅ Infrared Camera Support – Work in low-light conditions.

✅ Cloud Integration – Store and analyze heart rate trends over time.

