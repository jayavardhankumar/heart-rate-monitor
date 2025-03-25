import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
from process import Process
from webcam import Webcam
from video import Video

class GUI(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Heart Rate Monitor")
        self.setGeometry(100, 100, 1200, 700)

        self.process = Process()
        self.webcam = Webcam()
        self.video = Video()
        self.input_source = None

        self.initUI()

    def initUI(self):
        """Initialize GUI components."""
        font = QtGui.QFont()
        font.setPointSize(12)

        # **Webcam Preview (Left)**
        self.webcam_display = QtWidgets.QLabel(self)
        self.webcam_display.setGeometry(20, 20, 500, 375)
        self.webcam_display.setStyleSheet("background-color: black;")

        # **ROI Preview (Right)**
        self.roi_display = QtWidgets.QLabel(self)
        self.roi_display.setGeometry(540, 20, 300, 200)
        self.roi_display.setStyleSheet("background-color: black;")

        # **Heart Rate Display**
        self.heart_rate_label = QtWidgets.QLabel("Heart Rate: -- BPM", self)
        self.heart_rate_label.setGeometry(880, 40, 250, 50)
        self.heart_rate_label.setFont(font)

        # **Graph for Signal**
        self.signal_plot = pg.PlotWidget(self)
        self.signal_plot.setGeometry(540, 250, 600, 180)
        self.signal_plot.setLabel("bottom", "Signal")

        # **Graph for FFT**
        self.fft_plot = pg.PlotWidget(self)
        self.fft_plot.setGeometry(540, 450, 600, 180)
        self.fft_plot.setLabel("bottom", "FFT")

        # **Buttons**
        self.btn_webcam = QtWidgets.QPushButton("Start Webcam", self)
        self.btn_webcam.setGeometry(20, 420, 200, 50)
        self.btn_webcam.clicked.connect(self.start_webcam)

        self.btn_video = QtWidgets.QPushButton("Open Video", self)
        self.btn_video.setGeometry(250, 420, 200, 50)
        self.btn_video.clicked.connect(self.open_video)

        self.btn_stop = QtWidgets.QPushButton("Stop", self)
        self.btn_stop.setGeometry(20, 500, 200, 50)
        self.btn_stop.clicked.connect(self.stop_processing)

        # **Timer for Updating Frames**
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frame)

    def start_webcam(self):
        """Start capturing from the webcam."""
        self.input_source = self.webcam
        if self.process.start_webcam():
            self.timer.start(30)  # 30ms update interval

    def open_video(self):
        """Open a video file for processing."""
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open Video File", "", "Videos (*.mp4 *.avi)")
        if file_path:
            self.input_source = self.video
            if self.process.start_video(file_path):
                self.timer.start(30)

    def stop_processing(self):
        """Stop video processing."""
        self.timer.stop()
        self.process.stop()
        self.webcam_display.clear()
        self.roi_display.clear()
        self.signal_plot.clear()
        self.fft_plot.clear()
        self.heart_rate_label.setText("Heart Rate: -- BPM")

    def update_frame(self):
        """Update webcam and graphs in real-time."""
        frame = self.process.get_frame()
        if frame is None:
            return

        processed_frame, roi_frame, bpm, signal_data, freqs, fft_values = self.process.run(frame)

        # **Update Webcam Display**
        self.display_image(processed_frame, self.webcam_display)

        # **Update ROI Display**
        if roi_frame is not None:
            self.display_image(roi_frame, self.roi_display)

        # **Update Graphs**
        self.signal_plot.clear()
        self.fft_plot.clear()

        if len(signal_data) > 20:
            self.signal_plot.plot(signal_data, pen="g")
        if len(freqs) > 0 and len(fft_values) > 0:
            self.fft_plot.plot(freqs, fft_values, pen="r")

        # **Update Heart Rate Display**
        if bpm:
            self.heart_rate_label.setText(f"Heart Rate: {bpm:.2f} BPM")

    def display_image(self, frame, label):
        """Convert OpenCV frame to QPixmap and display."""
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_img = QtGui.QImage(frame.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
        label.setPixmap(QtGui.QPixmap.fromImage(q_img))

if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = GUI()
    window.show()
    app.exec_()
