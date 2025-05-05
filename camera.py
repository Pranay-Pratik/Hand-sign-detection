import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import tensorflow as tf
import time

class VideoCamera:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.detector = HandDetector(maxHands=1)
        self.active = False
        self.frame_count = 0

        self.interpreter = tf.lite.Interpreter(model_path="Model/model_unquant.tflite")
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        with open("Model/labels.txt", "r") as f:
            self.labels = [line.strip() for line in f.readlines()]

        self.offset = 20
        self.imgSize = 300
        self.prediction_text = "No sign detected"

        # 🔠 Prediction state
        self.prediction_history = ""
        self.last_prediction = None
        self.last_recorded_label = None
        self.stable_count = 0
        self.stability_threshold = 3
        self.last_prediction_time = 0
        self.prediction_delay = 0.5

    def set_active(self, active: bool):
        self.active = active

    def get_frame(self):
        if not self.active:
            return None

        success, img = self.cap.read()
        if not success:
            return None

        img = cv2.flip(img, 1)
        hands, img = self.detector.findHands(img)
        self.frame_count += 1

        if hands and self.frame_count % 5 == 0:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            imgWhite = np.ones((self.imgSize, self.imgSize, 3), np.uint8) * 255
            y1, y2 = max(0, y - self.offset), min(img.shape[0], y + h + self.offset)
            x1, x2 = max(0, x - self.offset), min(img.shape[1], x + w + self.offset)

            imgCrop = img[y1:y2, x1:x2]
            if imgCrop.size > 0:
                aspectRatio = h / w

                if aspectRatio > 1:
                    newHeight = self.imgSize
                    newWidth = math.ceil((w * self.imgSize) / h)
                    imgResize = cv2.resize(imgCrop, (newWidth, newHeight))
                    wGap = (self.imgSize - newWidth) // 2
                    imgWhite[:, wGap:wGap + newWidth] = imgResize
                else:
                    newWidth = self.imgSize
                    newHeight = math.ceil((h * self.imgSize) / w)
                    imgResize = cv2.resize(imgCrop, (newWidth, newHeight))
                    hGap = (self.imgSize - newHeight) // 2
                    imgWhite[hGap:hGap + newHeight, :] = imgResize

                index, confidence = self.predict(imgWhite)
                label = self.labels[index]
                self.prediction_text = f"{label} ({confidence:.2f})"

                # ✨ Stable prediction logic with safe repeat support
                current_time = time.time()

                if self.last_prediction == label:
                    self.stable_count += 1

                    if self.stable_count >= self.stability_threshold:
                        if (label != self.last_recorded_label or
                            (current_time - self.last_prediction_time) > self.prediction_delay * 2):
                            self.prediction_history += label
                            self.last_recorded_label = label
                            self.last_prediction_time = current_time
                        self.stable_count = 0
                else:
                    self.last_prediction = label
                    self.stable_count = 1

                cv2.putText(img, f"Detected: {label}", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 2)

        ret, jpeg = cv2.imencode('.jpg', img)
        return jpeg.tobytes()

    def predict(self, imgWhite):
        input_shape = self.input_details[0]['shape'][1:3]
        imgInput = cv2.resize(imgWhite, tuple(input_shape))
        imgInput = np.expand_dims(imgInput, axis=0).astype(np.float32)
        imgInput = imgInput / 255.0

        self.interpreter.set_tensor(self.input_details[0]['index'], imgInput)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        index = np.argmax(output_data)
        return index, output_data[0][index]

    def get_prediction_text(self):
        return getattr(self, "prediction_text", "No prediction yet")

    def get_prediction_history(self):
        return self.prediction_history

    def reset_prediction_history(self):
        self.prediction_history = ""
        self.last_prediction = None
        self.last_recorded_label = None
        self.stable_count = 0
