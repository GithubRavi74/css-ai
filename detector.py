import onnxruntime as ort
import numpy as np
import cv2

class PPEModel:
    def __init__(self, model_path="models/best.onnx"):
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name

    def preprocess(self, frame):
        img = cv2.resize(frame, (640, 640))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img

    def predict(self, frame):
        inp = self.preprocess(frame)
        return self.session.run(None, {self.input_name: inp})
