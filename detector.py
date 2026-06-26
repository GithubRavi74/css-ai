# detector.py
import onnxruntime as ort
import numpy as np
import cv2

class PPEModel:
    def __init__(self, model_path="models/best.onnx"):
        # Explicit CPU initialization for deployment stability
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        
        # Explicit model index dictionary mapping from training
        self.classes = [
            'Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 
            'Person', 'Safety Cone', 'Safety Vest', 'machinery', 'vehicle'
        ]
        self.violators = ["NO-Hardhat", "NO-Mask", "NO-Safety Vest"]

    def preprocess(self, frame):
        # YOLO ONNX expects scaled float32 arrays structured as [1, 3, 640, 640]
        img = cv2.resize(frame, (640, 640))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC to CHW
        img = np.expand_dims(img, axis=0)   # Add batch dimension
        return img

    #def process_frame(self, orig_frame, conf_threshold=0.40, nms_threshold=0.45):
      def process_frame(self, orig_frame, conf_threshold=0.38, nms_threshold=0.60):
        """
        Processes a single BGR frame, runs inference, draws boxes, and returns 
        a list of final text detections along with the processed annotated image.
        """
        # 1. Run inference
        inp = self.preprocess(orig_frame)
        outputs = self.session.run(None, {self.input_name: inp})
        
        # 2. Reshape multi-dimensional array matrix output
        output = np.squeeze(outputs[0])  # (1, 14, 8400) -> (14, 8400)
        output = output.T                # (14, 8400) -> (8400, 14)

        boxes = []
        confidences = []
        class_ids = []
        
        # 3. Filter boxes by a confidence threshold
        for row in output:
            scores = row[4:]  # The 10 class confidence scores
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > conf_threshold:
                xc, yc, w, h = row[0:4]
                
                # Convert center coordinates to top-left corner coordinates for OpenCV
                x1 = int((xc - w/2))
                y1 = int((yc - h/2))
                width = int(w)
                height = int(h)
                
                boxes.append([x1, y1, width, height])
                confidences.append(float(confidence))
                class_ids.append(class_id)

        # 4. Apply Non-Maximum Suppression to clear up over-lapping anchor boxes
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

        # 5. Coordinate Scaling & Annotation Rendering
        annotated_img = orig_frame.copy()
        orig_h, orig_w, _ = annotated_img.shape
        
        x_scale = orig_w / 640
        y_scale = orig_h / 640
        
        final_detections = []

        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                current_class = self.classes[class_ids[i]]
                final_detections.append(current_class)
                
                # Scale boxes back to matching input resolution metrics
                x = int(x * x_scale)
                y = int(y * y_scale)
                w = int(w * x_scale)
                h = int(h * y_scale)
                
                # Dynamic box coloring (Red for infractions, Green for assets/safe gear)
                box_color = (0, 0, 255) if current_class in self.violators else (0, 255, 0) # BGR Colors
                
                # Draw box
                cv2.rectangle(annotated_img, (x, y), (x + w, y + h), box_color, 3)
                
                # Render text tags
                label = f"{current_class}: {confidences[i]:.2f}"
                cv2.putText(annotated_img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return final_detections, annotated_img
