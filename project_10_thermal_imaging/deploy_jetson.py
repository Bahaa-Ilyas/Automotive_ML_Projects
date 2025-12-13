import cv2
import numpy as np
import tensorflow as tf

class ThermalAnalyzer:
    def __init__(self, model_path='model.tflite'):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.classes = ['Normal', 'Hot Spot', 'Cold Spot']
    
    def analyze(self, thermal_image):
        img = cv2.resize(thermal_image, (128, 128))
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=(0, -1))
        
        self.interpreter.set_tensor(self.input_details[0]['index'], img)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        class_idx = np.argmax(output)
        confidence = output[0][class_idx]
        return self.classes[class_idx], confidence

if __name__ == '__main__':
    analyzer = ThermalAnalyzer()
    cap = cv2.VideoCapture(0)  # Replace with thermal camera
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        result, conf = analyzer.analyze(frame)
        text = f"{result}: {conf:.2f}"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Thermal Analysis', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
