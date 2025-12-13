import cv2
import numpy as np
import tensorflow as tf

class QualityInspector:
    def __init__(self, model_path='model.tflite'):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
    
    def inspect(self, image):
        img = cv2.resize(image, (224, 224))
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        self.interpreter.set_tensor(self.input_details[0]['index'], img)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        return "DEFECT" if np.argmax(output) == 1 else "OK"

if __name__ == '__main__':
    inspector = QualityInspector()
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        result = inspector.inspect(frame)
        cv2.putText(frame, result, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Quality Control', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
