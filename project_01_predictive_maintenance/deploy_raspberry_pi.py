import numpy as np
import tensorflow as tf
import time

class EdgePredictor:
    def __init__(self, model_path='model.tflite'):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
    
    def predict(self, sensor_value):
        input_data = np.array([[[[sensor_value]]]], dtype=np.float32)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        return output[0][0]

if __name__ == '__main__':
    predictor = EdgePredictor()
    while True:
        sensor_reading = np.random.normal(50, 10)  # Replace with actual sensor
        prediction = predictor.predict(sensor_reading)
        status = "ALERT" if prediction > 0.5 else "NORMAL"
        print(f"Sensor: {sensor_reading:.2f} | Prediction: {prediction:.4f} | Status: {status}")
        time.sleep(1)
