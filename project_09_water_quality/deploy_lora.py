# LoRa deployment for remote water quality monitoring
import time
import random

class WaterQualityMonitor:
    def __init__(self):
        self.thresholds = {'ph': (6.5, 8.5), 'turbidity': (0, 5), 'do': (6, 14)}
    
    def read_sensors(self):
        return {
            'ph': random.uniform(6, 9),
            'turbidity': random.uniform(0, 10),
            'dissolved_oxygen': random.uniform(4, 10),
            'conductivity': random.uniform(200, 800),
            'temperature': random.uniform(10, 25)
        }
    
    def assess_quality(self, readings):
        score = 0
        if self.thresholds['ph'][0] <= readings['ph'] <= self.thresholds['ph'][1]:
            score += 1
        if readings['turbidity'] < self.thresholds['turbidity'][1]:
            score += 1
        if readings['dissolved_oxygen'] >= self.thresholds['do'][0]:
            score += 1
        return "GOOD" if score >= 2 else "POOR"
    
    def send_lora(self, data):
        # Simulate LoRa transmission
        print(f"LoRa TX: {data}")

if __name__ == '__main__':
    monitor = WaterQualityMonitor()
    while True:
        readings = monitor.read_sensors()
        quality = monitor.assess_quality(readings)
        message = f"pH:{readings['ph']:.1f}|Turb:{readings['turbidity']:.1f}|DO:{readings['dissolved_oxygen']:.1f}|Q:{quality}"
        monitor.send_lora(message)
        time.sleep(300)  # Every 5 minutes
