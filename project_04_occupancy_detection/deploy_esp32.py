# ESP32 MicroPython deployment
# Upload this to ESP32 with sensor array

import machine
import time
import math

# Simplified decision tree (converted from trained model)
def predict_occupancy(temp, humidity, light, co2, sound):
    if light > 250:
        if co2 > 500:
            if sound > 25:
                return 1  # Occupied
    if temp > 21 and humidity > 35:
        return 1
    return 0  # Vacant

# Sensor setup (example pins)
adc_temp = machine.ADC(machine.Pin(34))
adc_light = machine.ADC(machine.Pin(35))

while True:
    temp = adc_temp.read() / 40.96  # Convert to temperature
    humidity = 45  # From DHT22 sensor
    light = adc_light.read()
    co2 = 450  # From MQ-135 sensor
    sound = 20  # From sound sensor
    
    occupied = predict_occupancy(temp, humidity, light, co2, sound)
    print(f"Status: {'OCCUPIED' if occupied else 'VACANT'}")
    time.sleep(5)
