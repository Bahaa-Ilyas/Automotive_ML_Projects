import os
import sys
import numpy as np
import tensorflow as tf
from pathlib import Path

sys.path.append('shared_utils')
from data_generator import SyntheticDataGenerator

class ModelTester:
    def __init__(self):
        self.results = {}
    
    def test_tflite_model(self, model_path, input_data):
        try:
            interpreter = tf.lite.Interpreter(model_path=model_path)
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])
            
            model_size = Path(model_path).stat().st_size / 1024
            return True, model_size, output.shape
        except Exception as e:
            return False, 0, str(e)
    
    def run_tests(self):
        print("=" * 70)
        print("TESTING ALL MODELS")
        print("=" * 70)
        
        tests = [
            {
                'name': 'Predictive Maintenance',
                'path': 'project_01_predictive_maintenance/model.tflite',
                'input': np.random.rand(1, 1, 1).astype(np.float32)
            },
            {
                'name': 'Quality Control',
                'path': 'project_02_quality_control/model.tflite',
                'input': np.random.rand(1, 224, 224, 3).astype(np.float32)
            },
            {
                'name': 'Energy Forecasting',
                'path': 'project_03_energy_forecasting/model.tflite',
                'input': np.random.rand(1, 24, 1).astype(np.float32)
            },
            {
                'name': 'Structural Health',
                'path': 'project_05_structural_health/model.tflite',
                'input': np.random.rand(1, 10).astype(np.float32)
            },
            {
                'name': 'Traffic Flow',
                'path': 'project_06_traffic_flow/model.tflite',
                'input': np.random.rand(1, 12, 4).astype(np.float32)
            },
            {
                'name': 'Document Classification',
                'path': 'project_07_document_classification/model.tflite',
                'input': np.random.randint(0, 5000, (1, 100)).astype(np.float32)
            },
            {
                'name': 'Vibration Analysis',
                'path': 'project_08_vibration_analysis/model.tflite',
                'input': np.random.rand(1, 128, 1).astype(np.float32)
            },
            {
                'name': 'Thermal Imaging',
                'path': 'project_10_thermal_imaging/model.tflite',
                'input': np.random.rand(1, 128, 128, 1).astype(np.float32)
            }
        ]
        
        for i, test in enumerate(tests, 1):
            print(f"\n[{i}/{len(tests)}] Testing {test['name']}...")
            print("-" * 70)
            
            if os.path.exists(test['path']):
                success, size, output = self.test_tflite_model(test['path'], test['input'])
                if success:
                    print(f"✓ Model loaded successfully")
                    print(f"  Size: {size:.2f} KB")
                    print(f"  Output shape: {output}")
                    self.results[test['name']] = {'status': 'PASS', 'size': size}
                else:
                    print(f"✗ Error: {output}")
                    self.results[test['name']] = {'status': 'FAIL', 'error': output}
            else:
                print(f"✗ Model not found (run training first)")
                self.results[test['name']] = {'status': 'NOT_FOUND'}
        
        self.print_summary()
    
    def print_summary(self):
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        
        passed = sum(1 for r in self.results.values() if r['status'] == 'PASS')
        failed = sum(1 for r in self.results.values() if r['status'] == 'FAIL')
        not_found = sum(1 for r in self.results.values() if r['status'] == 'NOT_FOUND')
        
        print(f"\nTotal: {len(self.results)}")
        print(f"✓ Passed: {passed}")
        print(f"✗ Failed: {failed}")
        print(f"⊘ Not Found: {not_found}")
        
        if passed > 0:
            total_size = sum(r['size'] for r in self.results.values() if r['status'] == 'PASS')
            print(f"\nTotal model size: {total_size:.2f} KB ({total_size/1024:.2f} MB)")
        
        print("\n" + "=" * 70)

if __name__ == '__main__':
    tester = ModelTester()
    tester.run_tests()
