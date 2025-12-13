"""
Run all 43 ML/NLP/RAG training scripts
"""
import os
import subprocess
import time

# All 43 projects
projects = [
    # ML Projects (1-22)
    'project_01_predictive_maintenance',
    'project_02_quality_control',
    'project_03_energy_forecasting',
    'project_04_occupancy_detection',
    'project_05_structural_health',
    'project_06_traffic_flow',
    'project_07_document_classification',
    'project_08_vibration_analysis',
    'project_09_water_quality',
    'project_10_thermal_imaging',
    'project_11_facial_recognition',
    'project_12_speech_recognition',
    'project_13_anomaly_detection',
    'project_14_recommendation_system',
    'project_15_object_tracking',
    'project_16_sentiment_analysis',
    'project_17_pose_estimation',
    'project_18_time_series_clustering',
    'project_19_image_segmentation',
    'project_20_reinforcement_learning',
    'project_21_log_anomaly_detection',
    'project_22_llm_root_cause_analysis',
    # NLP Projects (23-33)
    'project_23_text_classification',
    'project_24_named_entity_recognition',
    'project_25_text_summarization',
    'project_26_question_answering',
    'project_27_machine_translation',
    'project_28_conversational_ai',
    'project_29_document_understanding',
    'project_30_multimodal_nlp',
    'project_31_domain_adaptation',
    'project_32_few_shot_learning',
    'project_33_automotive_voice_assistant',
    # RAG Projects (34-43)
    'project_34_basic_rag',
    'project_35_semantic_search',
    'project_36_document_qa',
    'project_37_hybrid_rag',
    'project_38_conversational_rag',
    'project_39_multimodal_rag',
    'project_40_agentic_rag',
    'project_41_enterprise_rag',
    'project_42_code_rag',
    'project_43_automotive_rag',
]

print("=" * 80)
print("TRAINING ALL 43 ML/NLP/RAG PROJECTS")
print("=" * 80)

success_count = 0
failed_count = 0

for i, project in enumerate(projects, 1):
    print(f"\n[{i}/43] Training {project}...")
    print("-" * 80)
    
    project_path = os.path.join(os.getcwd(), project)
    train_script = os.path.join(project_path, 'train.py')
    
    if os.path.exists(train_script):
        start_time = time.time()
        try:
            result = subprocess.run(['python', train_script], 
                                  cwd=project_path, 
                                  capture_output=True, 
                                  text=True,
                                  timeout=600)
            elapsed = time.time() - start_time
            
            if result.returncode == 0:
                print(f"✓ SUCCESS ({elapsed:.1f}s)")
                success_count += 1
            else:
                print(f"✗ FAILED")
                print(result.stderr[-200:] if result.stderr else "No error output")
                failed_count += 1
        except subprocess.TimeoutExpired:
            print(f"✗ TIMEOUT (>10 min)")
            failed_count += 1
        except Exception as e:
            print(f"✗ ERROR: {e}")
            failed_count += 1
    else:
        print(f"✗ train.py not found")
        failed_count += 1

print("\n" + "=" * 80)
print("TRAINING COMPLETE")
print("=" * 80)
print(f"\nResults:")
print(f"  ✓ Success: {success_count}/43")
print(f"  ✗ Failed:  {failed_count}/43")
print(f"  Success Rate: {success_count/43*100:.1f}%")
print("=" * 80)
