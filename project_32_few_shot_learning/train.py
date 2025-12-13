"""=============================================================================
PROJECT 32: FEW-SHOT LEARNING NLP - TRAINING SCRIPT
=============================================================================

DIFFICULTY: ★★★★★ (Advanced)
NLP MODEL: GPT-3.5 / GPT-4 API (175B+ parameters)

PURPOSE:
Perform NLP tasks with minimal examples using few-shot learning.
Leverages large language models' in-context learning abilities.

TECHNIQUE: Few-Shot Prompting
- Provide examples in prompt
- No fine-tuning required
- Adapts to new tasks instantly
- Uses massive pre-trained models

USE CASES:
- Rapid prototyping
- Low-data scenarios
- Custom classification
- Domain-specific tasks

WHY FEW-SHOT?
- No training data needed
- Instant adaptation
- Flexible task definition
- State-of-the-art performance
=============================================================================
"""

import openai
import os

print("\n" + "="*80)
print("PROJECT 32: FEW-SHOT LEARNING NLP")
print("GPT-3.5/4 Few-Shot Prompting (175B+ parameters)")
print("="*80)

print("\n[1/5] Setting up OpenAI API...")
print("   Note: Requires OpenAI API key")
print("   ✓ Using GPT-3.5-turbo (cost-effective)")
print("   ✓ Alternative: GPT-4 (higher quality)")

# Simulated API (replace with actual API key)
USE_REAL_API = False

if USE_REAL_API:
    openai.api_key = os.getenv("OPENAI_API_KEY")
else:
    print("   ⚠ Demo mode: Simulating API responses")

print("\n[2/5] Few-shot automotive classification...")

# Few-shot prompt with examples
few_shot_prompt = """
Classify automotive diagnostic messages into categories: ENGINE, BATTERY, BRAKES, TRANSMISSION, or OTHER.

Examples:
Message: "Engine misfire detected in cylinder 3"
Category: ENGINE

Message: "Battery voltage below 12V"
Category: BATTERY

Message: "ABS sensor malfunction front left"
Category: BRAKES

Message: "Transmission fluid temperature high"
Category: TRANSMISSION

Now classify:
Message: "{message}"
Category:"""

test_messages = [
    "Check engine light on, rough idle",
    "Battery drains overnight",
    "Brake pedal feels soft",
    "Gear shifting delayed",
    "Tire pressure low"
]

print("\n   Few-Shot Classification Results:")
for msg in test_messages:
    prompt = few_shot_prompt.format(message=msg)
    
    if USE_REAL_API:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0
        )
        category = response.choices[0].message.content.strip()
    else:
        # Simulated response
        if "engine" in msg.lower():
            category = "ENGINE"
        elif "battery" in msg.lower():
            category = "BATTERY"
        elif "brake" in msg.lower():
            category = "BRAKES"
        elif "gear" in msg.lower() or "shifting" in msg.lower():
            category = "TRANSMISSION"
        else:
            category = "OTHER"
    
    print(f"\n   Message: {msg}")
    print(f"   Category: {category}")

print("\n[3/5] Few-shot entity extraction...")

extraction_prompt = """
Extract key information from automotive service requests.

Examples:
Request: "2019 BMW X5 needs oil change and tire rotation"
Vehicle: 2019 BMW X5
Services: oil change, tire rotation

Request: "Tesla Model 3 battery check and software update"
Vehicle: Tesla Model 3
Services: battery check, software update

Now extract:
Request: "{request}"
Vehicle:
Services:"""

test_requests = [
    "2021 Mercedes E-Class brake inspection and alignment",
    "Ford F-150 engine diagnostic and filter replacement"
]

print("\n   Few-Shot Entity Extraction:")
for req in test_requests:
    print(f"\n   Request: {req}")
    
    if USE_REAL_API:
        prompt = extraction_prompt.format(request=req)
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
            temperature=0
        )
        result = response.choices[0].message.content.strip()
        print(f"   {result}")
    else:
        # Simulated extraction
        words = req.split()
        vehicle = " ".join(words[:3])
        services = " ".join(words[3:])
        print(f"   Vehicle: {vehicle}")
        print(f"   Services: {services}")

print("\n[4/5] Zero-shot vs Few-shot comparison...")
print("\n   Zero-Shot (no examples):")
print("      • Model uses general knowledge")
print("      • May not understand domain specifics")
print("      • Accuracy: 60-70%")
print("\n   Few-Shot (with examples):")
print("      • Model learns from examples")
print("      • Adapts to domain and format")
print("      • Accuracy: 80-95%")

print("\n[5/5] Best practices...")
print("\n   Few-Shot Prompting Tips:")
print("      1. Provide 3-5 diverse examples")
print("      2. Use consistent formatting")
print("      3. Include edge cases")
print("      4. Set temperature=0 for consistency")
print("      5. Validate outputs")

print("\n" + "="*80)
print("COMPLETE!")
print("="*80)
print("\nFew-Shot Learning Advantages:")
print("  • No training data collection")
print("  • Instant task adaptation")
print("  • Flexible task definition")
print("  • State-of-the-art performance")
print("\nLimitations:")
print("  • API costs per request")
print("  • Requires internet connection")
print("  • Limited context window")
print("  • Less control than fine-tuning")
print("\nAutomotive Applications:")
print("  • Diagnostic message classification")
print("  • Service request parsing")
print("  • Technical document QA")
print("  • Customer inquiry routing")
print("\nNext Level: Project 33 - Automotive Voice Assistant (FINAL)")
print("="*80 + "\n")
