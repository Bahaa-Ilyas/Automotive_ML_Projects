"""=============================================================================
PROJECT 28: CONVERSATIONAL AI - TRAINING SCRIPT
=============================================================================

DIFFICULTY: ★★★★☆ (Advanced)
NLP MODEL: DialoGPT (117M-345M parameters)

PURPOSE:
Build conversational AI chatbot using DialoGPT.
Generates human-like responses in multi-turn conversations.

TECHNIQUE: DialoGPT
- GPT-2 architecture trained on Reddit conversations
- Maintains conversation context
- Generates coherent responses
- Multiple model sizes available

USE CASES:
- Customer service chatbots
- Virtual assistants
- Interactive help systems
- Conversational interfaces

WHY DialoGPT?
- Trained on conversational data
- Context-aware responses
- Natural dialogue flow
- Easy to fine-tune
=============================================================================
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

print("\n" + "="*80)
print("PROJECT 28: CONVERSATIONAL AI")
print("DialoGPT Model (117M parameters)")
print("="*80)

print("\n[1/5] Loading DialoGPT...")
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
print(f"   ✓ Model loaded: {model.num_parameters():,} parameters")

print("\n[2/5] Testing conversational AI...")

conversations = [
    ["Hello! How can I help you today?"],
    ["I need help with my car's battery.", "What seems to be the problem with your battery?"],
    ["The car won't start.", "Have you checked if the battery terminals are clean and tight?"],
]

print("\n   Conversation Examples:")
for conv in conversations:
    chat_history_ids = None
    
    for i, user_input in enumerate(conv):
        print(f"\n   {'User' if i%2==0 else 'Bot'}: {user_input}")
        
        if i % 2 == 0:  # User input
            new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
            
            bot_input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1) if chat_history_ids is not None else new_input_ids
            
            chat_history_ids = model.generate(
                bot_input_ids,
                max_length=1000,
                pad_token_id=tokenizer.eos_token_id,
                no_repeat_ngram_size=3,
                do_sample=True,
                top_k=100,
                top_p=0.7,
                temperature=0.8
            )
            
            response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
            print(f"   Bot: {response}")

print("\n[3/5] Multi-turn conversation demo...")
print("\n   Starting interactive conversation:")

chat_history_ids = None
user_inputs = [
    "What is an electric vehicle?",
    "How far can they drive?",
    "Are they expensive?"
]

for user_input in user_inputs:
    print(f"\n   User: {user_input}")
    
    new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
    bot_input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1) if chat_history_ids is not None else new_input_ids
    
    chat_history_ids = model.generate(
        bot_input_ids,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=3,
        do_sample=True,
        top_k=100,
        top_p=0.7,
        temperature=0.8
    )
    
    response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    print(f"   Bot: {response}")

print("\n[4/5] Fine-tuning considerations...")
print("   ✓ Can fine-tune on domain-specific conversations")
print("   ✓ Requires conversation dataset")
print("   ✓ Improves relevance for specific use cases")

print("\n[5/5] Saving model...")
model.save_pretrained('./conversational_model')
tokenizer.save_pretrained('./conversational_model')
print("   ✓ Saved")

print("\n" + "="*80)
print("COMPLETE!")
print("="*80)
print("\nCapabilities:")
print("  • Multi-turn conversations")
print("  • Context-aware responses")
print("  • Natural dialogue flow")
print("\nNext Level: Project 29 - Document Understanding")
print("="*80 + "\n")
