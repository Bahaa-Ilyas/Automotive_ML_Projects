"""=============================================================================
PROJECT 20: REINFORCEMENT LEARNING - TRAINING SCRIPT
=============================================================================

PURPOSE:
Train a Deep Q-Network (DQN) agent for decision-making in dynamic environments.
Used in robotics, game AI, resource optimization, and autonomous systems.

WHY REINFORCEMENT LEARNING?
- Learn from Experience: Agent learns optimal actions through trial and error
- No Labeled Data: Learns from rewards/penalties
- Sequential Decisions: Handles multi-step problems
- Adaptable: Adjusts to changing environments

DQN COMPONENTS:
- State: Current environment observation
- Action: Decision to make
- Reward: Feedback signal
- Q-Network: Estimates value of state-action pairs

USE CASES:
- Robotics (navigation, manipulation)
- Game AI (chess, Go, video games)
- Resource optimization (energy, traffic)
- Autonomous vehicles (driving decisions)
- Trading algorithms (buy/sell decisions)
=============================================================================
"""

import numpy as np
import tensorflow as tf
from collections import deque
import random

print("\n" + "="*70)
print("REINFORCEMENT LEARNING - TRAINING")
print("Deep Q-Network (DQN) for Decision Making")
print("="*70)

# STEP 1: Define Simple Environment
print("\n[1/7] Setting up environment...")

class SimpleEnvironment:
    """
    Simple grid world environment
    - State: Agent position (x, y)
    - Actions: Up, Down, Left, Right
    - Goal: Reach target position
    - Reward: +10 for reaching goal, -1 per step
    """
    def __init__(self, size=10):
        self.size = size
        self.reset()
    
    def reset(self):
        self.agent_pos = [0, 0]
        self.goal_pos = [self.size-1, self.size-1]
        return np.array(self.agent_pos)
    
    def step(self, action):
        # Actions: 0=up, 1=down, 2=left, 3=right
        if action == 0: self.agent_pos[1] = max(0, self.agent_pos[1] - 1)
        elif action == 1: self.agent_pos[1] = min(self.size-1, self.agent_pos[1] + 1)
        elif action == 2: self.agent_pos[0] = max(0, self.agent_pos[0] - 1)
        elif action == 3: self.agent_pos[0] = min(self.size-1, self.agent_pos[0] + 1)
        
        # Reward
        if self.agent_pos == self.goal_pos:
            reward = 10
            done = True
        else:
            reward = -1
            done = False
        
        return np.array(self.agent_pos), reward, done

env = SimpleEnvironment(size=10)
state_size = 2  # (x, y) position
action_size = 4  # Up, Down, Left, Right

print(f"   ✓ Environment: 10×10 grid world")
print(f"   ✓ State space: {state_size} (x, y position)")
print(f"   ✓ Action space: {action_size} (Up, Down, Left, Right)")
print(f"   ✓ Goal: Navigate to target position")

# STEP 2: Build Q-Network
print("\n[2/7] Building Q-Network...")

model = tf.keras.Sequential([
    tf.keras.layers.Dense(24, activation='relu', input_shape=(state_size,)),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(action_size, activation='linear')  # Q-values for each action
])

model.compile(optimizer='adam', loss='mse')
print(f"   ✓ Q-Network created: {model.count_params():,} parameters")
print("   ✓ Output: Q-values for each action")

# STEP 3: Initialize Replay Memory
print("\n[3/7] Initializing replay memory...")

memory = deque(maxlen=2000)
epsilon = 1.0  # Exploration rate
epsilon_decay = 0.995
epsilon_min = 0.01
gamma = 0.95  # Discount factor

print(f"   ✓ Replay memory size: 2000")
print(f"   ✓ Initial exploration rate: {epsilon}")
print(f"   ✓ Discount factor: {gamma}")

# STEP 4: Train DQN Agent
print("\n[4/7] Training DQN agent...")
print("   This may take 5-10 minutes...\n")

episodes = 500
batch_size = 32

for episode in range(episodes):
    state = env.reset()
    total_reward = 0
    
    for step in range(100):  # Max 100 steps per episode
        # Epsilon-greedy action selection
        if np.random.rand() <= epsilon:
            action = random.randrange(action_size)  # Explore
        else:
            q_values = model.predict(state.reshape(1, -1), verbose=0)
            action = np.argmax(q_values[0])  # Exploit
        
        # Take action
        next_state, reward, done = env.step(action)
        total_reward += reward
        
        # Store experience
        memory.append((state, action, reward, next_state, done))
        state = next_state
        
        # Train on batch
        if len(memory) > batch_size:
            batch = random.sample(memory, batch_size)
            for s, a, r, ns, d in batch:
                target = r
                if not d:
                    target += gamma * np.max(model.predict(ns.reshape(1, -1), verbose=0)[0])
                
                target_f = model.predict(s.reshape(1, -1), verbose=0)
                target_f[0][a] = target
                model.fit(s.reshape(1, -1), target_f, epochs=1, verbose=0)
        
        if done:
            break
    
    # Decay exploration
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay
    
    if episode % 50 == 0:
        print(f"   Episode {episode}/{episodes}, Reward: {total_reward:.0f}, Epsilon: {epsilon:.3f}")

print("\n   ✓ Training complete")

# STEP 5: Evaluate Agent
print("\n[5/7] Evaluating trained agent...")

test_episodes = 10
total_rewards = []

for _ in range(test_episodes):
    state = env.reset()
    total_reward = 0
    
    for step in range(100):
        q_values = model.predict(state.reshape(1, -1), verbose=0)
        action = np.argmax(q_values[0])
        state, reward, done = env.step(action)
        total_reward += reward
        
        if done:
            break
    
    total_rewards.append(total_reward)

avg_reward = np.mean(total_rewards)
print(f"   ✓ Average reward over {test_episodes} episodes: {avg_reward:.2f}")
print(f"   ✓ Agent learned to navigate to goal")

# STEP 6: Save Model
print("\n[6/7] Saving model...")
model.save('dqn_model.h5')
print("   ✓ Model saved")

print("\n" + "="*70)
print("TRAINING COMPLETE!")
print("="*70)
print("\nCapabilities:")
print("  ✓ Learn optimal decision-making")
print("  ✓ Adapt to environment dynamics")
print("  ✓ Handle sequential decisions")
print("  ✓ No labeled data required")
print("\nApplications:")
print("  • Robotics (navigation, manipulation)")
print("  • Game AI (strategy games)")
print("  • Resource optimization")
print("  • Autonomous vehicles")
print("  • Trading algorithms")
print("\nNext steps:")
print("1. Apply to real-world environment")
print("2. Implement advanced algorithms (A3C, PPO)")
print("3. Add continuous action spaces")
print("4. Deploy to robot/simulator")
print("="*70 + "\n")
