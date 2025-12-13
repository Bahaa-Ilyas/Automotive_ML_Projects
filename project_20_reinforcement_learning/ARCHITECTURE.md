# Reinforcement Learning Architecture

## Deep Q-Network (DQN)

```
State → Q-Network → Q-values for each action → Select best action
```

## Q-Network Architecture
```
Input(state_size) → Dense(24) → Dense(24) → Output(action_size)
```

## Training Algorithm
1. **Experience Collection**: Agent interacts with environment
2. **Replay Memory**: Store (state, action, reward, next_state)
3. **Batch Training**: Sample random batch from memory
4. **Q-Learning Update**: Q(s,a) = r + γ * max(Q(s',a'))
5. **Epsilon-Greedy**: Balance exploration vs exploitation

## Key Concepts
- **State**: Current environment observation
- **Action**: Decision to make
- **Reward**: Feedback signal
- **Q-value**: Expected future reward for state-action pair
- **Epsilon**: Exploration rate (decays over time)
- **Gamma**: Discount factor for future rewards

## Deployment
- Robotics: Real-time decision-making
- Simulation: Train in simulator, deploy to real world
- Continuous learning: Update policy with new experiences
