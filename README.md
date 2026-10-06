# Reinforcement Learning with CartPole --- DQN

A hands-on Reinforcement Learning project built while learning the
fundamentals of Deep Reinforcement Learning using the **CartPole-v1**
environment from OpenAI Gym.

The project uses a **Deep Q-Network (DQN)** agent with **Keras-RL2** and
TensorFlow. The agent learns to balance a pole on a moving cart by
interacting with the environment and improving its actions based on
accumulated rewards.

## Project Overview

The main goal of this project was to understand the practical workflow
of a Reinforcement Learning problem:

-   Create and interact with an RL environment
-   Understand states, actions, rewards, and episodes
-   Build a neural-network-based Q-function
-   Use experience replay
-   Train a DQN agent
-   Evaluate the trained agent
-   Visualize the agent playing CartPole
-   Understand the basic concepts behind Deep Reinforcement Learning

## Environment

**CartPole-v1** is a classic Reinforcement Learning environment.

At every time step, the agent observes the state of the environment and
chooses one of two actions:

-   Move the cart to the left
-   Move the cart to the right

The objective is to keep the pole balanced for as many time steps as
possible.

### State

The environment provides four observations:

1.  Cart position
2.  Cart velocity
3.  Pole angle
4.  Pole angular velocity

### Actions

There are two discrete actions:

``` text
0 → Push cart to the left
1 → Push cart to the right
```

### Reward

The agent receives a reward for successfully keeping the pole balanced.

The training objective is therefore to learn a policy that maximizes the
total accumulated reward.

## Reinforcement Learning Concepts Used

### Q-Learning

Q-Learning estimates the expected future reward for taking an action in
a particular state.

The DQN used in this project replaces the traditional Q-table with a
neural network.

### Deep Q-Network (DQN)

The neural network takes the current state as input and predicts a
Q-value for each possible action.

The action with the highest learned value can then be selected by the
agent's policy.

### Experience Replay

The project uses:

``` python
SequentialMemory(limit=50000, window_length=1)
```

Experience replay stores previous interactions between the agent and
environment and samples experiences during training.

This helps make neural-network training more stable by reducing the
dependence between consecutive experiences.

### Target Model

The DQN agent maintains a target model and periodically updates it
using:

``` python
target_model_update=0.01
```

This helps stabilize the learning process.

## Neural Network Architecture

The DQN model used in this project is:

``` text
Input
  ↓
Flatten
  ↓
Dense(24, ReLU)
  ↓
Dense(32, ReLU)
  ↓
Dense(2, Linear)
```

The final layer contains two outputs because CartPole has two possible
actions.

## Tech Stack

-   Python 3.10.11
-   TensorFlow 2.12.0
-   Keras
-   Keras-RL2
-   OpenAI Gym 0.25.2
-   NumPy
-   Pygame

## Installation

Create and activate a virtual environment:

``` powershell
py -3.10 -m venv .mynewenv
.\.mynewenv\Scripts\Activate.ps1
```

Install the project dependencies:

``` powershell
pip install gym==0.25.2 tensorflow==2.12.0 keras-rl2 numpy pygame
```

Run the project:

``` powershell
python myfile.py
```

## Main Code

The DQN agent is created using:

``` python
agent = DQNAgent(
    model=model,
    memory=SequentialMemory(limit=50000, window_length=1),
    policy=BoltzmannQPolicy(),
    nb_actions=actions,
    nb_steps_warmup=10,
    target_model_update=0.01
)
```

The model is compiled using Adam:

``` python
agent.compile(
    optimizer=Adam(lr=0.001),
    metrics=['mae']
)
```

The agent is then trained for 10,000 steps:

``` python
agent.fit(
    env,
    nb_steps=10000,
    visualize=False,
    verbose=1
)
```

Finally, the trained agent is evaluated for 10 episodes:

``` python
result = agent.test(
    env,
    nb_episodes=10,
    visualize=True
)

print(np.mean(result.history['episode_reward']))
```

## Project Structure

``` text
reinforcement-learning-cartpole-dqn/
│
├── myfile.py
├── README.md
├── requirements.txt
├── notes/
│   └── Reinforcement_Learning_Notes.pdf
│
└── .gitignore
```

The `notes/` folder contains my handwritten Reinforcement Learning notes
used while learning the concepts behind this project.

## Learning Outcome

This project helped me build a practical understanding of:

-   Reinforcement Learning fundamentals
-   Markov Decision Process concepts
-   States, actions, rewards, and episodes
-   Q-Learning
-   Deep Q-Networks
-   Experience Replay
-   Target Networks
-   Exploration vs. exploitation
-   Neural networks for Reinforcement Learning
-   Training and evaluating an RL agent

## Future Improvements

Possible extensions to this project include:

-   Compare different RL policies
-   Experiment with different neural-network architectures
-   Tune DQN hyperparameters
-   Train for more episodes/steps
-   Track and visualize reward progression
-   Experiment with other Gym environments
-   Implement a DQN from scratch without Keras-RL2
-   Explore more advanced algorithms such as Double DQN, Dueling DQN,
    and PPO

## Notes

This is a **learning project** created to understand Reinforcement
Learning and Deep Q-Networks through a practical CartPole
implementation.

The repository also contains handwritten notes documenting the
Reinforcement Learning concepts studied during the project.
