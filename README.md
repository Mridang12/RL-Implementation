# RL-Implementation
Wrote a DQN agent for flappy bird, reusing here to see how far I can push the DQN algorithm on different environments. So far I have tried using it on CartPole-v0 and LunarLander-v2, and it converges for both after tuning hyperparameters :)

DQN stands for Deep Q Learning, which is a technique that improves on the massive memory requirements of vanilla Q Learning by estimating the Q(s,a) value pairs for each action from a given state using Neural Networks. But given the nature of Reinforcement Learning, where the agent does not have information about its environment, trying to 'fit' the neural network becomes difficult since we do not have a set "target" value to fit towards. To solve this problem, Deep Q Learning maintains two different neural networks, one that predicts the Q values to form a policy, and the other that provides estimated "targets" to train towards. To make training more stable, remove the correlation between recent (state, action, reward) pairs and to reuse past experiences, a technique called Experience Replay is also used here. 

To learn more about Deep Q Learning, I recommend reading this article [here.](https://www.analyticsvidhya.com/blog/2019/04/introduction-deep-q-learning-python/)

# Dependencies 

    -- PyTorch
    -- Numpy
    -- Cuda for PyTorch (Not Required but Highly Recommended)

# Usage

The DQN Agent is pretty much plug and play. The agent outputs a discrete action space [0, n-1]

It takes in the following parameters:

    1. layerSizes: The sizes of the Neural Network layers as a list. Typically you would want to input [len(state), ... , len(action_space)].

    2. epsilon : A start value for the epsilon greedy policy. Recommended value = 1

    3. eps_decay : Epsilon multiplier at the end of each episode. value < 1.

    4. min_eps : Minimum value beyond which epsilon will not decay.

    5. mem_size : Size for the Replay Memory buffer.

    6. batch_size : The size of the random batch taken from replay memory buffer to train on.

    7. discount_factor : The gamma value for Q updates.

    8. update_freq : Policy Network will update every update_freq timesteps. 

    9. target_update_freq : Target Network will be refreshed with the weights of the policy network every target_update_freq timesteps.

    10. lr : Learning rate for the Adam Optimizer.

    11. num_actions : Size of the action_space. 

    12. start_training_after : The agent will start training after this number of experiences has been collected.


Every timestep of the environment:

    -- Call agent.getAction(state) to get back an action depending on the eps greedy policy.

    -- Call agent.step(state, action, reward, next_state, done) to store experience and train according to the hyperparameters.

# Example training CartPole-v0 env from openAI gym

```python
import gym
from DQNAgent import DQNAgent
import numpy as np

#Change hyperparameters here
agent = DQNAgent(layerSizes=[4, 64, 2],
                epsilon=1, eps_decay=0.9997,min_eps=0.001, batch_size=32, discount_fact=0.99, 
                 lr=0.15, num_actions=2, target_update_freq=10000, mem_size = 65000,
                    update_freq = 2)

env = gym.make('CartPole-v0')

i_episode = 0
last_n_scores = [0]
n = 10
progress = []
running = []
done = False

# Convergence Condition
while np.mean(last_n_scores) < 195:
    
    i_episode += 1
    state = env.reset()
    score = 0
    agent.loss_val = 0
    running = []
    
    for t in range(1000):
        env.render()
        action = agent.getAction(state)
        n_state, reward, done, info = env.step(action)
        score += reward
        agent.step(state, action, reward, n_state, int(done))
        running.append(action)
        state = n_state
        
        if done:
            print(f"{i_episode} Episode finished after {t+1} timesteps")
            if len(last_n_scores) == n:
                last_n_scores.pop(0)
            last_n_scores.append(score)
            break
env.close()

```

