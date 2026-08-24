# Policy Gradient with PyTorch

Till now, we've trained a value based methods. In value based metods, policy uses greedy algorithm or another policy to select the state, action pair with maximum value(Q-Learning, DQN). In this notebook, we'll cover policy gradient(subset of policy-based methods) in policy based method to optimize policy directly.

Environmens: CartPole-v1 and PixelCopter.

## What are the policy-based methods

As repeatedly iterated: what is the goal of reinforcement learning?
- To maximize the expected cumulative reward. What is the cumulative reward, given an environment, possible set of actions. Starting at a given state and the cumulative reward until the episode ends.
- Reinforcement learning is grounded in *reward hypotheseis*: All goals can be described as maximization of expected cumulative reward.

Example:
In a footabll game, goal is to maximize the goals scored and minimze the goals conceded.

### Difference between value and policy methods.

1. Value-Based methods: Optimize a value function to lead to an optimal policy. Here the objective is to reduce the loss between predicted and target value. Here it's off policy. The policy is learned in the value and the action taken is based on values, it's implicit not explicit. The parameter optimization is to find the optimal value function.
2. Policy-Based methods: Directly optimize the policy. Parameterize the policy(netword) to output probability over the action space. *Stochastic Policy*

Stochastic Policy
$\pi_{\theta}(s) = ℙ[A | s;\theta]$

Policy $\pi$ given a state $s$ outputs a probability distribution $ℙ$ over actions $A$ at that state $s$. $\theta$ is the parameters.

In policy-based methods:
- We directly optimize the objective function $J(\theta)$ which is the expected cumulative reward, and we want to find out the parameters $\theta$ that maximizes this objective function.
- Policy is optimized by gradient ascent(not descent).

### Difference between Policy-Based and Policy-Gradient methods

1. Policy-Based: Maximize local approxation of the objectve function(cumulative reward) with techniques like hill climibing, simulated annealing etc. Indirect parameter updates: why? Update parameters --> action changes --> calculate reward --> Higher reward(retain params)/Lower reward(revert).
2. Policy-Gradient: Maximize cumulative reward(objective) by directly updating policy parameters using gradient ascent on expected reward.

## Advantages of Policy-Gradient methods

1. Integration simplicity, we don't have to store additional data(values).
2. They can learn stochastic(random) policy, what does this mean:
    - We don't have to setup exploration/exploitation like in value-based methods to explore the state space. Since it's stochastic[output a probability distribution over actions], the agent explores the state space.
    - Get rid of perceptual aliasing, when two states seem the same but requires different actions.
3. In Deep Q-Learning, assigns a prediction score(maximum expected cumulative reward) for every action given a time step and current state. What if the action is space is infinite, as in a self driving car? We'll have to produce a Q-Value for each possible action and then taking the max of continuous output is another optimization problem itself.
4. In Value-based methods, we use epsilon greedy to select an action. What if an action value aribtrarily changes and becomes the maximum, this small arbitrary change creates a dramatic change in action. On other hand stoachastic(policy-gradient) actions change smoothly over time.

## Disadvantages of Policy-Gradient methods

1. They converge to a local maximum instead of global optimum.
2. Training takes longer, step by step.
3. Can have high variance.

PS: These are all abstract theory, the advantages make sense theoretically and intuitivley, but if possible obtain a deeper understanding with implementation.

## Diving Deeper into the policy-gradient methods

### The Big Picture

Policy-Gradient aim: To find parameters $\theta$ that maximize cumulative reward.

Policy: Stochastic parameterized policy --> Neural network that outputs a probability distribution over actions. Probability of taking each action is called action preference.

```
Cart Pole Example
-------------------------------------------------
Inputs: Cart Position, Velocity, Pole Angle, Angular Velocity
-------------------------------------------------
                    ||
                    ||
                    ||
-------------------------------------------------
                    ||
                    ||
                    ||
-------------------------------------------------
      Input Layer |
      FC Layer    |  Policy
      Soft Max    |
-------------------------------------------------
                    ||
                    ||
                    ||
-------------------------------------------------
Action Preferences/Output: push Left | Push Right
-------------------------------------------------
```

Policy-Gradient goal is to maximize the cumulative reward, neural network outputs a probability distribution over actions. How do we maximize the cumulative reward? Good actions, actions that maximize return are sample more frequently in the future.

How do we tweak the weights for good actions? Idea is to let the agent interact during an episode. Calculate the return, if positive update all action states taken during the epiosde to have higher probability.

```
|episode_start| --> |state_0|action_0| --> |state_1|action_1| --> |state_n|action_n| --> |episode_end|
```

For each state|action increase or decrease $P(a|s)$ occured during the episode.

### Deeper Dive

Terminology:
- $Π$ --> Stocahstic Policy
- $\theta$ --> Parameter
- $P[A|s;\theta]$ -> Probability distribution of actions.

Parametrized stochastic policy, given a state emits a probability distribution of actions.

But how do we know the policy is good? How do we calculate the expected cumulative reward? We define a score/objective function $J(\theta)$.

Let's first define the objective function:
- Given a state, agent can take any path. This path is called trajectory. Trajectory - state action sequence without considering reward. Here it's the datamodel, an episode is a [(s,a,r),...] and trajectory is [(s,a)...]
- What trajectory is taken is determined the policy(parameters).
- Calculate Cumulative return from trajectory.
- Given the policy is stochastic, we sample certain possible trajectories given a state.
- Objective function output is the weighted average of all possible rewards across sampled trajectories. Here weights are the probability of a trajectory.

Equation:

$$
J(\theta)=\sum_{\tau} P(\tau;\theta)\,R(\tau)
$$

$$
P(\tau;\theta)=\rho(s_0)\prod_{t=0}^{T-1}\pi_\theta(a_t \mid s_t)\,P(s_{t+1}\mid s_t,a_t)
$$

$$
R(\tau)=\sum_{t=0}^{T-1}\gamma^t r_t
$$

- $P(\tau;\theta)$:
  - Probability of observing trajectory $\tau$ under policy $\pi_\theta$. -- formal
  - A trajectory is a sequence of state actions we've sequence of probability for each state action. these are multiplied(combined) together to give trajectory likelihood.
- $R(\tau)$:
  - Total discounted return accumulated along trajectory $\tau$
  - This is the cumulative reward if an trajectory is experienced.
$$

- $J(\theta)$ is the probability-weighted average of rewards over all possible trajectories.
- In practice, we estimate this using sampled trajectories.

So objective is the way of measuring the cumulative reward, then the goal of the policy is to maximize the objective.

Now that the theory is done, let's code a policy-gradient reinforcement algorithm: Monte-Carlo Policy Gradient from scratch using PyTorch.


```python
%%capture
!apt install python-opengl
!apt install ffmpeg
!apt install xvfb
!pip install pyvirtualdisplay
!pip install pyglet==1.5.1
```


```python
# Virtual display
from pyvirtualdisplay import Display

virtual_display = Display(visible=0, size=(1400, 900))
virtual_display.start()
```




    <pyvirtualdisplay.display.Display at 0x7aa1ba6ca180>



For this hands-onn excercise: we're gonna use gym instead of gymnasium, Because gym-games isn't updated with gymnaisum.

Differences:
1. In gym, we don't have terminated and truncated but only done.
2. In gym using env.step() returns state, reward, done, info.


```python
!pip install git+https://github.com/ntasfi/PyGame-Learning-Environment.git
!pip install git+https://github.com/simoninithomas/gym-games
!pip install huggingface_hub imageio-ffmpeg pyyaml
!pip install numpy==1.26.4
```

    Collecting git+https://github.com/ntasfi/PyGame-Learning-Environment.git
      Cloning https://github.com/ntasfi/PyGame-Learning-Environment.git to /tmp/pip-req-build-nitbb_rl
      Running command git clone --filter=blob:none --quiet https://github.com/ntasfi/PyGame-Learning-Environment.git /tmp/pip-req-build-nitbb_rl
      Resolved https://github.com/ntasfi/PyGame-Learning-Environment.git to commit 3dbe79dc0c35559bb441b9359948aabf9bb3d331
      Preparing metadata (setup.py) ... [?25l[?25hdone
    Requirement already satisfied: numpy in /usr/local/lib/python3.12/dist-packages (from ple==0.0.1) (2.0.2)
    Requirement already satisfied: Pillow in /usr/local/lib/python3.12/dist-packages (from ple==0.0.1) (11.3.0)
    Building wheels for collected packages: ple
      Building wheel for ple (setup.py) ... [?25l[?25hdone
      Created wheel for ple: filename=ple-0.0.1-py3-none-any.whl size=50769 sha256=16529eb7d6d784a559070d284c2df435ae2ed1b22d2eeb76b40245de277fa79d
      Stored in directory: /tmp/pip-ephem-wheel-cache-kl_iswpr/wheels/6d/3c/74/aa0f046a54330af388e34b880213857c59e03b701cdcd9c38f
    Successfully built ple
    Installing collected packages: ple
    Successfully installed ple-0.0.1
    Collecting git+https://github.com/simoninithomas/gym-games
      Cloning https://github.com/simoninithomas/gym-games to /tmp/pip-req-build-tqld0r3k
      Running command git clone --filter=blob:none --quiet https://github.com/simoninithomas/gym-games /tmp/pip-req-build-tqld0r3k
      Resolved https://github.com/simoninithomas/gym-games to commit f31695e4ba028400628dc054ee8a436f28193f0b
      Preparing metadata (setup.py) ... [?25l[?25hdone
    Requirement already satisfied: numpy>=1.16.4 in /usr/local/lib/python3.12/dist-packages (from gym-games==1.0.4) (2.0.2)
    Requirement already satisfied: gym>=0.13.0 in /usr/local/lib/python3.12/dist-packages (from gym-games==1.0.4) (0.25.2)
    Requirement already satisfied: setuptools>=65.5.1 in /usr/local/lib/python3.12/dist-packages (from gym-games==1.0.4) (75.2.0)
    Requirement already satisfied: pygame>=1.9.6 in /usr/local/lib/python3.12/dist-packages (from gym-games==1.0.4) (2.6.1)
    Requirement already satisfied: ple>=0.0.1 in /usr/local/lib/python3.12/dist-packages (from gym-games==1.0.4) (0.0.1)
    Requirement already satisfied: cloudpickle>=1.2.0 in /usr/local/lib/python3.12/dist-packages (from gym>=0.13.0->gym-games==1.0.4) (3.1.2)
    Requirement already satisfied: gym-notices>=0.0.4 in /usr/local/lib/python3.12/dist-packages (from gym>=0.13.0->gym-games==1.0.4) (0.1.0)
    Requirement already satisfied: Pillow in /usr/local/lib/python3.12/dist-packages (from ple>=0.0.1->gym-games==1.0.4) (11.3.0)
    Building wheels for collected packages: gym-games
      Building wheel for gym-games (setup.py) ... [?25l[?25hdone
      Created wheel for gym-games: filename=gym_games-1.0.4-py3-none-any.whl size=17308 sha256=f8bb9a021947dcc1f7b927b6f619324da03753bea869775009cbc2166995dcbf
      Stored in directory: /tmp/pip-ephem-wheel-cache-d0qsnd4j/wheels/b8/fb/47/2b4c7a78820f5b608efeb73af613bd6ef8f8e15bf003744158
    Successfully built gym-games
    Installing collected packages: gym-games
    Successfully installed gym-games-1.0.4
    Requirement already satisfied: huggingface_hub in /usr/local/lib/python3.12/dist-packages (1.20.1)
    Requirement already satisfied: imageio-ffmpeg in /usr/local/lib/python3.12/dist-packages (0.6.0)
    Requirement already satisfied: pyyaml in /usr/local/lib/python3.12/dist-packages (6.0.3)
    Requirement already satisfied: click>=8.4.0 in /usr/local/lib/python3.12/dist-packages (from huggingface_hub) (8.4.2)
    Requirement already satisfied: filelock>=3.10.0 in /usr/local/lib/python3.12/dist-packages (from huggingface_hub) (3.29.4)
    Requirement already satisfied: fsspec>=2023.5.0 in /usr/local/lib/python3.12/dist-packages (from huggingface_hub) (2025.3.0)
    Requirement already satisfied: hf-xet<2.0.0,>=1.5.1 in /usr/local/lib/python3.12/dist-packages (from huggingface_hub) (1.5.1)
    Requirement already satisfied: httpx<1,>=0.23.0 in /usr/local/lib/python3.12/dist-packages (from huggingface_hub) (0.28.1)
    Requirement already satisfied: packaging>=20.9 in /usr/local/lib/python3.12/dist-packages (from huggingface_hub) (26.2)
    Requirement already satisfied: tqdm>=4.42.1 in /usr/local/lib/python3.12/dist-packages (from huggingface_hub) (4.67.3)
    Requirement already satisfied: typer<0.26.0,>=0.20.0 in /usr/local/lib/python3.12/dist-packages (from huggingface_hub) (0.25.1)
    Requirement already satisfied: typing-extensions>=4.1.0 in /usr/local/lib/python3.12/dist-packages (from huggingface_hub) (4.15.0)
    Requirement already satisfied: anyio in /usr/local/lib/python3.12/dist-packages (from httpx<1,>=0.23.0->huggingface_hub) (4.14.0)
    Requirement already satisfied: certifi in /usr/local/lib/python3.12/dist-packages (from httpx<1,>=0.23.0->huggingface_hub) (2026.6.17)
    Requirement already satisfied: httpcore==1.* in /usr/local/lib/python3.12/dist-packages (from httpx<1,>=0.23.0->huggingface_hub) (1.0.9)
    Requirement already satisfied: idna in /usr/local/lib/python3.12/dist-packages (from httpx<1,>=0.23.0->huggingface_hub) (3.18)
    Requirement already satisfied: h11>=0.16 in /usr/local/lib/python3.12/dist-packages (from httpcore==1.*->httpx<1,>=0.23.0->huggingface_hub) (0.16.0)
    Requirement already satisfied: shellingham>=1.3.0 in /usr/local/lib/python3.12/dist-packages (from typer<0.26.0,>=0.20.0->huggingface_hub) (1.5.4)
    Requirement already satisfied: rich>=13.8.0 in /usr/local/lib/python3.12/dist-packages (from typer<0.26.0,>=0.20.0->huggingface_hub) (13.9.4)
    Requirement already satisfied: annotated-doc>=0.0.2 in /usr/local/lib/python3.12/dist-packages (from typer<0.26.0,>=0.20.0->huggingface_hub) (0.0.4)
    Requirement already satisfied: markdown-it-py>=2.2.0 in /usr/local/lib/python3.12/dist-packages (from rich>=13.8.0->typer<0.26.0,>=0.20.0->huggingface_hub) (4.2.0)
    Requirement already satisfied: pygments<3.0.0,>=2.13.0 in /usr/local/lib/python3.12/dist-packages (from rich>=13.8.0->typer<0.26.0,>=0.20.0->huggingface_hub) (2.20.0)
    Requirement already satisfied: mdurl~=0.1 in /usr/local/lib/python3.12/dist-packages (from markdown-it-py>=2.2.0->rich>=13.8.0->typer<0.26.0,>=0.20.0->huggingface_hub) (0.1.2)
    Collecting numpy==1.26.4
      Downloading numpy-1.26.4-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (61 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m61.0/61.0 kB[0m [31m2.3 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading numpy-1.26.4-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (18.0 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m18.0/18.0 MB[0m [31m104.2 MB/s[0m eta [36m0:00:00[0m
    [?25hInstalling collected packages: numpy
      Attempting uninstall: numpy
        Found existing installation: numpy 2.0.2
        Uninstalling numpy-2.0.2:
          Successfully uninstalled numpy-2.0.2
    [31mERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
    tifffile 2026.4.11 requires numpy>=2.0, but you have numpy 1.26.4 which is incompatible.
    opencv-contrib-python 4.13.0.92 requires numpy>=2; python_version >= "3.9", but you have numpy 1.26.4 which is incompatible.
    rasterio 1.5.0 requires numpy>=2, but you have numpy 1.26.4 which is incompatible.
    xarray-einstats 0.10.0 requires numpy>=2.0, but you have numpy 1.26.4 which is incompatible.
    tobler 0.14.0 requires numpy>=2.0, but you have numpy 1.26.4 which is incompatible.
    opencv-python 4.13.0.92 requires numpy>=2; python_version >= "3.9", but you have numpy 1.26.4 which is incompatible.
    shap 0.52.0 requires numpy>=2, but you have numpy 1.26.4 which is incompatible.
    pytensor 2.38.3 requires numpy>=2.0, but you have numpy 1.26.4 which is incompatible.
    jax 0.7.2 requires numpy>=2.0, but you have numpy 1.26.4 which is incompatible.
    opencv-python-headless 4.13.0.92 requires numpy>=2; python_version >= "3.9", but you have numpy 1.26.4 which is incompatible.
    cupy-cuda12x 14.0.1 requires numpy<2.6,>=2.0, but you have numpy 1.26.4 which is incompatible.
    jaxlib 0.7.2 requires numpy>=2.0, but you have numpy 1.26.4 which is incompatible.[0m[31m
    [0mSuccessfully installed numpy-1.26.4





```python
import numpy as np

from collections import deque

import matplotlib.pyplot as plt
%matplotlib inline

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# Gym
import gym
import gym_pygame

# Hugging Face Hub
from huggingface_hub import notebook_login # To log to our Hugging Face account to be able to upload models to the Hub.
import imageio
```

    Gym has been unmaintained since 2022 and does not support NumPy 2.0 amongst other critical functionality.
    Please upgrade to Gymnasium, the maintained drop-in replacement of Gym, or contact the authors of your software and request that they upgrade.
    See the migration guide at https://gymnasium.farama.org/introduction/migration_guide/ for additional information.
    /usr/local/lib/python3.12/dist-packages/jupyter_client/session.py:203: DeprecationWarning: datetime.datetime.utcnow() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.now(datetime.UTC).
      return datetime.utcnow().replace(tzinfo=utc)
    /usr/local/lib/python3.12/dist-packages/jupyter_client/session.py:203: DeprecationWarning: datetime.datetime.utcnow() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.now(datetime.UTC).
      return datetime.utcnow().replace(tzinfo=utc)



```python
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device
```




    device(type='cuda', index=0)




```python
env_id = "CartPole-v1"
env = gym.make(env_id)
eval_env = gym.make(env_id)
```

    /usr/local/lib/python3.12/dist-packages/gym/core.py:317: DeprecationWarning: [33mWARN: Initializing wrapper in old step API which returns one bool instead of two. It is recommended to set `new_step_api=True` to use new step API. This will be the default behaviour in future.[0m
      deprecation(
    /usr/local/lib/python3.12/dist-packages/gym/wrappers/step_api_compatibility.py:39: DeprecationWarning: [33mWARN: Initializing environment in old step API which returns one bool instead of two. It is recommended to set `new_step_api=True` to use new step API. This will be the default behaviour in future.[0m
      deprecation(



```python
env.observation_space
```




    Box([-4.8000002e+00 -3.4028235e+38 -4.1887903e-01 -3.4028235e+38], [4.8000002e+00 3.4028235e+38 4.1887903e-01 3.4028235e+38], (4,), float32)




```python
env.action_space
```




    Discrete(2)




```python
# Reinforce architecture

from torch import nn
from torch.nn import functional as F


class Policy(nn.Module):
  def __init__(self, s_size, a_size, h_size):
    super(Policy, self).__init__()
    self.fc1 = nn.Linear(s_size, h_size)
    self.fc2 = nn.Linear(h_size, a_size)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return F.softmax(x, dim=1)

  def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)

```


```python
from collections import deque
def reinforce(policy, optimizer, n_training_episodes, max_t, gamma, print_every):

  # only for human understanding
  scores_dequeue = deque(maxlen=100) # Stores last 100 rewards in the episode
  scores = [] # Stores all rewards

  for i_episode in range(1, n_training_episodes+1):
    # returns by the policy.act
    saved_log_probs = []
    rewards = []
    state = env.reset() # reset at start of every episode
    # Generate episodes
    for t in range(max_t): # Loop in the environment until max_t or episode ends
      action, log_prob = policy.act(state) # act on current state
      saved_log_probs.append(log_prob) # store for loss calculation
      state, reward, done, _ = env.step(action) # post action, take a new step
      # store rewards at every timestep
      rewards.append(reward)
      if done:
        break

    # store sum of rewards in an episode, last max_len
    scores_dequeue.append(sum(rewards))
    # All rewards
    scores.append(sum(rewards))

    returns = deque(maxlen=max_t) # Initialize an empty deque to caclulate discounted reward
    n_steps = len(rewards) # max steps with rewards in the episode

    # Calculate discounted return for every timestep
    for t in range(n_steps)[::-1]: # number of steps -- len(rewards)
      # Using Bellman's equation to reduce number of calculations
      # for trajectory starting at 1 we need the discounted rewards untl epidode ends
      # we do the calculations in reverse and use the discounted return

      disc_return_t = (returns[0] if len(returns) > 0 else 0) # last step 0, there's no action reward.
      returns.appendleft(gamma * disc_return_t + rewards[t]) # then we start calculating cumulative rewards with next step undiscounted rewaard
      # with appendleft rewards are again in chronological order

    # Normalize the returns
    eps = np.finfo(np.float32).eps.item()
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)

    # Calculate loss for this episode
    policy_loss = []
    for log_prob, disc_return in zip(saved_log_probs, returns):
      policy_loss.append(-log_prob * disc_return) # negative log likehood of disc_return(action because disc_reward)
    policy_loss = torch.cat(policy_loss).sum()

    # optimize
    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()

    if i_episode % print_every == 0:
      print('Episode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_dequeue)))

  return scores

```


```python
# Define all the hyperparameters in a dict
a_size = env.action_space.n
s_size = env.observation_space.shape[0]
cartpole_hyperparameters = {
    "h_size": 16,
    "n_training_episodes": 1500,
    "n_evaluation_episodes": 10,
    "max_t": 500,
    "gamma": 1.0,
    "lr": 1e-2,
    "env_id": env_id,
    "state_space": s_size,
    "action_space": a_size,
}
```


```python
# # Instantiate policy
# from torch import optim
# cartpole_policy = Policy(
#     cartpole_hyperparameters["state_space"],
#     cartpole_hyperparameters["action_space"],
#     cartpole_hyperparameters["h_size"],
# ).to(device)
# cartpole_optimizer = optim.Adam(cartpole_policy.parameters(), lr=cartpole_hyperparameters["lr"])
```


```python
cartpole_policy
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    /tmp/ipykernel_1758/1409044714.py in <cell line: 0>()
    ----> 1 cartpole_policy
    

    NameError: name 'cartpole_policy' is not defined



```python
# scores = reinforce(
#   cartpole_policy,
#   cartpole_optimizer,
#   cartpole_hyperparameters["n_training_episodes"],
#   cartpole_hyperparameters["max_t"],
#   cartpole_hyperparameters["gamma"],
#   100,
# )
```


```python
def evaluate_agent(env, max_steps, n_eval_episodes, policy):
    """
    Evaluate the agent for `n_evaluation_episodes` episodes and returns average reward and std of reward.
    :param env: The evaluation environment
    :param n_evaluation_episodes: Number of episode to evaluate the agent
    :param policy: The Reinforce agent
    """
    episode_rewards = []
    for episode in range(n_eval_episodes):
        state = env.reset()
        step = 0
        done = False
        total_rewards_ep = 0

        for step in range(max_steps):
            action, _ = policy.act(state)
            new_state, reward, done, info = env.step(action)
            total_rewards_ep += reward

            if done:
                break
            state = new_state
        episode_rewards.append(total_rewards_ep)
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    return mean_reward, std_reward
```


```python
evaluate_agent(
    eval_env, cartpole_hyperparameters["max_t"], cartpole_hyperparameters["n_evaluation_episodes"], cartpole_policy
)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    /tmp/ipykernel_1758/1382172138.py in <cell line: 0>()
          1 evaluate_agent(
    ----> 2     eval_env, cartpole_hyperparameters["max_t"], cartpole_hyperparameters["n_evaluation_episodes"], cartpole_policy
          3 )


    NameError: name 'cartpole_policy' is not defined



```python
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.repocard import metadata_eval_result, metadata_save

from pathlib import Path
import datetime
import json
import imageio

import tempfile

import os
```


```python
def record_video(env, policy, out_directory, fps=30):
    """
    Generate a replay video of the agent
    :param env
    :param Qtable: Qtable of our agent
    :param out_directory
    :param fps: how many frame per seconds (with taxi-v3 and frozenlake-v1 we use 1)
    """
    images = []
    done = False
    state = env.reset()
    img = env.render(mode="rgb_array")
    images.append(img)
    while not done:
        # Take the action (index) that have the maximum expected future reward given that state
        action, _ = policy.act(state)
        state, reward, done, info = env.step(action)  # We directly put next_state = state for recording logic
        img = env.render(mode="rgb_array")
        images.append(img)
    imageio.mimsave(out_directory, [np.array(img) for i, img in enumerate(images)], fps=fps)
```


```python
def push_to_hub(repo_id,
                model,
                hyperparameters,
                eval_env,
                video_fps=30
                ):
  """
  Evaluate, Generate a video and Upload a model to Hugging Face Hub.
  This method does the complete pipeline:
  - It evaluates the model
  - It generates the model card
  - It generates a replay video of the agent
  - It pushes everything to the Hub

  :param repo_id: repo_id: id of the model repository from the Hugging Face Hub
  :param model: the pytorch model we want to save
  :param hyperparameters: training hyperparameters
  :param eval_env: evaluation environment
  :param video_fps: how many frame per seconds to record our video replay
  """

  _, repo_name = repo_id.split("/")
  api = HfApi()

  # Step 1: Create the repo
  repo_url = api.create_repo(
        repo_id=repo_id,
        exist_ok=True,
  )

  with tempfile.TemporaryDirectory() as tmpdirname:
    local_directory = Path(tmpdirname)

    # Step 2: Save the model
    torch.save(model, local_directory / "model.pt")

    # Step 3: Save the hyperparameters to JSON
    with open(local_directory / "hyperparameters.json", "w") as outfile:
      json.dump(hyperparameters, outfile)

    # Step 4: Evaluate the model and build JSON
    mean_reward, std_reward = evaluate_agent(eval_env,
                                            hyperparameters["max_t"],
                                            hyperparameters["n_evaluation_episodes"],
                                            model)
    # Get datetime
    eval_datetime = datetime.datetime.now()
    eval_form_datetime = eval_datetime.isoformat()

    evaluate_data = {
          "env_id": hyperparameters["env_id"],
          "mean_reward": mean_reward,
          "n_evaluation_episodes": hyperparameters["n_evaluation_episodes"],
          "eval_datetime": eval_form_datetime,
    }

    # Write a JSON file
    with open(local_directory / "results.json", "w") as outfile:
        json.dump(evaluate_data, outfile)

    # Step 5: Create the model card
    env_name = hyperparameters["env_id"]

    metadata = {}
    metadata["tags"] = [
          env_name,
          "reinforce",
          "reinforcement-learning",
          "custom-implementation",
          "deep-rl-class"
      ]

    # Add metrics
    eval = metadata_eval_result(
        model_pretty_name=repo_name,
        task_pretty_name="reinforcement-learning",
        task_id="reinforcement-learning",
        metrics_pretty_name="mean_reward",
        metrics_id="mean_reward",
        metrics_value=f"{mean_reward:.2f} +/- {std_reward:.2f}",
        dataset_pretty_name=env_name,
        dataset_id=env_name,
      )

    # Merges both dictionaries
    metadata = {**metadata, **eval}

    model_card = f"""
  # **Reinforce** Agent playing **{env_id}**
  This is a trained model of a **Reinforce** agent playing **{env_id}** .
  To learn to use this model and train yours check Unit 4 of the Deep Reinforcement Learning Course: https://huggingface.co/deep-rl-course/unit4/introduction
  """

    readme_path = local_directory / "README.md"
    readme = ""
    if readme_path.exists():
        with readme_path.open("r", encoding="utf8") as f:
          readme = f.read()
    else:
      readme = model_card

    with readme_path.open("w", encoding="utf-8") as f:
      f.write(readme)

    # Save our metrics to Readme metadata
    metadata_save(readme_path, metadata)

    # Step 6: Record a video
    video_path =  local_directory / "replay.mp4"
    record_video(env, model, video_path, video_fps)

    # Step 7. Push everything to the Hub
    api.upload_folder(
          repo_id=repo_id,
          folder_path=local_directory,
          path_in_repo=".",
    )

    print(f"Your model is pushed to the Hub. You can view your model here: {repo_url}")
```


```python
# repo_id = "JpChi/CartPole-v1"  # TODO Define your repo id {username/Reinforce-{model-id}}
# push_to_hub(
#     repo_id,
#     cartpole_policy,  # The model we want to save
#     cartpole_hyperparameters,  # Hyperparameters
#     eval_env,  # Evaluation environment
#     video_fps=30
# )
```


```python
from huggingface_hub import notebook_login
```


```python
notebook_login()
```


<center><img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" width="100" alt="Hugging Face"><br><br><p>To log in, open this URL and enter the code:</p><p><a href="https://hf.co/oauth/device" target="_blank"><b>https://hf.co/oauth/device</b></a></p><p style="font-size: 1.6em; letter-spacing: 0.3em; font-family: monospace;"><b>LJ10-LCHL</b></p></center>



<center><i>Waiting for authorization...</i></center>



    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    /tmp/ipykernel_1758/1380886486.py in <cell line: 0>()
    ----> 1 notebook_login()
    

    /usr/local/lib/python3.12/dist-packages/huggingface_hub/_login.py in notebook_login(skip_if_logged_in)
        376     display(HTML("<center><i>Waiting for authorization...</i></center>"))
        377     try:
    --> 378         response = poll_device_token(device_info)
        379     except DeviceCodeError as e:
        380         display(HTML(f"<center><b style='color: red;'>Login failed: {html.escape(str(e))}</b></center>"))


    /usr/local/lib/python3.12/dist-packages/huggingface_hub/utils/_oauth_device.py in poll_device_token(device_info, on_pending)
        148                     )
        149 
    --> 150         time.sleep(interval)
        151 
        152     raise DeviceCodeError("Device code expired (timeout). Please try again.", error_code=OAuthErrorCode.EXPIRED_TOKEN)


    KeyboardInterrupt: 


Next let's implement PixelCopter. State space, 7, action space 2.


```python
env_id = "Pixelcopter-PLE-v0"
```


```python
env = gym.make("Pixelcopter-PLE-v0")
eval_env = gym.make("Pixelcopter-PLE-v0")
```

    couldn't import doomish
    Couldn't import doom


    /usr/local/lib/python3.12/dist-packages/pygame/pkgdata.py:25: DeprecationWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html
      from pkg_resources import resource_stream, resource_exists
    /usr/local/lib/python3.12/dist-packages/pkg_resources/__init__.py:3154: DeprecationWarning: Deprecated call to `pkg_resources.declare_namespace('google')`.
    Implementing implicit namespace packages (as specified in PEP 420) is preferred to `pkg_resources.declare_namespace`. See https://setuptools.pypa.io/en/latest/references/keywords.html#keyword-namespace-packages
      declare_namespace(pkg)
    /usr/local/lib/python3.12/dist-packages/pkg_resources/__init__.py:3154: DeprecationWarning: Deprecated call to `pkg_resources.declare_namespace('sphinxcontrib')`.
    Implementing implicit namespace packages (as specified in PEP 420) is preferred to `pkg_resources.declare_namespace`. See https://setuptools.pypa.io/en/latest/references/keywords.html#keyword-namespace-packages
      declare_namespace(pkg)
    /usr/local/lib/python3.12/dist-packages/gym/core.py:317: DeprecationWarning: [33mWARN: Initializing wrapper in old step API which returns one bool instead of two. It is recommended to set `new_step_api=True` to use new step API. This will be the default behaviour in future.[0m
      deprecation(
    /usr/local/lib/python3.12/dist-packages/gym/wrappers/step_api_compatibility.py:39: DeprecationWarning: [33mWARN: Initializing environment in old step API which returns one bool instead of two. It is recommended to set `new_step_api=True` to use new step API. This will be the default behaviour in future.[0m
      deprecation(
    /usr/local/lib/python3.12/dist-packages/jupyter_client/session.py:203: DeprecationWarning: datetime.datetime.utcnow() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.now(datetime.UTC).
      return datetime.utcnow().replace(tzinfo=utc)



```python
s_size = env.observation_space.shape[0]
```


```python
a_size = env.action_space.n
```


```python
class Policy(nn.Module):
    def __init__(self, s_size, a_size, h_size):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(s_size, h_size)
        self.fc2 = nn.Linear(h_size, h_size * 2)
        self.fc3 = nn.Linear(h_size * 2, a_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=1)

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)
```


```python
pixelcopter_hyperparameters = {
    "h_size": 64,
    "n_training_episodes": 50000,
    "n_evaluation_episodes": 10,
    "max_t": 10000,
    "gamma": 0.99,
    "lr": 1e-4,
    "env_id": env_id,
    "state_space": s_size,
    "action_space": a_size,
}
```


```python
# Create policy and place it to the device
# torch.manual_seed(50)
policy = Policy(
    pixelcopter_hyperparameters["state_space"],
    pixelcopter_hyperparameters["action_space"],
    pixelcopter_hyperparameters["h_size"],
).to(device)
pixelcopter_optimizer = optim.Adam(policy.parameters(), lr=pixelcopter_hyperparameters["lr"])
```

    /usr/local/lib/python3.12/dist-packages/jupyter_client/session.py:203: DeprecationWarning: datetime.datetime.utcnow() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.now(datetime.UTC).
      return datetime.utcnow().replace(tzinfo=utc)



```python
scores = reinforce(
    policy,
    pixelcopter_optimizer,
    pixelcopter_hyperparameters["n_training_episodes"],
    pixelcopter_hyperparameters["max_t"],
    pixelcopter_hyperparameters["gamma"],
    1000,
)
```

    /usr/local/lib/python3.12/dist-packages/gym/utils/passive_env_checker.py:174: UserWarning: [33mWARN: Future gym versions will require that `Env.reset` can be passed a `seed` instead of using `Env.seed` for resetting the environment random number generator.[0m
      logger.warn(
    /usr/local/lib/python3.12/dist-packages/gym/utils/passive_env_checker.py:190: UserWarning: [33mWARN: Future gym versions will require that `Env.reset` can be passed `return_info` to return information from the environment resetting.[0m
      logger.warn(
    /usr/local/lib/python3.12/dist-packages/gym/utils/passive_env_checker.py:195: UserWarning: [33mWARN: Future gym versions will require that `Env.reset` can be passed `options` to allow the environment initialisation to be passed additional information.[0m
      logger.warn(
    /usr/local/lib/python3.12/dist-packages/gym/utils/passive_env_checker.py:141: UserWarning: [33mWARN: The obs returned by the `reset()` method was expecting numpy array dtype to be float32, actual type: float64[0m
      logger.warn(
    /usr/local/lib/python3.12/dist-packages/gym/utils/passive_env_checker.py:165: UserWarning: [33mWARN: The obs returned by the `reset()` method is not within the observation space.[0m
      logger.warn(f"{pre} is not within the observation space.")
    /usr/local/lib/python3.12/dist-packages/gym/utils/passive_env_checker.py:227: DeprecationWarning: [33mWARN: Core environment is written in old step API which returns one bool instead of two. It is recommended to rewrite the environment with new step API. [0m
      logger.deprecation(
    /usr/local/lib/python3.12/dist-packages/gym/utils/passive_env_checker.py:233: DeprecationWarning: `np.bool8` is a deprecated alias for `np.bool_`.  (Deprecated NumPy 1.24)
      if not isinstance(done, (bool, np.bool8)):
    /usr/local/lib/python3.12/dist-packages/gym/utils/passive_env_checker.py:141: UserWarning: [33mWARN: The obs returned by the `step()` method was expecting numpy array dtype to be float32, actual type: float64[0m
      logger.warn(
    /usr/local/lib/python3.12/dist-packages/gym/utils/passive_env_checker.py:165: UserWarning: [33mWARN: The obs returned by the `step()` method is not within the observation space.[0m
      logger.warn(f"{pre} is not within the observation space.")


    Episode 1000	Average Score: 3.83
    Episode 2000	Average Score: 7.94
    Episode 3000	Average Score: 8.84
    Episode 4000	Average Score: 11.96
    Episode 5000	Average Score: 12.35
    Episode 6000	Average Score: 14.67
    Episode 7000	Average Score: 18.17
    Episode 8000	Average Score: 15.14
    Episode 9000	Average Score: 17.94
    Episode 10000	Average Score: 15.65
    Episode 11000	Average Score: 18.11
    Episode 12000	Average Score: 19.57
    Episode 13000	Average Score: 27.86
    Episode 14000	Average Score: 23.56
    Episode 15000	Average Score: 18.35
    Episode 16000	Average Score: 19.90
    Episode 17000	Average Score: 23.02
    Episode 18000	Average Score: 18.52
    Episode 19000	Average Score: 24.97
    Episode 20000	Average Score: 19.55
    Episode 21000	Average Score: 16.09
    Episode 22000	Average Score: 19.83
    Episode 23000	Average Score: 18.99
    Episode 24000	Average Score: 28.26
    Episode 25000	Average Score: 24.59
    Episode 26000	Average Score: 36.63
    Episode 27000	Average Score: 27.66
    Episode 28000	Average Score: 28.84
    Episode 29000	Average Score: 37.80
    Episode 30000	Average Score: 30.42
    Episode 31000	Average Score: 35.34
    Episode 32000	Average Score: 29.83
    Episode 33000	Average Score: 32.83
    Episode 34000	Average Score: 37.23
    Episode 35000	Average Score: 30.57
    Episode 36000	Average Score: 38.79
    Episode 37000	Average Score: 33.74
    Episode 38000	Average Score: 34.47
    Episode 39000	Average Score: 26.97
    Episode 40000	Average Score: 35.42
    Episode 41000	Average Score: 34.89
    Episode 42000	Average Score: 45.96
    Episode 43000	Average Score: 28.67
    Episode 44000	Average Score: 45.92
    Episode 45000	Average Score: 39.37
    Episode 46000	Average Score: 42.53
    Episode 47000	Average Score: 50.73
    Episode 48000	Average Score: 42.21
    Episode 49000	Average Score: 54.90
    Episode 50000	Average Score: 46.82



```python
evaluate_agent(
    eval_env, pixelcopter_hyperparameters["max_t"], pixelcopter_hyperparameters["n_evaluation_episodes"], policy
)
```




    (55.2, 55.05778782334067)




```python
from huggingface_hub import notebook_login
notebook_login()
```


<center><img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" width="100" alt="Hugging Face"><br><br><p>To log in, open this URL and enter the code:</p><p><a href="https://hf.co/oauth/device" target="_blank"><b>https://hf.co/oauth/device</b></a></p><p style="font-size: 1.6em; letter-spacing: 0.3em; font-family: monospace;"><b>K8KF-DUAM</b></p></center>



<center><i>Waiting for authorization...</i></center>



<center>Login successful. Logged in as <b>JpChi</b> (token: <code>oauth-JpChi</code>).<br>This token will be refreshed automatically when it expires.</center>



```python
repo_id = f"JpChi/{env_id}"
push_to_hub(
    repo_id,
    policy,  # The model we want to save
    pixelcopter_hyperparameters,  # Hyperparameters
    eval_env,  # Evaluation environment
    video_fps=30
)
```

    /usr/local/lib/python3.12/dist-packages/gym/core.py:43: DeprecationWarning: [33mWARN: The argument mode in render method is deprecated; use render_mode during environment initialization instead.
    See here for more information: https://www.gymlibrary.ml/content/api/[0m
      deprecation(
    /usr/local/lib/python3.12/dist-packages/gym/utils/passive_env_checker.py:280: UserWarning: [33mWARN: No render modes was declared in the environment (env.metadata['render_modes'] is None or not defined), you may have trouble when calling `.render()`.[0m
      logger.warn(


    Your model is pushed to the Hub. You can view your model here: https://huggingface.co/JpChi/Pixelcopter-PLE-v0



```python

```
