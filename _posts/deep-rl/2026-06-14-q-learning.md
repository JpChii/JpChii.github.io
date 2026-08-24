# Q-Learning

* In this notebook we'll learn one of the Reinforcement Learning algorithms: Q-Learning --> is a value-based method.
* Implement a agent from scratch and train them in two environments:
  1. Frozen lake: Agent has to move from starting point to goal by avoding hole tiles and walking in frozen tiles.
  2. Autonomous taxi: Navigate a city to trasnport its passengers from point A to point B.

## Value based method

```
Input state --> Outputs expected value of being at that state.
Expected value --> expected discounted return(reward hypotheseis)
Agent obtains this reward, if it's starts at this state and act according to policy.
```

***But Value-based methods doesn't have an policy*** 👀 How does it work then?

1. Policy takes an action based on the value. This value can be associated with state or state action pair. - *Policy is function defined by hand*.
2. Since policy is not trained, we've to define it's behaviour based on value. Ex ***Greedy policy***: Takes action that always leads to biggest reward(biggest value).

***RL Flow:*** State -> Value of State-Action pairs -> Select one, go to state, take action(In case of Greedy the biggest action-value pair).

In value-based training, finding an optimal value function leads to having an optimal policy.

Math equation
$$
\pi^*(s) = \arg\max_{a} Q^*(s,a)
$$

### Two types of Value-Based methods

1. State-Value function: Trains to find optimal value associated with the state to select the state, to start, and then follows the policy forever.

$V_\pi(s) = \mathbb{E}_\pi \left[ G_t \mid S_t = s \right]$

$ \mathbb{E}_\pi $ - Expected reward.

2. State-Action-Value function: Trains to find optimal value associated with the state to select the state, to start and takes the action and then follows the policy forever after.

$V_\pi(s) = \mathbb{E}_\pi \left[ G_t \mid S_t = s ,A_t = a \right]$

The problem is the computation is expensive. To calculate EACH value of a state or state-action pair, we need to sum all possible rewards an agent can get if it starts at that state. Let's say current state 10 and there's 2^6 states ahead in this epsiode, we'll have to summ all 2^6 rewards. This is where Bellman Equatuation helps with.

### Bellman Equation

Let's see how the reward is calculated.

Given four states $S_1$ --> $S_2$ --> $S_3$ --> $S_4$ and their rewards at each step -1, -1, -1, -1 respectivley.

$V(S_1)$ = -1 + (-1) + (-1) + (-1) = -4

$V(S_2)$ = -1 + (-1) + (-1) = -3$

We're repeating the computations above. Bellman equation improves this as below.

Value of a state = Immediate reward + discounted reward of next state

$V(S_1) = r_{(t+2)} + \gamma * V(S_2)$

Now this reduces the computations and corrected expected return is obtained as well. Think about the gamma defintion from unit 1. Lower discount lower reward, Higher discount higher reward. This holds true won't it?

* If $\gamma$ = 0.1, reward of next state is reduced to 0.1% of it's original value.
* If $\gamma$ = 1^e6, reward of next state is 6xed of it's original.

### Diff between Value and Reward

* value - is the expected cumulative reward if agent starts in the given state and then acts according to policy.
* Reward - is the expected reward for an action taken in a given state. - Immediate reward.

### Learning strategies

RL agents learn by interacting with environment. Given an experience and recieved reward, agent updates the value or policy.

There are two learning strategies:

1. Monte-Carlo Learning Technique
2. Temporal Diffusion Learning Technique

#### Monte-Carlo Learning

Waits until end of episode to update value of a state.

$V(S_t) <- V(S_t) + \alpha[G_t - V(S_t)]$ - (1)

* $V(S_t)$[right side] --> Expected return or value for starting at that state.
* $G_t$ --> Discounted Return at timestep t. *Target* to update the policy.
* $V(S_t)$[left side] --> New value of state t.
* New state is updated with $\alpha$(learning rate) * error.

Let's run this through an example:

* Agent always starts the episode at same starting point.
* Agent takes action based on policy.
* Episode ends when agent is terminated(mouse is eaten by cat) or mouse moves 10 steps.
* At the end, we'll have a list of State, Action, Reward, Next State. [[$S_0$, $A_0$, $R_0$, $S_1$], [$S_1$, $A_1$, $R_1$, $S_2$]] etc.
* Then we'll update the value for each state using formula(1).
* $G_t$ - discounted return from state $S$

### Temporal Difference Learning

* Updates Value after a single step.
* Target for this update is Immediate Reward + Discounted value of next state.
* Discounted value is an estimate, as we're updating immedeiatley after a step.

$V(S_t) <- V(S_t) + \alpha[R_{t+1} + \gamma V(S_{t+1}) - V(S_t)]$ - (2)

* $R_{t+1} + \gamma V(S_{t+1})$ - is the expected return.
* This expected return is called bootstrapping, because we're using an existing estimate.


## Q-Learning

Q-Function(action-value function) is learned using Q-Learning algorithm. Q-Learning is an off-policy value based method that uses Temporal Difference to learn the value function.

* Q - Quality(value)
* Q-Function --> Q-Table. Q-Table is an state-action : value pair table. This table has a value for every state-action : pair possible.
* Intitially all values are set to zero.
* Then the table is updated at every step to arrive at an optimal Q-Table --> Q-Function.
* Off-Policy vs On-Policy: Off-Policy: Uses a different policy for training(actino to take, exploitation/exploration policy) and inference(greedy) choose the larges value state-action pair. On-Policy: Uses same policy for training and inference(updation).
* One more take at Off-Policy:
  - Exploration/Exploitation(acting policy) takes actions and returns RL Loop inputs(state, action, reward, next_state). This is the expreience from interacting with environment. Update Value function.
  - Greedy(Inference policy)-from training: Select largest value from Q-Table(Optimal) until epsiode terminates or truncates.


## Let's code Q-Learning Algorithm from scratch

Libraries:
1. gymnaisum - environments
2. numpy - Q-table(state-action, rows and columns)
3. pygame - for ui


```python
!pip install gymnasium numpy pygame -q
```


```python
# Virtual Display setup for rendering video
!sudo apt-get update -qq
!sudo apt-get install -y python3-opengl -qq
!apt install ffmpeg xvfb -qq
!pip3 install pyvirtualdisplay -q

import os
os.kill(os.getpid(), 9)
```

    W: Skipping acquire of configured file 'main/source/Sources' as repository 'https://r2u.stat.illinois.edu/ubuntu jammy InRelease' does not seem to provide it (sources.list entry misspelt?)
    debconf: unable to initialize frontend: Dialog
    debconf: (No usable dialog-like program is installed, so the dialog based frontend cannot be used. at /usr/share/perl5/Debconf/FrontEnd/Dialog.pm line 78, <> line 3.)
    debconf: falling back to frontend: Readline
    debconf: unable to initialize frontend: Readline
    debconf: (This frontend requires a controlling tty.)
    debconf: falling back to frontend: Teletype
    dpkg-preconfigure: unable to re-open stdin: 
    Selecting previously unselected package freeglut3:amd64.
    (Reading database ... 126109 files and directories currently installed.)
    Preparing to unpack .../freeglut3_2.8.1-6_amd64.deb ...
    Unpacking freeglut3:amd64 (2.8.1-6) ...
    Selecting previously unselected package libglu1-mesa:amd64.
    Preparing to unpack .../libglu1-mesa_9.0.2-1_amd64.deb ...
    Unpacking libglu1-mesa:amd64 (9.0.2-1) ...
    Selecting previously unselected package python3-opengl.
    Preparing to unpack .../python3-opengl_3.1.5+dfsg-1_all.deb ...
    Unpacking python3-opengl (3.1.5+dfsg-1) ...
    Setting up freeglut3:amd64 (2.8.1-6) ...
    Setting up libglu1-mesa:amd64 (9.0.2-1) ...
    Setting up python3-opengl (3.1.5+dfsg-1) ...
    Processing triggers for libc-bin (2.35-0ubuntu3.8) ...
    /sbin/ldconfig.real: /usr/local/lib/libtcm_debug.so.1 is not a symbolic link
    
    /sbin/ldconfig.real: /usr/local/lib/libtbbmalloc.so.2 is not a symbolic link
    
    /sbin/ldconfig.real: /usr/local/lib/libhwloc.so.15 is not a symbolic link
    
    /sbin/ldconfig.real: /usr/local/lib/libur_adapter_opencl.so.0 is not a symbolic link
    
    /sbin/ldconfig.real: /usr/local/lib/libumf.so.0 is not a symbolic link
    
    /sbin/ldconfig.real: /usr/local/lib/libur_adapter_level_zero.so.0 is not a symbolic link
    
    /sbin/ldconfig.real: /usr/local/lib/libtcm.so.1 is not a symbolic link
    
    /sbin/ldconfig.real: /usr/local/lib/libtbbbind_2_5.so.3 is not a symbolic link
    
    /sbin/ldconfig.real: /usr/local/lib/libtbb.so.12 is not a symbolic link
    
    /sbin/ldconfig.real: /usr/local/lib/libtbbbind_2_0.so.3 is not a symbolic link
    
    /sbin/ldconfig.real: /usr/local/lib/libtbbbind.so.3 is not a symbolic link
    
    /sbin/ldconfig.real: /usr/local/lib/libtbbmalloc_proxy.so.2 is not a symbolic link
    
    /sbin/ldconfig.real: /usr/local/lib/libur_loader.so.0 is not a symbolic link
    
    ffmpeg is already the newest version (7:4.4.2-0ubuntu0.22.04.1).
    xvfb is already the newest version (2:21.1.4-2ubuntu1.7~22.04.14).
    0 upgraded, 0 newly installed, 0 to remove and 47 not upgraded.



```python
# Virtual display
from pyvirtualdisplay import Display

virtual_display = Display(visible=0, size=(1400, 900))
virtual_display.start()
```




    <pyvirtualdisplay.display.Display at 0x79a43f7d8690>




```python
# Imports
import gymnasium
import imageio # Storing video
import random
import numpy as np  # numpy state action table
import os
```

### Frozen Lake

In Frozen lake environment agent needs to learn to reach the goal by navigating around holes.

Actions space:

- 0 - left
- 1 - up
- 2 - right
- 3 - up

Observation Space(States):

- Each tile has a number based on current_row * num_cols + current_column.
- Consider this a 4 * 4 grid. Columns[0-3], rows[0-3]
- Observation spaces walkthrough:
    - At starting position [row, col] -> [0, 0]. Observation space -> 0.
    - Second tile [row, col] -> [0, 1]. Observation space(0 * 4 + 1) -> 1.
    - In this way tiles are numbered from 0 to 15(goal) in observation space.

Starting state: [0, 0], observation: 0

Rewards:
- +1 - reach goal
- +0 - reach hole
- +0 - reach frozen

Termination:

- Into Hole.
- Episode ends(n_steps)
- Reaches goal(n_rows * n_columns - 1) --> 15

We can setup the environment with array of decsription [SFH, SFH, SFG]. This'll create a 3x3 grid with way to goal in diagonal. S - starting state, F- Frozen, H - Hole, G - Goal.

Additional_info:

Tiles can be made slippery to make the agent learn unslippery path to the goal as well! parameter: `is_slippery()`


```python
# create env
env_fl = gymnasium.make(
    'FrozenLake-v1', # Env id
    desc=None, # Create custom grid, if needed
    map_name="4x4", # Grid size
    is_slippery=False, # Slippery or Non Slippery Grid
    )
```


```python
env_fl.action_space, env_fl.action_space.sample()
```


```python
env_fl.observation_space, env_fl.observation_space.sample()
```




```python
print(f"Possible states: {env_fl.observation_space.n}")
print(f"Possible actions: {env_fl.action_space.n}")
```


```python
# 1. Initialize Q-Table
# Q-Table is state x action matrix
def intialize_q_table(state_space, action_space):
  return np.zeros((state_space, action_space))

q_table = intialize_q_table(env_fl.observation_space.n, env_fl.action_space.n) # 14 possible state, with 4 actions per state, 14 * 4 matrix. 14 * 4 values for state * action combinations
q_table.shape
```




    (16, 4)




```python
q_table
```


```python
# 2. Training Policy(Epsilon greedy), Greedy policy(inference policy)
# Greedy policy - Maximum value given a state
def greedy_policy(q_table, state):
  return np.argmax(q_table[state][:]) # Action with max value given the state

# Epsilon policy - Epsilon value is from (0,1)
# Bigger epsilon - random action, lower epsilon - exploitation. epsilon undergoes a decay.
def epsilon_greedy(q_table, state, epsilon):

  random_num = random.uniform(0, 1)

  if random_num > epsilon: # Exploitation
    action = greedy_policy(q_table, state)
  else: # Exploration
    action = env_fl.action_space.sample()

  return action
```


```python
# 3.HyperParameters
# Training parameters
n_training_episodes = 100000 # number of episodes
learning_rate = 0.7 # learning rate determines the value update scale(of Target of TD)

# Eval
n_eval_episodes = 100

# Environment
env_id = "FrozenLake-v1" # Env name
max_step = 99 # max steps per episode,
gamma = 0.95 # Discount
eval_seed = [] # eval seed of the env

# Exploration/Exploitation. Decay should be slow otherwise no exploration(all possible moves) to learn value function
# Then followed by exploitation, we'll be stuck with bad agent.
# So, exponential decay for explore then exploit
max_epsilon = 1.0
min_epsilon = 0.05
decay_rate = 0.0005
```


```python
# 4. Training
# 1. train for n_training_episodes
# decay learning rate every episode exponentially
# Select action with greedy epsilon policy
# Get new state, reward etc with action taken
# Update the policy with bellman equation
# If terminated or truncated stop
# else continue

from tqdm import tqdm
def train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, env, max_step, q_table):

  for episode in tqdm(range(n_training_episodes)):

    # Exponential decay
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
    state, info = env.reset() # Initial step
    terminated = False
    truncated = False
    step = 0

    for step in range(max_step):
      # Select action with epsilon greedy
      action = epsilon_greedy(
          q_table,
          state,
          epsilon=epsilon,
      )

      # Take action
      new_state, reward, terminated, truncated, info = env.step(action)

      # Update Q-Table
      q_table[state][action] = (
          q_table[state][action] + # Estimated value of state
          learning_rate * # determines how much to update the policy
          (
              reward + # reward from action taken
              gamma * np.max(q_table[new_state]) - q_table[state][action] # discounted reward of next state - current estimated value of state.
              )
      )

      if terminated or truncated:
        break

      state = new_state # Update current state.

  return q_table
```


```python
# Train agent
q_optimal_table = train(
    n_training_episodes=n_training_episodes,
    min_epsilon=min_epsilon,
    max_epsilon=max_epsilon,
    decay_rate=decay_rate,
    env=env_fl,
    max_step=max_step,
    q_table=q_table,
)
```


```python
q_optimal_table
```


```python
# Evaluation of Q-Learning agent
# Calculation of maximum cumulative reward
# Iterate for n_episodes
# Take action according to greedy-policy
# Store episode level rewards
# mean of epsiode rewards - cumulative retrurn

def eval_agent(env, max_step, q_table, n_eval_episodes, seed):

  episode_rewards = []
  for episode in tqdm(range(n_eval_episodes)):

    # Take action
    if seed:
      state, info = env.reset(seed=seed[episode])
    else:
      state, info = env.reset() # Initial state

    truncated = False
    terminated = False
    total_rewards_ep = 0

    for step in range(max_step):

      action = greedy_policy(q_table, state) # Selecting action with optimal(trained) Q-Table
      new_state, reward, terminated, truncated, info = env.step(action)
      total_rewards_ep += reward

      if terminated or truncated:
        break

      state = new_state

    episode_rewards.append(total_rewards_ep)

  mean_reward = np.mean(episode_rewards)
  std_reward = np.std(episode_rewards)

  return mean_reward, std_reward
```


```python
# Run eval
mean_reward, std_reward = eval_agent(
    env=env_fl,
    max_step=max_step,
    q_table=q_optimal_table,
    n_eval_episodes=n_eval_episodes,
    seed=eval_seed,
)
```


```python
print(f"Mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")
```


```python
# Let's push this model to hub
from huggingface_hub import notebook_login
notebook_login()
```


    VBox(children=(HTML(value='<center> <img\nsrc=https://huggingface.co/front/assets/huggingface_logo-noborder.sv…



```python
# Next, let's write functions for below functionalites:
# 1. record Video
# 2. Build Model card
# 3. Push to Hub
```


```python
# 1. Record video
record_env = gymnasium.make(
    'FrozenLake-v1', # Env id
    desc=None, # Create custom grid, if needed
    map_name="4x4", # Grid size
    is_slippery=False, # Slippery or Non Slippery Grid
    render_mode="rgb_array",
)

def record_video(env, q_table, out_directory, fps=1):
  """
  Record single frame per second and store the video replay.
  """

  images = []
  terminated, truncated = False, False
  state, info = env.reset(seed=random.randint(0, 500)) # Random initialization of state
  img = env.render()
  images.append(img)

  while not terminated or truncated:
    # Take action
    action = greedy_policy(q_table, state) # Select action
    state, reward, terminated, truncated, info = env.step(action) # Make action
    img = env.render()
    images.append(img)

  imageio.mimsave(out_directory, [np.array(img) for i, img in enumerate(images)], fps=fps)
```


```python
record_video(
    env=record_env,
    q_table=q_optimal_table,
    out_directory="replay.mp4",
    fps=1,
)
```


```python
record_env.spec.kwargs.get('map_name')
```


```python
from pathlib import Path
import pickle
import datetime
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.repocard import metadata_eval_result, metadata_save

def push_to_hub(repo_id, model, env, video_fps=1, local_repo_path="hubn"):

  # Repo details
  _, repo_name = repo_id.split("/")

  eval_env = env
  api = HfApi()

  # 1. create repo
  repo_url = api.create_repo(
      repo_id=repo_id,
      exist_ok=True,
  )

  # 2. Download files
  repo_local_path = Path(snapshot_download(repo_id=repo_id))

  # 3. Save model
  if env.spec.kwargs.get("map_name"): # grid size
    model["map_name"] = env.spec.kwargs.get("map_name")

  if env.spec.kwargs.get("is_slippery", "") == False:
    model["slippery"] = False

  # 4. Pickle the model
  with open((repo_local_path) / "q-learning.pkl", "wb") as f:
    pickle.dump(model, f)

  # 5. Evaluate
  mean_reward, std_reward = eval_agent(
      env=eval_env,
      max_step=model["max_steps"],
      q_table=model["q_table"],
      n_eval_episodes=model["n_eval_episodes"],
      seed=model['eval_seed']
    )

  evaluate_data = {
        "env_id": model["env_id"],
        "mean_reward": mean_reward,
        "n_eval_episodes": model["n_eval_episodes"],
        "eval_datetime": datetime.datetime.now().isoformat(),
    }

  with open(repo_local_path / "results.json", "wb") as f:
    pickle.dump(evaluate_data, f)

  # 6.Create model card

    env_name = model["env_id"]
    if env.spec.kwargs.get("map_name"):
        env_name += "-" + env.spec.kwargs.get("map_name")

    if env.spec.kwargs.get("is_slippery", "") == False:
        env_name += "-" + "no_slippery"

    metadata = {}
    metadata["tags"] = [env_name, "q-learning", "reinforcement-learning", "custom-implementation"]

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
    # **Q-Learning** Agent playing1 **{env_id}**
    This is a trained model of a **Q-Learning** agent playing **{env_id}** .

    ## Usage

    model = load_from_hub(repo_id="{repo_id}", filename="q-learning.pkl")

    # Don't forget to check if you need to add additional attributes (is_slippery=False etc)
    env = gym.make(model["env_id"])
    """
    # eval_agent(env, model["max_steps"], model["n_eval_episodes"], model["qtable"], model["eval_seed"])

    readme_path = repo_local_path / "README.md"
    readme = ""
    print(readme_path.exists())
    if readme_path.exists():
        with readme_path.open("r", encoding="utf8") as f:
            readme = f.read()
    else:
        readme = model_card

    with readme_path.open("w", encoding="utf-8") as f:
        f.write(readme)

    # Save our metrics to Readme metadata
    metadata_save(readme_path, metadata)

    video_path = repo_local_path / "replay.mp4"
    record_video(env, model["q_table"], video_path, video_fps)

    # Step 7. Push everything to the Hub
    api.upload_folder(
        repo_id=repo_id,
        folder_path=repo_local_path,
        path_in_repo=".",
    )
```


```python
# Model
model = {
    "env_id": env_fl.spec.id,
    "max_steps": max_step,
    "n_training_episodes": n_training_episodes,
    "n_eval_episodes": n_eval_episodes,
    "eval_seed": [],
    "learning_rate": learning_rate,
    "gamma": gamma,
    "max_epsilon": max_epsilon,
    "min_epsilon": min_epsilon,
    "decay_rate": decay_rate,
    "q_table":q_optimal_table,
}

push_to_hub("JpChi/FrozenLake", model=model, env=record_env)
```

### Taxi-V3

Training setup and flow all remains same as Frozen Lake, only the complexity of environment is bigger.

We've four destination locations, with 25 possible states for taxi, 5 possible locatios of passenger(including inside taxi). This gives us 500 possible states.

Actions: six possible actions[south, north, east, west, pickup, drop]
Rewards: -1 - per step, +20 - successfuly drop off, -10 - failed pickup or drop off.


```python
# Training parameters
n_training_episodes = 25000  # Total training episodes
learning_rate = 0.7  # Learning rate

# Evaluation parameters
n_eval_episodes = 100  # Total number of test episodes

# DO NOT MODIFY EVAL_SEED
eval_seed = [
    16,
    54,
    165,
    177,
    191,
    191,
    120,
    80,
    149,
    178,
    48,
    38,
    6,
    125,
    174,
    73,
    50,
    172,
    100,
    148,
    146,
    6,
    25,
    40,
    68,
    148,
    49,
    167,
    9,
    97,
    164,
    176,
    61,
    7,
    54,
    55,
    161,
    131,
    184,
    51,
    170,
    12,
    120,
    113,
    95,
    126,
    51,
    98,
    36,
    135,
    54,
    82,
    45,
    95,
    89,
    59,
    95,
    124,
    9,
    113,
    58,
    85,
    51,
    134,
    121,
    169,
    105,
    21,
    30,
    11,
    50,
    65,
    12,
    43,
    82,
    145,
    152,
    97,
    106,
    55,
    31,
    85,
    38,
    112,
    102,
    168,
    123,
    97,
    21,
    83,
    158,
    26,
    80,
    63,
    5,
    81,
    32,
    11,
    28,
    148,
]  # Evaluation seed, this ensures that all classmates agents are trained on the same taxi starting position
# Each seed has a specific starting state

# Environment parameters
env_id = "Taxi-v3"  # Name of the environment
max_steps = 99  # Max steps per episode
gamma = 0.95  # Discounting rate

# Exploration parameters
max_epsilon = 1.0  # Exploration probability at start
min_epsilon = 0.05  # Minimum exploration probability
decay_rate = 0.005  # Exponential decay rate for exploration prob
```


```python
taxi_env = gymnasium.make(
    "Taxi-v3",
    render_mode="rgb_array"
)
```


```python
taxi_env.observation_space.n, taxi_env.action_space.n
```




    (np.int64(500), np.int64(6))




```python
q_table = intialize_q_table(taxi_env.observation_space.n, taxi_env.action_space.n)
```


```python
optimal_q_table = train(
    n_training_episodes=n_training_episodes,
    min_epsilon=min_epsilon,
    max_epsilon=max_epsilon,
    decay_rate=decay_rate,
    env=taxi_env,
    max_step=max_steps,
    q_table=q_table,
)
```

    100%|██████████| 25000/25000 [00:14<00:00, 1763.84it/s]



```python
mean_reward, std_reward = eval_agent(
    env=taxi_env,
    max_step=max_steps,
    q_table=optimal_q_table,
    n_eval_episodes=n_eval_episodes,
    seed=eval_seed,
)
```

    100%|██████████| 100/100 [00:00<00:00, 2136.77it/s]



```python
mean_reward
```




    np.float64(7.54)




```python
# Let's update the hyperparameters
n_training_episodes = 50000
```


```python
optimal_q_table_1 = train(
    n_training_episodes=n_training_episodes,
    min_epsilon=min_epsilon,
    max_epsilon=max_epsilon,
    decay_rate=decay_rate,
    env=taxi_env,
    max_step=max_steps,
    q_table=q_table,
)
```

    100%|██████████| 50000/50000 [00:26<00:00, 1900.28it/s]



```python
mean_reward, std_reward = eval_agent(
    env=taxi_env,
    max_step=max_steps,
    q_table=optimal_q_table_1,
    n_eval_episodes=n_eval_episodes,
    seed=eval_seed,
)
```

    100%|██████████| 100/100 [00:00<00:00, 3265.50it/s]



```python
mean_reward
```




    np.float64(7.56)




```python
# Increase max_steps per episode
max_steps = 499
```


```python
optimal_q_table_2 = train(
    n_training_episodes=n_training_episodes,
    min_epsilon=min_epsilon,
    max_epsilon=max_epsilon,
    decay_rate=decay_rate,
    env=taxi_env,
    max_step=max_steps,
    q_table=q_table,
)
```

    100%|██████████| 50000/50000 [00:27<00:00, 1811.34it/s]



```python
mean_reward, std_reward = eval_agent(
    env=taxi_env,
    max_step=max_steps,
    q_table=optimal_q_table_2,
    n_eval_episodes=n_eval_episodes,
    seed=eval_seed,
)
```

    100%|██████████| 100/100 [00:00<00:00, 3906.29it/s]



```python
mean_reward
```




    np.float64(7.56)




```python
# Let's do more exploration
decay_rate = 0.0005
```


```python
optimal_q_table_3 = train(
    n_training_episodes=n_training_episodes,
    min_epsilon=min_epsilon,
    max_epsilon=max_epsilon,
    decay_rate=decay_rate,
    env=taxi_env,
    max_step=max_steps,
    q_table=q_table,
)
```

    100%|██████████| 50000/50000 [00:30<00:00, 1615.61it/s]



```python
mean_reward, std_reward = eval_agent(
    env=taxi_env,
    max_step=max_steps,
    q_table=optimal_q_table_3,
    n_eval_episodes=n_eval_episodes,
    seed=eval_seed,
)
```

    100%|██████████| 100/100 [00:00<00:00, 2016.91it/s]



```python
mean_reward
```




    np.float64(7.56)




```python
n_training_episodes = 1000000
```


```python
optimal_q_table_4 = train(
    n_training_episodes=n_training_episodes,
    min_epsilon=min_epsilon,
    max_epsilon=max_epsilon,
    decay_rate=decay_rate,
    env=taxi_env,
    max_step=max_steps,
    q_table=q_table,
)
```

    100%|██████████| 1000000/1000000 [08:26<00:00, 1972.42it/s]



```python
mean_reward, std_reward = eval_agent(
    env=taxi_env,
    max_step=max_steps,
    q_table=optimal_q_table_4,
    n_eval_episodes=n_eval_episodes,
    seed=eval_seed,
)
```

    100%|██████████| 100/100 [00:00<00:00, 4077.21it/s]



```python
mean_reward
```




    np.float64(7.56)



Even with 20x steps, mean reward remains at 7.56. Probably requires more hyperparams optimization or a different policy.


```python
env_id = "Taxi-v3"
model = {
    "env_id": env_id,
    "max_steps": max_steps,
    "n_training_episodes": n_training_episodes,
    "n_eval_episodes": n_eval_episodes,
    "eval_seed": eval_seed,
    "learning_rate": learning_rate,
    "gamma": gamma,
    "max_epsilon": max_epsilon,
    "min_epsilon": min_epsilon,
    "decay_rate": decay_rate,
    "q_table":optimal_q_table_2,
}

push_to_hub(
    "JpChi/Taxi-v3",
    model=model,
    env=taxi_env,
    video_fps=1
)
```


    Fetching 1 files:   0%|          | 0/1 [00:00<?, ?it/s]


    100%|██████████| 100/100 [00:00<00:00, 3892.08it/s]


    False


    WARNING:imageio_ffmpeg:IMAGEIO FFMPEG_WRITER WARNING: input image is not divisible by macro_block_size=16, resizing from (550, 350) to (560, 352) to ensure video compatibility with most codecs and players. To prevent resizing, make your input image divisible by the macro_block_size or set the macro_block_size to 1 (risking incompatibility).



    Upload 2 LFS files:   0%|          | 0/2 [00:00<?, ?it/s]



    q-learning.pkl:   0%|          | 0.00/24.6k [00:00<?, ?B/s]



    replay.mp4:   0%|          | 0.00/119k [00:00<?, ?B/s]



```python

```
