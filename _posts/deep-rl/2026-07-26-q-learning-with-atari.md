# Deep Q-Learning with Atari Games

In Frozen-Lake, Taxi. The states are discrete and limited. Atari games state space may contain $10^9$ to $10^{11}$ states. In this notebook, we'll cover the difficulty of training a Q-table and train a Deep Q-Learning network that takes a state and returns an estimated Q-Value.

Q-Learning is a tabular method, this is not scalable, if states and actions spaces are not small enough to be represented by arrays and tables.

Let's compare the state space difference:
1. FrozenLake: 16 states.
2. Taxi-v3: 500 states.
3. Atari(Frames as input)
    - (210, 160, 3) RGB
    - each pixel value 0-255.
    - $256^{210x160x3} = 256^{100800}$
    - This is really large than $10^8$ atoms in the observable universe.

With this gigantic state space, Q-Table is not efficient. Deep Q Learning, accepts a state and approximates a Q Value for all possible actions.

## Deep-Q Network

1. Inputs:
    - To reduce complexity of our state space and training time. Atari frames are preprocessed as follows:
    - Images are grayscaled to 84 x 84.
    - 4 frames are stacked together to handle temporal limitation.(Temporal limitation - No sense of motion, without sense of motion, Network can't learn the patterns).
2. Network:
    - CNN: To retain spatial information in frames and learn the patterns to approximat actions.
    - Fully Connected Layers
3. Outputs:
    - Three actions:
      - LEFT
      - RIGHT
      - SHOOT

## Deep Q-Learning Algorithm

Q-Learning:- In Training, directly update state-action q value in Q-Table.
Deep Q-Learning: As in any deep learning neural networks, we'll calculate a loss and update weights of Deep Q-Network using gradient descent.
Deep Q-Learning Loss:

```text
[Immediate reward(for an action) + Discounted estimate of optimal Q-Value] - [Former Q-Value estimation]
```

### Training algorithm

Phase 1 (Sampling): Take actions and store expreience tuples in replay memory. Experience tuple - (Current state, action taken, immediate reward, next state).
Phase 2 (Training): Select, use a small batch of expreience(random) and learn from this batch using gradint descent.

Unlike Q-Learning, Deep Q-Learning is suceptible to instability(as with all neural networks). Main factors of instability are:
1. Not fixed targets - Target value is immediate reward + Estimaged Q-Value of next state.
2. In combination with neural network.

Ways to avoid instability:


#### Experience Replay

1. Use replay to store te experience and *reuse* them. In online reinforcement learning we use an experiecne to learn and discard it. This is a waste. Allows agents to learn muliple times from same experience. Breaks correlation, ex: learn only to move left, if sequential experience of moving left is used, a combination of moves(left, right, jump) etc has to be used.
2. Catastrophic forgetting: This is equivavlent to why we shuffle datapoints in deep learning. If we provide sequential expereinces level1, level2. When it reaches level2, agent might forget how to play in first level. To avoid this we'll sample randoms samples from replay buffer to provide a good distribution and prevents network from only learning from it's immediate experience.

#### Fixed Q-Target to stabilize the training

Problem:

- Loss is TD Target minus Former estimated Q-value of the state.
- TD target is not fixed, it's sum of immediate reward and estimated optimal Q-Value of next state. Q-Value is what we're trying to approximate.
- Significant correlation between weights and TD Target.
- As we optimize the network for a loss, respective TD target gets updated as well.
- We'll not reach a minima as the target keeps getting updated by the network.

Fix:

- Use a seperate weights(Q-dash) with fixed (for C steps alone). This is for target alone.
- Perform gradient descent on main weights(Q).
- copy parameters from the DQN(main network) every C steps.
- By doing this we've a fixed target as well as the values get updated as well.

#### Double DQN

Problem:

- We select the action with maximum Q-Value.
- At training start, this maximum Q-Value is noise, if we continuously use the noise actions, Q-value will be higher for non-optimal actions and learning will be compolicated.

Fix:

- Select max Q-Value action during sampling.(DQN - main network).
- In Training(target network/ weights): Calculate the Target Q-Value for taking that action.
- By decoupling action taken and Target calculation, we reduce the overstimation bias by providing higher Q-Values for non-optimal actions.

## HandsOn

### Libraries

1. RL-Baselines-3 Zoo integation - vanilla version of Deep Q-Learning.
2. gymnasium
3. pyvirtual display


```python
!apt-get install swig cmake ffmpeg
!apt install python-opengl
!apt install xvfb
```

    Reading package lists... Done
    Building dependency tree... Done
    Reading state information... Done
    cmake is already the newest version (3.22.1-1ubuntu1.22.04.2).
    ffmpeg is already the newest version (7:4.4.2-0ubuntu0.22.04.1).
    The following additional packages will be installed:
      swig4.0
    Suggested packages:
      swig-doc swig-examples swig4.0-examples swig4.0-doc
    The following NEW packages will be installed:
      swig swig4.0
    0 upgraded, 2 newly installed, 0 to remove and 35 not upgraded.
    Need to get 1,116 kB of archives.
    After this operation, 5,542 kB of additional disk space will be used.
    Get:1 http://archive.ubuntu.com/ubuntu jammy/universe amd64 swig4.0 amd64 4.0.2-1ubuntu1 [1,110 kB]
    Get:2 http://archive.ubuntu.com/ubuntu jammy/universe amd64 swig all 4.0.2-1ubuntu1 [5,632 B]
    Fetched 1,116 kB in 2s (601 kB/s)
    Selecting previously unselected package swig4.0.
    (Reading database ... 126371 files and directories currently installed.)
    Preparing to unpack .../swig4.0_4.0.2-1ubuntu1_amd64.deb ...
    Unpacking swig4.0 (4.0.2-1ubuntu1) ...
    Selecting previously unselected package swig.
    Preparing to unpack .../swig_4.0.2-1ubuntu1_all.deb ...
    Unpacking swig (4.0.2-1ubuntu1) ...
    Setting up swig4.0 (4.0.2-1ubuntu1) ...
    Setting up swig (4.0.2-1ubuntu1) ...
    Processing triggers for man-db (2.10.2-1) ...
    Reading package lists... Done
    Building dependency tree... Done
    Reading state information... Done
    [1;31mE: [0mUnable to locate package python-opengl[0m
    Reading package lists... Done
    Building dependency tree... Done
    Reading state information... Done
    xvfb is already the newest version (2:21.1.4-2ubuntu1.7~22.04.15).
    0 upgraded, 0 newly installed, 0 to remove and 35 not upgraded.



```python
!pip install git+https://github.com/DLR-RM/rl-baselines3-zoo -q
!pip install gymnasium[atari] gymnasium[accept-rom-license] -q
!pip3 install pyvirtualdisplay -q
```

      Installing build dependencies ... [?25l[?25hdone
      Getting requirements to build wheel ... [?25l[?25hdone
      Preparing metadata (pyproject.toml) ... [?25l[?25hdone
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m400.9/400.9 kB[0m [31m31.6 MB/s[0m eta [36m0:00:00[0m
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m91.1/91.1 kB[0m [31m8.7 MB/s[0m eta [36m0:00:00[0m
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m93.2/93.2 kB[0m [31m10.0 MB/s[0m eta [36m0:00:00[0m
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m247.4/247.4 kB[0m [31m22.9 MB/s[0m eta [36m0:00:00[0m
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m187.2/187.2 kB[0m [31m17.2 MB/s[0m eta [36m0:00:00[0m
    [?25h  Building wheel for rl_zoo3 (pyproject.toml) ... [?25l[?25hdone
    [33mWARNING: gymnasium 1.2.0 does not provide the extra 'accept-rom-license'[0m[33m
    [0m


```python
# Virtual display
from pyvirtualdisplay import Display

virtual_display = Display(visible=0, size=(1400, 900))
virtual_display.start()
```




    <pyvirtualdisplay.display.Display at 0x78647f8654c0>



RL-Baselines3 requires only an yaml file of hyperparameters to train DQN or any deep-rl algorithm on gymnasium environments.

Yaml params:

```yaml
SpaceInvadersNoFrameskip-v4:
  env_wrapper:
    - stable_baselines3.common.atari_wrappers.AtariWrapper # Atari wrapper, preprocess atari frames
  frame_stack: 4 # 4 grayscale frames stacked
  policy: 'CnnPolicy' # CNN to process stacks
  n_timesteps: !!float 1e6 # Train for 1M env steps
  buffer_size: 100000 # Replay buffer size
  learning_rate: !!float 1e-4 # Learning rate initial
  batch_size: 32
  learning_starts: 100000 # Number of steps before learning, warmup steps
  target_update_interval: 1000 # C steps Q copied to Q_hat
  train_freq: 4 # Trains the model every 4 steps.
  gradient_steps: 1
  exploration_fraction: 0.1
  exploration_final_eps: 0.01
  # If True, you need to deactivate handle_timeout_termination
  # in the replay_buffer_kwargs
  optimize_memory_usage: False
```


```python
# Train
!python -m rl_zoo3.train --algo dqn --env SpaceInvadersNoFrameskip-v4 -f logs/ -c params.yaml
```

    [1;30;43mStreaming output truncated to the last 5000 lines.[0m
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.25e+03 |
    |    ep_rew_mean      | 406      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 2744     |
    |    fps              | 250      |
    |    time_elapsed     | 2239     |
    |    total_timesteps  | 560055   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0262   |
    |    n_updates        | 115013   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.29e+03 |
    |    ep_rew_mean      | 412      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 2748     |
    |    fps              | 250      |
    |    time_elapsed     | 2243     |
    |    total_timesteps  | 561077   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0286   |
    |    n_updates        | 115269   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.28e+03 |
    |    ep_rew_mean      | 410      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 2752     |
    |    fps              | 250      |
    |    time_elapsed     | 2246     |
    |    total_timesteps  | 561760   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0118   |
    |    n_updates        | 115439   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.26e+03 |
    |    ep_rew_mean      | 408      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 2756     |
    |    fps              | 250      |
    |    time_elapsed     | 2249     |
    |    total_timesteps  | 562617   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.00936  |
    |    n_updates        | 115654   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.26e+03 |
    |    ep_rew_mean      | 407      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 2760     |
    |    fps              | 250      |
    |    time_elapsed     | 2251     |
    |    total_timesteps  | 563063   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.021    |
    |    n_updates        | 115765   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.25e+03 |
    |    ep_rew_mean      | 406      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 2764     |
    |    fps              | 250      |
    |    time_elapsed     | 2253     |
    |    total_timesteps  | 563457   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0663   |
    |    n_updates        | 115864   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.26e+03 |
    |    ep_rew_mean      | 408      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 2768     |
    |    fps              | 250      |
    |    time_elapsed     | 2258     |
    |    total_timesteps  | 564748   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0111   |
    |    n_updates        | 116186   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.25e+03 |
    |    ep_rew_mean      | 407      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 2772     |
    |    fps              | 250      |
    |    time_elapsed     | 2262     |
    |    total_timesteps  | 565859   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.104    |
    |    n_updates        | 116464   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.24e+03 |
    |    ep_rew_mean      | 406      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 2776     |
    |    fps              | 250      |
    |    time_elapsed     | 2265     |
    |    total_timesteps  | 566382   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0212   |
    |    n_updates        | 116595   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.23e+03 |
    |    ep_rew_mean      | 403      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 2780     |
    |    fps              | 250      |
    |    time_elapsed     | 2272     |
    |    total_timesteps  | 568184   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0459   |
    |    n_updates        | 117045   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.25e+03 |
    |    ep_rew_mean      | 407      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 2784     |
    |    fps              | 250      |
    |    time_elapsed     | 2276     |
    |    total_timesteps  | 569231   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0254   |
    |    n_updates        | 117307   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.28e+03 |
    |    ep_rew_mean      | 412      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 2788     |
    |    fps              | 250      |
    |    time_elapsed     | 2282     |
    |    total_timesteps  | 570694   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0265   |
    |    n_updates        | 117673   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.31e+03 |
    |    ep_rew_mean      | 423      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 2792     |
    |    fps              | 250      |
    |    time_elapsed     | 2288     |
    |    total_timesteps  | 572263   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.00682  |
    |    n_updates        | 118065   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.32e+03 |
    |    ep_rew_mean      | 423      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 2796     |
    |    fps              | 249      |
    |    time_elapsed     | 2293     |
    |    total_timesteps  | 573283   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0126   |
    |    n_updates        | 118320   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.32e+03 |
    |    ep_rew_mean      | 421      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 2800     |
    |    fps              | 250      |
    |    time_elapsed     | 2297     |
    |    total_timesteps  | 574396   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0201   |
    |    n_updates        | 118598   |
    ----------------------------------
    Eval num_timesteps=575000, episode_reward=452.00 +/- 112.19
    Episode length: 3810.20 +/- 708.57
    ----------------------------------
    | eval/               |          |
    |    mean_ep_length   | 3.81e+03 |
    |    mean_reward      | 452      |
    | rollout/            |          |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    total_timesteps  | 575000   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.028    |
    |    n_updates        | 118749   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.27e+03 |
    |    ep_rew_mean      | 413      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 2804     |
    |    fps              | 248      |
    |    time_elapsed     | 2313     |
    |    total_timesteps  | 575408   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0174   |
    |    n_updates        | 118851   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.27e+03 |
    |    ep_rew_mean      | 414      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 2808     |
    |    fps              | 248      |
    |    time_elapsed     | 2318     |
    |    total_timesteps  | 576643   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0148   |
    |    n_updates        | 119160   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.26e+03 |
    |    ep_rew_mean      | 414      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 2812     |
    |    fps              | 248      |
    |    time_elapsed     | 2323     |
    |    total_timesteps  | 578025   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.00883  |
    |    n_updates        | 119506   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.26e+03 |
    |    ep_rew_mean      | 414      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 2816     |
    |    fps              | 248      |
    |    time_elapsed     | 2328     |
    |    total_timesteps  | 579015   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0131   |
    |    n_updates        | 119753   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.27e+03 |
    |    ep_rew_mean      | 413      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 2820     |
    |    fps              | 248      |
    |    time_elapsed     | 2334     |
    |    total_timesteps  | 580560   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0248   |
    |    n_updates        | 120139   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.25e+03 |
    |    ep_rew_mean      | 412      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 2824     |
    |    fps              | 248      |
    |    time_elapsed     | 2335     |
    |    total_timesteps  | 580959   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0275   |
    |    n_updates        | 120239   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.25e+03 |
    |    ep_rew_mean      | 415      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 2828     |
    |    fps              | 248      |
    |    time_elapsed     | 2342     |
    |    total_timesteps  | 582569   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0321   |
    |    n_updates        | 120642   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.24e+03 |
    |    ep_rew_mean      | 413      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 2832     |
    |    fps              | 248      |
    |    time_elapsed     | 2347     |
    |    total_timesteps  | 583928   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0131   |
    |    n_updates        | 120981   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.24e+03 |
    |    ep_rew_mean      | 415      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 2836     |
    |    fps              | 248      |
    |    time_elapsed     | 2352     |
    |    total_timesteps  | 585000   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0239   |
    |    n_updates        | 121249   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.24e+03 |
    |    ep_rew_mean      | 417      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 2840     |
    |    fps              | 248      |
    |    time_elapsed     | 2356     |
    |    total_timesteps  | 586017   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.00541  |
    |    n_updates        | 121504   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.23e+03 |
    |    ep_rew_mean      | 413      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 2844     |
    |    fps              | 248      |
    |    time_elapsed     | 2362     |
    |    total_timesteps  | 587539   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0283   |
    |    n_updates        | 121884   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.25e+03 |
    |    ep_rew_mean      | 417      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 2848     |
    |    fps              | 248      |
    |    time_elapsed     | 2368     |
    |    total_timesteps  | 588959   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.019    |
    |    n_updates        | 122239   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.26e+03 |
    |    ep_rew_mean      | 417      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 2852     |
    |    fps              | 248      |
    |    time_elapsed     | 2372     |
    |    total_timesteps  | 590165   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0133   |
    |    n_updates        | 122541   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.25e+03 |
    |    ep_rew_mean      | 415      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 2856     |
    |    fps              | 248      |
    |    time_elapsed     | 2378     |
    |    total_timesteps  | 591428   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0374   |
    |    n_updates        | 122856   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.27e+03 |
    |    ep_rew_mean      | 418      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 2860     |
    |    fps              | 248      |
    |    time_elapsed     | 2382     |
    |    total_timesteps  | 592509   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0239   |
    |    n_updates        | 123127   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.27e+03 |
    |    ep_rew_mean      | 418      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 2864     |
    |    fps              | 248      |
    |    time_elapsed     | 2386     |
    |    total_timesteps  | 593578   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0379   |
    |    n_updates        | 123394   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.26e+03 |
    |    ep_rew_mean      | 415      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 2868     |
    |    fps              | 248      |
    |    time_elapsed     | 2389     |
    |    total_timesteps  | 594236   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0163   |
    |    n_updates        | 123558   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.27e+03 |
    |    ep_rew_mean      | 420      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 2872     |
    |    fps              | 248      |
    |    time_elapsed     | 2394     |
    |    total_timesteps  | 595537   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0166   |
    |    n_updates        | 123884   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.26e+03 |
    |    ep_rew_mean      | 419      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 2876     |
    |    fps              | 248      |
    |    time_elapsed     | 2403     |
    |    total_timesteps  | 597670   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.018    |
    |    n_updates        | 124417   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.27e+03 |
    |    ep_rew_mean      | 419      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 2880     |
    |    fps              | 248      |
    |    time_elapsed     | 2408     |
    |    total_timesteps  | 599028   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0119   |
    |    n_updates        | 124756   |
    ----------------------------------
    Eval num_timesteps=600000, episode_reward=532.00 +/- 106.14
    Episode length: 3853.80 +/- 602.37
    ----------------------------------
    | eval/               |          |
    |    mean_ep_length   | 3.85e+03 |
    |    mean_reward      | 532      |
    | rollout/            |          |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    total_timesteps  | 600000   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0481   |
    |    n_updates        | 124999   |
    ----------------------------------
    New best mean reward!
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.26e+03 |
    |    ep_rew_mean      | 416      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 2884     |
    |    fps              | 247      |
    |    time_elapsed     | 2423     |
    |    total_timesteps  | 600162   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0139   |
    |    n_updates        | 125040   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.26e+03 |
    |    ep_rew_mean      | 412      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 2888     |
    |    fps              | 247      |
    |    time_elapsed     | 2428     |
    |    total_timesteps  | 601394   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0472   |
    |    n_updates        | 125348   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.28e+03 |
    |    ep_rew_mean      | 417      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 2892     |
    |    fps              | 247      |
    |    time_elapsed     | 2433     |
    |    total_timesteps  | 602744   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.026    |
    |    n_updates        | 125685   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.26e+03 |
    |    ep_rew_mean      | 415      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 2896     |
    |    fps              | 247      |
    |    time_elapsed     | 2439     |
    |    total_timesteps  | 604120   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0224   |
    |    n_updates        | 126029   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.27e+03 |
    |    ep_rew_mean      | 414      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 2900     |
    |    fps              | 247      |
    |    time_elapsed     | 2445     |
    |    total_timesteps  | 605533   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0256   |
    |    n_updates        | 126383   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.28e+03 |
    |    ep_rew_mean      | 416      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 2904     |
    |    fps              | 247      |
    |    time_elapsed     | 2451     |
    |    total_timesteps  | 607120   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0155   |
    |    n_updates        | 126779   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.3e+03  |
    |    ep_rew_mean      | 419      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 2908     |
    |    fps              | 247      |
    |    time_elapsed     | 2456     |
    |    total_timesteps  | 608422   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0315   |
    |    n_updates        | 127105   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.34e+03 |
    |    ep_rew_mean      | 421      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 2912     |
    |    fps              | 247      |
    |    time_elapsed     | 2461     |
    |    total_timesteps  | 609688   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0418   |
    |    n_updates        | 127421   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.35e+03 |
    |    ep_rew_mean      | 425      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 2916     |
    |    fps              | 247      |
    |    time_elapsed     | 2468     |
    |    total_timesteps  | 611325   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0163   |
    |    n_updates        | 127831   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.41e+03 |
    |    ep_rew_mean      | 434      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 2920     |
    |    fps              | 247      |
    |    time_elapsed     | 2479     |
    |    total_timesteps  | 614015   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.024    |
    |    n_updates        | 128503   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.42e+03 |
    |    ep_rew_mean      | 435      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 2924     |
    |    fps              | 247      |
    |    time_elapsed     | 2483     |
    |    total_timesteps  | 615218   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0463   |
    |    n_updates        | 128804   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.4e+03  |
    |    ep_rew_mean      | 435      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 2928     |
    |    fps              | 247      |
    |    time_elapsed     | 2490     |
    |    total_timesteps  | 616686   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0262   |
    |    n_updates        | 129171   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.43e+03 |
    |    ep_rew_mean      | 436      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 2932     |
    |    fps              | 247      |
    |    time_elapsed     | 2494     |
    |    total_timesteps  | 617821   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0332   |
    |    n_updates        | 129455   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.44e+03 |
    |    ep_rew_mean      | 441      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 2936     |
    |    fps              | 247      |
    |    time_elapsed     | 2498     |
    |    total_timesteps  | 618791   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0386   |
    |    n_updates        | 129697   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.44e+03 |
    |    ep_rew_mean      | 444      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 2940     |
    |    fps              | 247      |
    |    time_elapsed     | 2502     |
    |    total_timesteps  | 619771   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0239   |
    |    n_updates        | 129942   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.43e+03 |
    |    ep_rew_mean      | 441      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 2944     |
    |    fps              | 247      |
    |    time_elapsed     | 2506     |
    |    total_timesteps  | 620692   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.023    |
    |    n_updates        | 130172   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.43e+03 |
    |    ep_rew_mean      | 441      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 2948     |
    |    fps              | 247      |
    |    time_elapsed     | 2511     |
    |    total_timesteps  | 622063   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0104   |
    |    n_updates        | 130515   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.42e+03 |
    |    ep_rew_mean      | 441      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 2952     |
    |    fps              | 247      |
    |    time_elapsed     | 2516     |
    |    total_timesteps  | 623266   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0288   |
    |    n_updates        | 130816   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.42e+03 |
    |    ep_rew_mean      | 439      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 2956     |
    |    fps              | 247      |
    |    time_elapsed     | 2520     |
    |    total_timesteps  | 624376   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0296   |
    |    n_updates        | 131093   |
    ----------------------------------
    Eval num_timesteps=625000, episode_reward=528.00 +/- 145.14
    Episode length: 3671.40 +/- 467.99
    ----------------------------------
    | eval/               |          |
    |    mean_ep_length   | 3.67e+03 |
    |    mean_reward      | 528      |
    | rollout/            |          |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    total_timesteps  | 625000   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.00524  |
    |    n_updates        | 131249   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.42e+03 |
    |    ep_rew_mean      | 438      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 2960     |
    |    fps              | 246      |
    |    time_elapsed     | 2538     |
    |    total_timesteps  | 626051   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0163   |
    |    n_updates        | 131512   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.44e+03 |
    |    ep_rew_mean      | 438      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 2964     |
    |    fps              | 246      |
    |    time_elapsed     | 2542     |
    |    total_timesteps  | 627206   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0209   |
    |    n_updates        | 131801   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.45e+03 |
    |    ep_rew_mean      | 440      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 2968     |
    |    fps              | 246      |
    |    time_elapsed     | 2547     |
    |    total_timesteps  | 628370   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0196   |
    |    n_updates        | 132092   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.43e+03 |
    |    ep_rew_mean      | 438      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 2972     |
    |    fps              | 246      |
    |    time_elapsed     | 2554     |
    |    total_timesteps  | 629894   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0314   |
    |    n_updates        | 132473   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.43e+03 |
    |    ep_rew_mean      | 440      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 2976     |
    |    fps              | 246      |
    |    time_elapsed     | 2557     |
    |    total_timesteps  | 630644   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0282   |
    |    n_updates        | 132660   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.43e+03 |
    |    ep_rew_mean      | 446      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 2980     |
    |    fps              | 246      |
    |    time_elapsed     | 2563     |
    |    total_timesteps  | 632176   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.00971  |
    |    n_updates        | 133043   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.44e+03 |
    |    ep_rew_mean      | 447      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 2984     |
    |    fps              | 246      |
    |    time_elapsed     | 2569     |
    |    total_timesteps  | 633738   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0186   |
    |    n_updates        | 133434   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.46e+03 |
    |    ep_rew_mean      | 449      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 2988     |
    |    fps              | 246      |
    |    time_elapsed     | 2580     |
    |    total_timesteps  | 636401   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0242   |
    |    n_updates        | 134100   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.5e+03  |
    |    ep_rew_mean      | 460      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 2992     |
    |    fps              | 246      |
    |    time_elapsed     | 2587     |
    |    total_timesteps  | 637970   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0304   |
    |    n_updates        | 134492   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.52e+03 |
    |    ep_rew_mean      | 462      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 2996     |
    |    fps              | 246      |
    |    time_elapsed     | 2592     |
    |    total_timesteps  | 639397   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.00516  |
    |    n_updates        | 134849   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.54e+03 |
    |    ep_rew_mean      | 463      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3000     |
    |    fps              | 246      |
    |    time_elapsed     | 2600     |
    |    total_timesteps  | 641173   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0291   |
    |    n_updates        | 135293   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.55e+03 |
    |    ep_rew_mean      | 465      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3004     |
    |    fps              | 246      |
    |    time_elapsed     | 2609     |
    |    total_timesteps  | 643412   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0102   |
    |    n_updates        | 135852   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.6e+03  |
    |    ep_rew_mean      | 473      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3008     |
    |    fps              | 246      |
    |    time_elapsed     | 2615     |
    |    total_timesteps  | 644988   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.034    |
    |    n_updates        | 136246   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.64e+03 |
    |    ep_rew_mean      | 477      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3012     |
    |    fps              | 246      |
    |    time_elapsed     | 2624     |
    |    total_timesteps  | 647178   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0143   |
    |    n_updates        | 136794   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.65e+03 |
    |    ep_rew_mean      | 479      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3016     |
    |    fps              | 246      |
    |    time_elapsed     | 2630     |
    |    total_timesteps  | 648632   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0481   |
    |    n_updates        | 137157   |
    ----------------------------------
    Eval num_timesteps=650000, episode_reward=629.00 +/- 106.88
    Episode length: 3753.20 +/- 1094.54
    ----------------------------------
    | eval/               |          |
    |    mean_ep_length   | 3.75e+03 |
    |    mean_reward      | 629      |
    | rollout/            |          |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    total_timesteps  | 650000   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0373   |
    |    n_updates        | 137499   |
    ----------------------------------
    New best mean reward!
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.63e+03 |
    |    ep_rew_mean      | 477      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3020     |
    |    fps              | 245      |
    |    time_elapsed     | 2650     |
    |    total_timesteps  | 650705   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0319   |
    |    n_updates        | 137676   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.64e+03 |
    |    ep_rew_mean      | 480      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3024     |
    |    fps              | 245      |
    |    time_elapsed     | 2656     |
    |    total_timesteps  | 652335   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0255   |
    |    n_updates        | 138083   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.67e+03 |
    |    ep_rew_mean      | 484      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3028     |
    |    fps              | 245      |
    |    time_elapsed     | 2661     |
    |    total_timesteps  | 653475   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0493   |
    |    n_updates        | 138368   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.71e+03 |
    |    ep_rew_mean      | 495      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3032     |
    |    fps              | 245      |
    |    time_elapsed     | 2669     |
    |    total_timesteps  | 655500   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0127   |
    |    n_updates        | 138874   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.71e+03 |
    |    ep_rew_mean      | 495      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3036     |
    |    fps              | 245      |
    |    time_elapsed     | 2673     |
    |    total_timesteps  | 656451   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0521   |
    |    n_updates        | 139112   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.71e+03 |
    |    ep_rew_mean      | 499      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3040     |
    |    fps              | 245      |
    |    time_elapsed     | 2680     |
    |    total_timesteps  | 658236   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0422   |
    |    n_updates        | 139558   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.74e+03 |
    |    ep_rew_mean      | 508      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3044     |
    |    fps              | 245      |
    |    time_elapsed     | 2684     |
    |    total_timesteps  | 659208   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0233   |
    |    n_updates        | 139801   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.75e+03 |
    |    ep_rew_mean      | 510      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3048     |
    |    fps              | 245      |
    |    time_elapsed     | 2691     |
    |    total_timesteps  | 660761   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0275   |
    |    n_updates        | 140190   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.75e+03 |
    |    ep_rew_mean      | 510      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3052     |
    |    fps              | 245      |
    |    time_elapsed     | 2695     |
    |    total_timesteps  | 661959   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0239   |
    |    n_updates        | 140489   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.75e+03 |
    |    ep_rew_mean      | 514      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3056     |
    |    fps              | 245      |
    |    time_elapsed     | 2700     |
    |    total_timesteps  | 663059   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.00583  |
    |    n_updates        | 140764   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.75e+03 |
    |    ep_rew_mean      | 511      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3060     |
    |    fps              | 245      |
    |    time_elapsed     | 2705     |
    |    total_timesteps  | 664221   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0246   |
    |    n_updates        | 141055   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.74e+03 |
    |    ep_rew_mean      | 508      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3064     |
    |    fps              | 245      |
    |    time_elapsed     | 2711     |
    |    total_timesteps  | 665600   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0215   |
    |    n_updates        | 141399   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.75e+03 |
    |    ep_rew_mean      | 513      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3068     |
    |    fps              | 245      |
    |    time_elapsed     | 2715     |
    |    total_timesteps  | 666628   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0232   |
    |    n_updates        | 141656   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.77e+03 |
    |    ep_rew_mean      | 515      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3072     |
    |    fps              | 245      |
    |    time_elapsed     | 2722     |
    |    total_timesteps  | 668414   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0396   |
    |    n_updates        | 142103   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.8e+03  |
    |    ep_rew_mean      | 520      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3076     |
    |    fps              | 245      |
    |    time_elapsed     | 2729     |
    |    total_timesteps  | 670174   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0292   |
    |    n_updates        | 142543   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.8e+03  |
    |    ep_rew_mean      | 520      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3080     |
    |    fps              | 245      |
    |    time_elapsed     | 2732     |
    |    total_timesteps  | 670797   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0132   |
    |    n_updates        | 142699   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.78e+03 |
    |    ep_rew_mean      | 516      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3084     |
    |    fps              | 245      |
    |    time_elapsed     | 2736     |
    |    total_timesteps  | 671823   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0144   |
    |    n_updates        | 142955   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.8e+03  |
    |    ep_rew_mean      | 519      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3088     |
    |    fps              | 245      |
    |    time_elapsed     | 2742     |
    |    total_timesteps  | 673213   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0302   |
    |    n_updates        | 143303   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.79e+03 |
    |    ep_rew_mean      | 519      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3092     |
    |    fps              | 245      |
    |    time_elapsed     | 2746     |
    |    total_timesteps  | 674290   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0619   |
    |    n_updates        | 143572   |
    ----------------------------------
    Eval num_timesteps=675000, episode_reward=496.00 +/- 201.28
    Episode length: 3621.20 +/- 884.19
    ----------------------------------
    | eval/               |          |
    |    mean_ep_length   | 3.62e+03 |
    |    mean_reward      | 496      |
    | rollout/            |          |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    total_timesteps  | 675000   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.00817  |
    |    n_updates        | 143749   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.82e+03 |
    |    ep_rew_mean      | 523      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3096     |
    |    fps              | 244      |
    |    time_elapsed     | 2766     |
    |    total_timesteps  | 676562   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0304   |
    |    n_updates        | 144140   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.83e+03 |
    |    ep_rew_mean      | 523      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3100     |
    |    fps              | 244      |
    |    time_elapsed     | 2770     |
    |    total_timesteps  | 677604   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0171   |
    |    n_updates        | 144400   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.86e+03 |
    |    ep_rew_mean      | 528      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3104     |
    |    fps              | 244      |
    |    time_elapsed     | 2775     |
    |    total_timesteps  | 678694   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0407   |
    |    n_updates        | 144673   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.88e+03 |
    |    ep_rew_mean      | 530      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3108     |
    |    fps              | 244      |
    |    time_elapsed     | 2782     |
    |    total_timesteps  | 680386   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.00621  |
    |    n_updates        | 145096   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.89e+03 |
    |    ep_rew_mean      | 530      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3112     |
    |    fps              | 244      |
    |    time_elapsed     | 2788     |
    |    total_timesteps  | 681774   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0432   |
    |    n_updates        | 145443   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.89e+03 |
    |    ep_rew_mean      | 533      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3116     |
    |    fps              | 244      |
    |    time_elapsed     | 2793     |
    |    total_timesteps  | 683095   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.031    |
    |    n_updates        | 145773   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.9e+03  |
    |    ep_rew_mean      | 535      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3120     |
    |    fps              | 244      |
    |    time_elapsed     | 2802     |
    |    total_timesteps  | 685238   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0334   |
    |    n_updates        | 146309   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.9e+03  |
    |    ep_rew_mean      | 535      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3124     |
    |    fps              | 244      |
    |    time_elapsed     | 2805     |
    |    total_timesteps  | 685981   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0249   |
    |    n_updates        | 146495   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.89e+03 |
    |    ep_rew_mean      | 535      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3128     |
    |    fps              | 244      |
    |    time_elapsed     | 2809     |
    |    total_timesteps  | 686994   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0644   |
    |    n_updates        | 146748   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.9e+03  |
    |    ep_rew_mean      | 535      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3132     |
    |    fps              | 244      |
    |    time_elapsed     | 2816     |
    |    total_timesteps  | 688715   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.00978  |
    |    n_updates        | 147178   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.9e+03  |
    |    ep_rew_mean      | 535      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3136     |
    |    fps              | 244      |
    |    time_elapsed     | 2820     |
    |    total_timesteps  | 689660   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0324   |
    |    n_updates        | 147414   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.92e+03 |
    |    ep_rew_mean      | 540      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3140     |
    |    fps              | 244      |
    |    time_elapsed     | 2826     |
    |    total_timesteps  | 690961   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0491   |
    |    n_updates        | 147740   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.91e+03 |
    |    ep_rew_mean      | 537      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3144     |
    |    fps              | 244      |
    |    time_elapsed     | 2830     |
    |    total_timesteps  | 692128   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.019    |
    |    n_updates        | 148031   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.93e+03 |
    |    ep_rew_mean      | 540      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3148     |
    |    fps              | 244      |
    |    time_elapsed     | 2836     |
    |    total_timesteps  | 693588   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0157   |
    |    n_updates        | 148396   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.95e+03 |
    |    ep_rew_mean      | 543      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3152     |
    |    fps              | 244      |
    |    time_elapsed     | 2843     |
    |    total_timesteps  | 695327   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0244   |
    |    n_updates        | 148831   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.95e+03 |
    |    ep_rew_mean      | 546      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3156     |
    |    fps              | 244      |
    |    time_elapsed     | 2849     |
    |    total_timesteps  | 696663   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0357   |
    |    n_updates        | 149165   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.96e+03 |
    |    ep_rew_mean      | 550      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3160     |
    |    fps              | 244      |
    |    time_elapsed     | 2856     |
    |    total_timesteps  | 698372   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.014    |
    |    n_updates        | 149592   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.97e+03 |
    |    ep_rew_mean      | 551      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3164     |
    |    fps              | 244      |
    |    time_elapsed     | 2861     |
    |    total_timesteps  | 699661   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.00693  |
    |    n_updates        | 149915   |
    ----------------------------------
    Eval num_timesteps=700000, episode_reward=452.00 +/- 235.79
    Episode length: 3486.80 +/- 1080.55
    ----------------------------------
    | eval/               |          |
    |    mean_ep_length   | 3.49e+03 |
    |    mean_reward      | 452      |
    | rollout/            |          |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    total_timesteps  | 700000   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0218   |
    |    n_updates        | 149999   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.97e+03 |
    |    ep_rew_mean      | 550      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3168     |
    |    fps              | 243      |
    |    time_elapsed     | 2875     |
    |    total_timesteps  | 700757   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0238   |
    |    n_updates        | 150189   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.97e+03 |
    |    ep_rew_mean      | 548      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3172     |
    |    fps              | 243      |
    |    time_elapsed     | 2881     |
    |    total_timesteps  | 702102   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0173   |
    |    n_updates        | 150525   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.99e+03 |
    |    ep_rew_mean      | 554      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3176     |
    |    fps              | 243      |
    |    time_elapsed     | 2887     |
    |    total_timesteps  | 703545   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0271   |
    |    n_updates        | 150886   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.96e+03 |
    |    ep_rew_mean      | 551      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3180     |
    |    fps              | 243      |
    |    time_elapsed     | 2892     |
    |    total_timesteps  | 704723   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0282   |
    |    n_updates        | 151180   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.97e+03 |
    |    ep_rew_mean      | 551      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3184     |
    |    fps              | 243      |
    |    time_elapsed     | 2897     |
    |    total_timesteps  | 706013   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0247   |
    |    n_updates        | 151503   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.96e+03 |
    |    ep_rew_mean      | 549      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3188     |
    |    fps              | 243      |
    |    time_elapsed     | 2901     |
    |    total_timesteps  | 706882   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0858   |
    |    n_updates        | 151720   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.91e+03 |
    |    ep_rew_mean      | 543      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3192     |
    |    fps              | 243      |
    |    time_elapsed     | 2904     |
    |    total_timesteps  | 707622   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0146   |
    |    n_updates        | 151905   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.88e+03 |
    |    ep_rew_mean      | 535      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3196     |
    |    fps              | 243      |
    |    time_elapsed     | 2911     |
    |    total_timesteps  | 709399   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0501   |
    |    n_updates        | 152349   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.88e+03 |
    |    ep_rew_mean      | 535      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3200     |
    |    fps              | 243      |
    |    time_elapsed     | 2916     |
    |    total_timesteps  | 710477   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.00694  |
    |    n_updates        | 152619   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.92e+03 |
    |    ep_rew_mean      | 536      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3204     |
    |    fps              | 243      |
    |    time_elapsed     | 2923     |
    |    total_timesteps  | 712324   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0172   |
    |    n_updates        | 153080   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.89e+03 |
    |    ep_rew_mean      | 537      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3208     |
    |    fps              | 243      |
    |    time_elapsed     | 2932     |
    |    total_timesteps  | 714495   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0414   |
    |    n_updates        | 153623   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.9e+03  |
    |    ep_rew_mean      | 536      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3212     |
    |    fps              | 243      |
    |    time_elapsed     | 2936     |
    |    total_timesteps  | 715512   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0139   |
    |    n_updates        | 153877   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.92e+03 |
    |    ep_rew_mean      | 538      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3216     |
    |    fps              | 243      |
    |    time_elapsed     | 2944     |
    |    total_timesteps  | 717397   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.00884  |
    |    n_updates        | 154349   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.91e+03 |
    |    ep_rew_mean      | 538      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3220     |
    |    fps              | 243      |
    |    time_elapsed     | 2948     |
    |    total_timesteps  | 718250   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0402   |
    |    n_updates        | 154562   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.9e+03  |
    |    ep_rew_mean      | 533      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3224     |
    |    fps              | 243      |
    |    time_elapsed     | 2952     |
    |    total_timesteps  | 719227   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.00578  |
    |    n_updates        | 154806   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.89e+03 |
    |    ep_rew_mean      | 532      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3228     |
    |    fps              | 243      |
    |    time_elapsed     | 2955     |
    |    total_timesteps  | 720049   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0231   |
    |    n_updates        | 155012   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.9e+03  |
    |    ep_rew_mean      | 534      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3232     |
    |    fps              | 243      |
    |    time_elapsed     | 2960     |
    |    total_timesteps  | 721239   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0109   |
    |    n_updates        | 155309   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.87e+03 |
    |    ep_rew_mean      | 535      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3236     |
    |    fps              | 243      |
    |    time_elapsed     | 2967     |
    |    total_timesteps  | 722861   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.00647  |
    |    n_updates        | 155715   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.85e+03 |
    |    ep_rew_mean      | 532      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3240     |
    |    fps              | 243      |
    |    time_elapsed     | 2972     |
    |    total_timesteps  | 723970   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0192   |
    |    n_updates        | 155992   |
    ----------------------------------
    Eval num_timesteps=725000, episode_reward=705.00 +/- 225.99
    Episode length: 3936.60 +/- 740.12
    ----------------------------------
    | eval/               |          |
    |    mean_ep_length   | 3.94e+03 |
    |    mean_reward      | 705      |
    | rollout/            |          |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    total_timesteps  | 725000   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0172   |
    |    n_updates        | 156249   |
    ----------------------------------
    New best mean reward!
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.87e+03 |
    |    ep_rew_mean      | 538      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3244     |
    |    fps              | 242      |
    |    time_elapsed     | 2987     |
    |    total_timesteps  | 725130   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0503   |
    |    n_updates        | 156282   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.88e+03 |
    |    ep_rew_mean      | 539      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3248     |
    |    fps              | 242      |
    |    time_elapsed     | 2994     |
    |    total_timesteps  | 726806   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.021    |
    |    n_updates        | 156701   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.88e+03 |
    |    ep_rew_mean      | 542      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3252     |
    |    fps              | 242      |
    |    time_elapsed     | 3002     |
    |    total_timesteps  | 728547   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0197   |
    |    n_updates        | 157136   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.89e+03 |
    |    ep_rew_mean      | 547      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3256     |
    |    fps              | 242      |
    |    time_elapsed     | 3009     |
    |    total_timesteps  | 730318   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0682   |
    |    n_updates        | 157579   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.88e+03 |
    |    ep_rew_mean      | 549      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3260     |
    |    fps              | 242      |
    |    time_elapsed     | 3015     |
    |    total_timesteps  | 731696   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0173   |
    |    n_updates        | 157923   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.83e+03 |
    |    ep_rew_mean      | 537      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3264     |
    |    fps              | 242      |
    |    time_elapsed     | 3018     |
    |    total_timesteps  | 732671   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0204   |
    |    n_updates        | 158167   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.83e+03 |
    |    ep_rew_mean      | 538      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3268     |
    |    fps              | 242      |
    |    time_elapsed     | 3025     |
    |    total_timesteps  | 734069   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0313   |
    |    n_updates        | 158517   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.81e+03 |
    |    ep_rew_mean      | 535      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3272     |
    |    fps              | 242      |
    |    time_elapsed     | 3030     |
    |    total_timesteps  | 735377   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0289   |
    |    n_updates        | 158844   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.81e+03 |
    |    ep_rew_mean      | 535      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3276     |
    |    fps              | 242      |
    |    time_elapsed     | 3035     |
    |    total_timesteps  | 736656   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0253   |
    |    n_updates        | 159163   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.79e+03 |
    |    ep_rew_mean      | 537      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3280     |
    |    fps              | 242      |
    |    time_elapsed     | 3041     |
    |    total_timesteps  | 738090   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0217   |
    |    n_updates        | 159522   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.76e+03 |
    |    ep_rew_mean      | 532      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3284     |
    |    fps              | 242      |
    |    time_elapsed     | 3046     |
    |    total_timesteps  | 739204   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0126   |
    |    n_updates        | 159800   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.74e+03 |
    |    ep_rew_mean      | 530      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3288     |
    |    fps              | 242      |
    |    time_elapsed     | 3051     |
    |    total_timesteps  | 740447   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.00873  |
    |    n_updates        | 160111   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.75e+03 |
    |    ep_rew_mean      | 532      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3292     |
    |    fps              | 242      |
    |    time_elapsed     | 3056     |
    |    total_timesteps  | 741866   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0155   |
    |    n_updates        | 160466   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.74e+03 |
    |    ep_rew_mean      | 529      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3296     |
    |    fps              | 242      |
    |    time_elapsed     | 3062     |
    |    total_timesteps  | 743276   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.00974  |
    |    n_updates        | 160818   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.78e+03 |
    |    ep_rew_mean      | 535      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3300     |
    |    fps              | 242      |
    |    time_elapsed     | 3069     |
    |    total_timesteps  | 744937   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0277   |
    |    n_updates        | 161234   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.81e+03 |
    |    ep_rew_mean      | 542      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3304     |
    |    fps              | 242      |
    |    time_elapsed     | 3078     |
    |    total_timesteps  | 747204   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.024    |
    |    n_updates        | 161800   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.81e+03 |
    |    ep_rew_mean      | 542      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3308     |
    |    fps              | 242      |
    |    time_elapsed     | 3085     |
    |    total_timesteps  | 748711   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0171   |
    |    n_updates        | 162177   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.84e+03 |
    |    ep_rew_mean      | 539      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3312     |
    |    fps              | 242      |
    |    time_elapsed     | 3089     |
    |    total_timesteps  | 749888   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0082   |
    |    n_updates        | 162471   |
    ----------------------------------
    Eval num_timesteps=750000, episode_reward=590.00 +/- 56.48
    Episode length: 4209.00 +/- 355.03
    ----------------------------------
    | eval/               |          |
    |    mean_ep_length   | 4.21e+03 |
    |    mean_reward      | 590      |
    | rollout/            |          |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    total_timesteps  | 750000   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0155   |
    |    n_updates        | 162499   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.85e+03 |
    |    ep_rew_mean      | 542      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3316     |
    |    fps              | 241      |
    |    time_elapsed     | 3107     |
    |    total_timesteps  | 751208   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0158   |
    |    n_updates        | 162801   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.89e+03 |
    |    ep_rew_mean      | 543      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3320     |
    |    fps              | 241      |
    |    time_elapsed     | 3116     |
    |    total_timesteps  | 753622   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0124   |
    |    n_updates        | 163405   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.88e+03 |
    |    ep_rew_mean      | 540      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3324     |
    |    fps              | 241      |
    |    time_elapsed     | 3122     |
    |    total_timesteps  | 754846   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0104   |
    |    n_updates        | 163711   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.89e+03 |
    |    ep_rew_mean      | 542      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3328     |
    |    fps              | 241      |
    |    time_elapsed     | 3127     |
    |    total_timesteps  | 756158   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0602   |
    |    n_updates        | 164039   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.92e+03 |
    |    ep_rew_mean      | 547      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3332     |
    |    fps              | 241      |
    |    time_elapsed     | 3134     |
    |    total_timesteps  | 757891   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0281   |
    |    n_updates        | 164472   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.91e+03 |
    |    ep_rew_mean      | 547      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3336     |
    |    fps              | 241      |
    |    time_elapsed     | 3140     |
    |    total_timesteps  | 759406   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0199   |
    |    n_updates        | 164851   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.9e+03  |
    |    ep_rew_mean      | 543      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3340     |
    |    fps              | 241      |
    |    time_elapsed     | 3145     |
    |    total_timesteps  | 760442   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0165   |
    |    n_updates        | 165110   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.89e+03 |
    |    ep_rew_mean      | 543      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3344     |
    |    fps              | 241      |
    |    time_elapsed     | 3151     |
    |    total_timesteps  | 761963   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0339   |
    |    n_updates        | 165490   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.9e+03  |
    |    ep_rew_mean      | 544      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3348     |
    |    fps              | 241      |
    |    time_elapsed     | 3160     |
    |    total_timesteps  | 764065   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0133   |
    |    n_updates        | 166016   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.88e+03 |
    |    ep_rew_mean      | 546      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3352     |
    |    fps              | 241      |
    |    time_elapsed     | 3167     |
    |    total_timesteps  | 766003   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0298   |
    |    n_updates        | 166500   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.91e+03 |
    |    ep_rew_mean      | 554      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3356     |
    |    fps              | 241      |
    |    time_elapsed     | 3173     |
    |    total_timesteps  | 767429   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0222   |
    |    n_updates        | 166857   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.92e+03 |
    |    ep_rew_mean      | 559      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3360     |
    |    fps              | 241      |
    |    time_elapsed     | 3180     |
    |    total_timesteps  | 769004   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.00674  |
    |    n_updates        | 167250   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.92e+03 |
    |    ep_rew_mean      | 560      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3364     |
    |    fps              | 241      |
    |    time_elapsed     | 3187     |
    |    total_timesteps  | 770612   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0591   |
    |    n_updates        | 167652   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.92e+03 |
    |    ep_rew_mean      | 568      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3368     |
    |    fps              | 241      |
    |    time_elapsed     | 3197     |
    |    total_timesteps  | 773043   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0772   |
    |    n_updates        | 168260   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.89e+03 |
    |    ep_rew_mean      | 565      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3372     |
    |    fps              | 241      |
    |    time_elapsed     | 3203     |
    |    total_timesteps  | 774630   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0214   |
    |    n_updates        | 168657   |
    ----------------------------------
    Eval num_timesteps=775000, episode_reward=712.00 +/- 352.68
    Episode length: 5134.40 +/- 2188.05
    ----------------------------------
    | eval/               |          |
    |    mean_ep_length   | 5.13e+03 |
    |    mean_reward      | 712      |
    | rollout/            |          |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    total_timesteps  | 775000   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0255   |
    |    n_updates        | 168749   |
    ----------------------------------
    New best mean reward!
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.9e+03  |
    |    ep_rew_mean      | 565      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3376     |
    |    fps              | 240      |
    |    time_elapsed     | 3223     |
    |    total_timesteps  | 775975   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0379   |
    |    n_updates        | 168993   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.87e+03 |
    |    ep_rew_mean      | 564      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3380     |
    |    fps              | 240      |
    |    time_elapsed     | 3232     |
    |    total_timesteps  | 777889   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0348   |
    |    n_updates        | 169472   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.88e+03 |
    |    ep_rew_mean      | 568      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3384     |
    |    fps              | 240      |
    |    time_elapsed     | 3236     |
    |    total_timesteps  | 779046   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0101   |
    |    n_updates        | 169761   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.87e+03 |
    |    ep_rew_mean      | 564      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3388     |
    |    fps              | 240      |
    |    time_elapsed     | 3241     |
    |    total_timesteps  | 780189   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0286   |
    |    n_updates        | 170047   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.85e+03 |
    |    ep_rew_mean      | 561      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3392     |
    |    fps              | 240      |
    |    time_elapsed     | 3247     |
    |    total_timesteps  | 781515   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.018    |
    |    n_updates        | 170378   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.85e+03 |
    |    ep_rew_mean      | 562      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3396     |
    |    fps              | 240      |
    |    time_elapsed     | 3253     |
    |    total_timesteps  | 783083   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0448   |
    |    n_updates        | 170770   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.86e+03 |
    |    ep_rew_mean      | 566      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3400     |
    |    fps              | 240      |
    |    time_elapsed     | 3259     |
    |    total_timesteps  | 784370   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0306   |
    |    n_updates        | 171092   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.88e+03 |
    |    ep_rew_mean      | 571      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3404     |
    |    fps              | 240      |
    |    time_elapsed     | 3265     |
    |    total_timesteps  | 785999   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0225   |
    |    n_updates        | 171499   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.86e+03 |
    |    ep_rew_mean      | 569      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3408     |
    |    fps              | 240      |
    |    time_elapsed     | 3270     |
    |    total_timesteps  | 787264   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0279   |
    |    n_updates        | 171815   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.86e+03 |
    |    ep_rew_mean      | 571      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3412     |
    |    fps              | 240      |
    |    time_elapsed     | 3274     |
    |    total_timesteps  | 788254   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0134   |
    |    n_updates        | 172063   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.84e+03 |
    |    ep_rew_mean      | 568      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3416     |
    |    fps              | 240      |
    |    time_elapsed     | 3281     |
    |    total_timesteps  | 789879   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.00683  |
    |    n_updates        | 172469   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.84e+03 |
    |    ep_rew_mean      | 564      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3420     |
    |    fps              | 240      |
    |    time_elapsed     | 3286     |
    |    total_timesteps  | 791235   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0393   |
    |    n_updates        | 172808   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.83e+03 |
    |    ep_rew_mean      | 561      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3424     |
    |    fps              | 240      |
    |    time_elapsed     | 3291     |
    |    total_timesteps  | 792395   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0316   |
    |    n_updates        | 173098   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.82e+03 |
    |    ep_rew_mean      | 560      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3428     |
    |    fps              | 240      |
    |    time_elapsed     | 3296     |
    |    total_timesteps  | 793633   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0227   |
    |    n_updates        | 173408   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.81e+03 |
    |    ep_rew_mean      | 562      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3432     |
    |    fps              | 240      |
    |    time_elapsed     | 3300     |
    |    total_timesteps  | 794631   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0129   |
    |    n_updates        | 173657   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.83e+03 |
    |    ep_rew_mean      | 565      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3436     |
    |    fps              | 240      |
    |    time_elapsed     | 3307     |
    |    total_timesteps  | 796291   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0168   |
    |    n_updates        | 174072   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.83e+03 |
    |    ep_rew_mean      | 565      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3440     |
    |    fps              | 240      |
    |    time_elapsed     | 3312     |
    |    total_timesteps  | 797487   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0765   |
    |    n_updates        | 174371   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.86e+03 |
    |    ep_rew_mean      | 566      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3444     |
    |    fps              | 240      |
    |    time_elapsed     | 3319     |
    |    total_timesteps  | 799164   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0276   |
    |    n_updates        | 174790   |
    ----------------------------------
    Eval num_timesteps=800000, episode_reward=595.00 +/- 102.27
    Episode length: 4255.40 +/- 472.71
    ----------------------------------
    | eval/               |          |
    |    mean_ep_length   | 4.26e+03 |
    |    mean_reward      | 595      |
    | rollout/            |          |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    total_timesteps  | 800000   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0148   |
    |    n_updates        | 174999   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.86e+03 |
    |    ep_rew_mean      | 566      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3448     |
    |    fps              | 239      |
    |    time_elapsed     | 3338     |
    |    total_timesteps  | 800792   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0136   |
    |    n_updates        | 175197   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.87e+03 |
    |    ep_rew_mean      | 571      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3452     |
    |    fps              | 239      |
    |    time_elapsed     | 3343     |
    |    total_timesteps  | 802128   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0185   |
    |    n_updates        | 175531   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.88e+03 |
    |    ep_rew_mean      | 572      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3456     |
    |    fps              | 239      |
    |    time_elapsed     | 3349     |
    |    total_timesteps  | 803448   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.022    |
    |    n_updates        | 175861   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.91e+03 |
    |    ep_rew_mean      | 578      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3460     |
    |    fps              | 239      |
    |    time_elapsed     | 3356     |
    |    total_timesteps  | 805322   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.023    |
    |    n_updates        | 176330   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.9e+03  |
    |    ep_rew_mean      | 581      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3464     |
    |    fps              | 239      |
    |    time_elapsed     | 3360     |
    |    total_timesteps  | 806311   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0145   |
    |    n_updates        | 176577   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.88e+03 |
    |    ep_rew_mean      | 582      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3468     |
    |    fps              | 239      |
    |    time_elapsed     | 3365     |
    |    total_timesteps  | 807432   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0195   |
    |    n_updates        | 176857   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.86e+03 |
    |    ep_rew_mean      | 582      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3472     |
    |    fps              | 239      |
    |    time_elapsed     | 3370     |
    |    total_timesteps  | 808676   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0568   |
    |    n_updates        | 177168   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.86e+03 |
    |    ep_rew_mean      | 580      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3476     |
    |    fps              | 239      |
    |    time_elapsed     | 3375     |
    |    total_timesteps  | 809970   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0195   |
    |    n_updates        | 177492   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.87e+03 |
    |    ep_rew_mean      | 582      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3480     |
    |    fps              | 239      |
    |    time_elapsed     | 3382     |
    |    total_timesteps  | 811457   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.00961  |
    |    n_updates        | 177864   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.87e+03 |
    |    ep_rew_mean      | 580      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3484     |
    |    fps              | 239      |
    |    time_elapsed     | 3389     |
    |    total_timesteps  | 813248   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0168   |
    |    n_updates        | 178311   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.89e+03 |
    |    ep_rew_mean      | 583      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3488     |
    |    fps              | 239      |
    |    time_elapsed     | 3397     |
    |    total_timesteps  | 815294   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.022    |
    |    n_updates        | 178823   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.91e+03 |
    |    ep_rew_mean      | 588      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3492     |
    |    fps              | 239      |
    |    time_elapsed     | 3400     |
    |    total_timesteps  | 815903   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0202   |
    |    n_updates        | 178975   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.92e+03 |
    |    ep_rew_mean      | 589      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3496     |
    |    fps              | 239      |
    |    time_elapsed     | 3405     |
    |    total_timesteps  | 817137   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0347   |
    |    n_updates        | 179284   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.94e+03 |
    |    ep_rew_mean      | 596      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3500     |
    |    fps              | 239      |
    |    time_elapsed     | 3413     |
    |    total_timesteps  | 819018   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.00946  |
    |    n_updates        | 179754   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.95e+03 |
    |    ep_rew_mean      | 594      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3504     |
    |    fps              | 239      |
    |    time_elapsed     | 3419     |
    |    total_timesteps  | 820557   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0254   |
    |    n_updates        | 180139   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.98e+03 |
    |    ep_rew_mean      | 597      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3508     |
    |    fps              | 239      |
    |    time_elapsed     | 3424     |
    |    total_timesteps  | 821914   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0318   |
    |    n_updates        | 180478   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.98e+03 |
    |    ep_rew_mean      | 595      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3512     |
    |    fps              | 239      |
    |    time_elapsed     | 3431     |
    |    total_timesteps  | 823524   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0127   |
    |    n_updates        | 180880   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.99e+03 |
    |    ep_rew_mean      | 594      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3516     |
    |    fps              | 239      |
    |    time_elapsed     | 3436     |
    |    total_timesteps  | 824642   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.018    |
    |    n_updates        | 181160   |
    ----------------------------------
    Eval num_timesteps=825000, episode_reward=657.00 +/- 139.20
    Episode length: 5074.40 +/- 1098.88
    ----------------------------------
    | eval/               |          |
    |    mean_ep_length   | 5.07e+03 |
    |    mean_reward      | 657      |
    | rollout/            |          |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    total_timesteps  | 825000   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0255   |
    |    n_updates        | 181249   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 4e+03    |
    |    ep_rew_mean      | 590      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3520     |
    |    fps              | 238      |
    |    time_elapsed     | 3456     |
    |    total_timesteps  | 826008   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0211   |
    |    n_updates        | 181501   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 4.04e+03 |
    |    ep_rew_mean      | 596      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3524     |
    |    fps              | 238      |
    |    time_elapsed     | 3467     |
    |    total_timesteps  | 828699   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0175   |
    |    n_updates        | 182174   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 4.03e+03 |
    |    ep_rew_mean      | 591      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3528     |
    |    fps              | 238      |
    |    time_elapsed     | 3472     |
    |    total_timesteps  | 829839   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0424   |
    |    n_updates        | 182459   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 4.03e+03 |
    |    ep_rew_mean      | 588      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3532     |
    |    fps              | 238      |
    |    time_elapsed     | 3479     |
    |    total_timesteps  | 831592   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0137   |
    |    n_updates        | 182897   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 4.05e+03 |
    |    ep_rew_mean      | 596      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3536     |
    |    fps              | 238      |
    |    time_elapsed     | 3487     |
    |    total_timesteps  | 833465   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0301   |
    |    n_updates        | 183366   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 4.06e+03 |
    |    ep_rew_mean      | 596      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3540     |
    |    fps              | 238      |
    |    time_elapsed     | 3494     |
    |    total_timesteps  | 835118   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0124   |
    |    n_updates        | 183779   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 4.08e+03 |
    |    ep_rew_mean      | 599      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3544     |
    |    fps              | 238      |
    |    time_elapsed     | 3499     |
    |    total_timesteps  | 836409   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.00943  |
    |    n_updates        | 184102   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 4.09e+03 |
    |    ep_rew_mean      | 597      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3548     |
    |    fps              | 238      |
    |    time_elapsed     | 3508     |
    |    total_timesteps  | 838498   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.00861  |
    |    n_updates        | 184624   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 4.1e+03  |
    |    ep_rew_mean      | 599      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3552     |
    |    fps              | 239      |
    |    time_elapsed     | 3514     |
    |    total_timesteps  | 839929   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0482   |
    |    n_updates        | 184982   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 4.09e+03 |
    |    ep_rew_mean      | 593      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3556     |
    |    fps              | 239      |
    |    time_elapsed     | 3517     |
    |    total_timesteps  | 840810   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0103   |
    |    n_updates        | 185202   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 4.11e+03 |
    |    ep_rew_mean      | 596      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3560     |
    |    fps              | 239      |
    |    time_elapsed     | 3523     |
    |    total_timesteps  | 842167   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0165   |
    |    n_updates        | 185541   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 4.1e+03  |
    |    ep_rew_mean      | 599      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3564     |
    |    fps              | 239      |
    |    time_elapsed     | 3532     |
    |    total_timesteps  | 844409   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0461   |
    |    n_updates        | 186102   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 4.09e+03 |
    |    ep_rew_mean      | 596      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3568     |
    |    fps              | 239      |
    |    time_elapsed     | 3538     |
    |    total_timesteps  | 845895   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0309   |
    |    n_updates        | 186473   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 4.07e+03 |
    |    ep_rew_mean      | 595      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3572     |
    |    fps              | 239      |
    |    time_elapsed     | 3544     |
    |    total_timesteps  | 847344   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0189   |
    |    n_updates        | 186835   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 4.07e+03 |
    |    ep_rew_mean      | 597      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3576     |
    |    fps              | 239      |
    |    time_elapsed     | 3551     |
    |    total_timesteps  | 849045   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.123    |
    |    n_updates        | 187261   |
    ----------------------------------
    Eval num_timesteps=850000, episode_reward=460.00 +/- 246.88
    Episode length: 3825.00 +/- 1141.57
    ----------------------------------
    | eval/               |          |
    |    mean_ep_length   | 3.82e+03 |
    |    mean_reward      | 460      |
    | rollout/            |          |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    total_timesteps  | 850000   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0106   |
    |    n_updates        | 187499   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 4.07e+03 |
    |    ep_rew_mean      | 599      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3580     |
    |    fps              | 238      |
    |    time_elapsed     | 3567     |
    |    total_timesteps  | 850230   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.016    |
    |    n_updates        | 187557   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 4.05e+03 |
    |    ep_rew_mean      | 593      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3584     |
    |    fps              | 238      |
    |    time_elapsed     | 3571     |
    |    total_timesteps  | 851024   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.012    |
    |    n_updates        | 187755   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 4.05e+03 |
    |    ep_rew_mean      | 593      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3588     |
    |    fps              | 238      |
    |    time_elapsed     | 3580     |
    |    total_timesteps  | 853236   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0515   |
    |    n_updates        | 188308   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 4.03e+03 |
    |    ep_rew_mean      | 592      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3592     |
    |    fps              | 238      |
    |    time_elapsed     | 3583     |
    |    total_timesteps  | 854007   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0422   |
    |    n_updates        | 188501   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 4.03e+03 |
    |    ep_rew_mean      | 594      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3596     |
    |    fps              | 238      |
    |    time_elapsed     | 3587     |
    |    total_timesteps  | 855010   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.017    |
    |    n_updates        | 188752   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 4.03e+03 |
    |    ep_rew_mean      | 593      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3600     |
    |    fps              | 238      |
    |    time_elapsed     | 3592     |
    |    total_timesteps  | 856063   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0396   |
    |    n_updates        | 189015   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 4e+03    |
    |    ep_rew_mean      | 588      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3604     |
    |    fps              | 238      |
    |    time_elapsed     | 3598     |
    |    total_timesteps  | 857688   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0256   |
    |    n_updates        | 189421   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 4.01e+03 |
    |    ep_rew_mean      | 591      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3608     |
    |    fps              | 238      |
    |    time_elapsed     | 3603     |
    |    total_timesteps  | 858814   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0217   |
    |    n_updates        | 189703   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 4.02e+03 |
    |    ep_rew_mean      | 591      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3612     |
    |    fps              | 238      |
    |    time_elapsed     | 3609     |
    |    total_timesteps  | 860117   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0339   |
    |    n_updates        | 190029   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 4.04e+03 |
    |    ep_rew_mean      | 591      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3616     |
    |    fps              | 238      |
    |    time_elapsed     | 3615     |
    |    total_timesteps  | 861670   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0207   |
    |    n_updates        | 190417   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 4.02e+03 |
    |    ep_rew_mean      | 590      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3620     |
    |    fps              | 238      |
    |    time_elapsed     | 3618     |
    |    total_timesteps  | 862471   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0161   |
    |    n_updates        | 190617   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 4e+03    |
    |    ep_rew_mean      | 588      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3624     |
    |    fps              | 238      |
    |    time_elapsed     | 3622     |
    |    total_timesteps  | 863315   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0457   |
    |    n_updates        | 190828   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.98e+03 |
    |    ep_rew_mean      | 580      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3628     |
    |    fps              | 238      |
    |    time_elapsed     | 3627     |
    |    total_timesteps  | 864603   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0126   |
    |    n_updates        | 191150   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.97e+03 |
    |    ep_rew_mean      | 574      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3632     |
    |    fps              | 238      |
    |    time_elapsed     | 3631     |
    |    total_timesteps  | 865500   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0155   |
    |    n_updates        | 191374   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.96e+03 |
    |    ep_rew_mean      | 574      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3636     |
    |    fps              | 238      |
    |    time_elapsed     | 3637     |
    |    total_timesteps  | 867024   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0508   |
    |    n_updates        | 191755   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.93e+03 |
    |    ep_rew_mean      | 569      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3640     |
    |    fps              | 238      |
    |    time_elapsed     | 3640     |
    |    total_timesteps  | 867816   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.00616  |
    |    n_updates        | 191953   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.94e+03 |
    |    ep_rew_mean      | 564      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3644     |
    |    fps              | 238      |
    |    time_elapsed     | 3647     |
    |    total_timesteps  | 869298   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0135   |
    |    n_updates        | 192324   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.98e+03 |
    |    ep_rew_mean      | 569      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3648     |
    |    fps              | 238      |
    |    time_elapsed     | 3656     |
    |    total_timesteps  | 871655   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.00803  |
    |    n_updates        | 192913   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.98e+03 |
    |    ep_rew_mean      | 571      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3652     |
    |    fps              | 238      |
    |    time_elapsed     | 3663     |
    |    total_timesteps  | 873302   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.029    |
    |    n_updates        | 193325   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.96e+03 |
    |    ep_rew_mean      | 565      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3656     |
    |    fps              | 238      |
    |    time_elapsed     | 3666     |
    |    total_timesteps  | 874013   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0314   |
    |    n_updates        | 193503   |
    ----------------------------------
    Eval num_timesteps=875000, episode_reward=500.00 +/- 71.48
    Episode length: 4110.60 +/- 446.87
    ----------------------------------
    | eval/               |          |
    |    mean_ep_length   | 4.11e+03 |
    |    mean_reward      | 500      |
    | rollout/            |          |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    total_timesteps  | 875000   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0286   |
    |    n_updates        | 193749   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.97e+03 |
    |    ep_rew_mean      | 566      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3660     |
    |    fps              | 237      |
    |    time_elapsed     | 3684     |
    |    total_timesteps  | 875295   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0196   |
    |    n_updates        | 193823   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.95e+03 |
    |    ep_rew_mean      | 564      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3664     |
    |    fps              | 237      |
    |    time_elapsed     | 3688     |
    |    total_timesteps  | 876235   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0287   |
    |    n_updates        | 194058   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.94e+03 |
    |    ep_rew_mean      | 558      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3668     |
    |    fps              | 237      |
    |    time_elapsed     | 3693     |
    |    total_timesteps  | 877409   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0273   |
    |    n_updates        | 194352   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.96e+03 |
    |    ep_rew_mean      | 563      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3672     |
    |    fps              | 237      |
    |    time_elapsed     | 3698     |
    |    total_timesteps  | 878701   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0119   |
    |    n_updates        | 194675   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.96e+03 |
    |    ep_rew_mean      | 563      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3676     |
    |    fps              | 237      |
    |    time_elapsed     | 3704     |
    |    total_timesteps  | 880222   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0094   |
    |    n_updates        | 195055   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.96e+03 |
    |    ep_rew_mean      | 564      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3680     |
    |    fps              | 237      |
    |    time_elapsed     | 3709     |
    |    total_timesteps  | 881314   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.019    |
    |    n_updates        | 195328   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.95e+03 |
    |    ep_rew_mean      | 561      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3684     |
    |    fps              | 237      |
    |    time_elapsed     | 3712     |
    |    total_timesteps  | 882098   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0138   |
    |    n_updates        | 195524   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.93e+03 |
    |    ep_rew_mean      | 559      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3688     |
    |    fps              | 237      |
    |    time_elapsed     | 3719     |
    |    total_timesteps  | 883659   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0203   |
    |    n_updates        | 195914   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.93e+03 |
    |    ep_rew_mean      | 557      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3692     |
    |    fps              | 237      |
    |    time_elapsed     | 3725     |
    |    total_timesteps  | 885172   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0142   |
    |    n_updates        | 196292   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.94e+03 |
    |    ep_rew_mean      | 557      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3696     |
    |    fps              | 237      |
    |    time_elapsed     | 3728     |
    |    total_timesteps  | 885822   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.009    |
    |    n_updates        | 196455   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.92e+03 |
    |    ep_rew_mean      | 554      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3700     |
    |    fps              | 237      |
    |    time_elapsed     | 3735     |
    |    total_timesteps  | 887496   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.00972  |
    |    n_updates        | 196873   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.91e+03 |
    |    ep_rew_mean      | 552      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3704     |
    |    fps              | 237      |
    |    time_elapsed     | 3740     |
    |    total_timesteps  | 888848   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.00788  |
    |    n_updates        | 197211   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.92e+03 |
    |    ep_rew_mean      | 553      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3708     |
    |    fps              | 237      |
    |    time_elapsed     | 3747     |
    |    total_timesteps  | 890442   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0239   |
    |    n_updates        | 197610   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.94e+03 |
    |    ep_rew_mean      | 553      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3712     |
    |    fps              | 237      |
    |    time_elapsed     | 3753     |
    |    total_timesteps  | 891981   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.00776  |
    |    n_updates        | 197995   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.93e+03 |
    |    ep_rew_mean      | 552      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3716     |
    |    fps              | 237      |
    |    time_elapsed     | 3761     |
    |    total_timesteps  | 893986   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0221   |
    |    n_updates        | 198496   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.92e+03 |
    |    ep_rew_mean      | 549      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3720     |
    |    fps              | 237      |
    |    time_elapsed     | 3766     |
    |    total_timesteps  | 895077   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0172   |
    |    n_updates        | 198769   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.9e+03  |
    |    ep_rew_mean      | 548      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3724     |
    |    fps              | 237      |
    |    time_elapsed     | 3771     |
    |    total_timesteps  | 896275   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0169   |
    |    n_updates        | 199068   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.88e+03 |
    |    ep_rew_mean      | 543      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3728     |
    |    fps              | 237      |
    |    time_elapsed     | 3774     |
    |    total_timesteps  | 897044   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0144   |
    |    n_updates        | 199260   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.87e+03 |
    |    ep_rew_mean      | 544      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3732     |
    |    fps              | 237      |
    |    time_elapsed     | 3782     |
    |    total_timesteps  | 898887   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0131   |
    |    n_updates        | 199721   |
    ----------------------------------
    Eval num_timesteps=900000, episode_reward=637.00 +/- 129.83
    Episode length: 4053.20 +/- 283.45
    ----------------------------------
    | eval/               |          |
    |    mean_ep_length   | 4.05e+03 |
    |    mean_reward      | 637      |
    | rollout/            |          |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    total_timesteps  | 900000   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0211   |
    |    n_updates        | 199999   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.85e+03 |
    |    ep_rew_mean      | 538      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3736     |
    |    fps              | 236      |
    |    time_elapsed     | 3799     |
    |    total_timesteps  | 900102   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0163   |
    |    n_updates        | 200025   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.85e+03 |
    |    ep_rew_mean      | 538      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3740     |
    |    fps              | 236      |
    |    time_elapsed     | 3804     |
    |    total_timesteps  | 901312   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0436   |
    |    n_updates        | 200327   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.85e+03 |
    |    ep_rew_mean      | 536      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3744     |
    |    fps              | 236      |
    |    time_elapsed     | 3810     |
    |    total_timesteps  | 902789   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0234   |
    |    n_updates        | 200697   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.86e+03 |
    |    ep_rew_mean      | 540      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3748     |
    |    fps              | 236      |
    |    time_elapsed     | 3816     |
    |    total_timesteps  | 904189   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0102   |
    |    n_updates        | 201047   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.87e+03 |
    |    ep_rew_mean      | 540      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3752     |
    |    fps              | 236      |
    |    time_elapsed     | 3820     |
    |    total_timesteps  | 905165   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0294   |
    |    n_updates        | 201291   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.87e+03 |
    |    ep_rew_mean      | 540      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3756     |
    |    fps              | 236      |
    |    time_elapsed     | 3826     |
    |    total_timesteps  | 906563   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0159   |
    |    n_updates        | 201640   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.85e+03 |
    |    ep_rew_mean      | 539      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3760     |
    |    fps              | 236      |
    |    time_elapsed     | 3834     |
    |    total_timesteps  | 908603   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.00602  |
    |    n_updates        | 202150   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.85e+03 |
    |    ep_rew_mean      | 540      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3764     |
    |    fps              | 236      |
    |    time_elapsed     | 3838     |
    |    total_timesteps  | 909509   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0388   |
    |    n_updates        | 202377   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.84e+03 |
    |    ep_rew_mean      | 537      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3768     |
    |    fps              | 236      |
    |    time_elapsed     | 3844     |
    |    total_timesteps  | 911048   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0186   |
    |    n_updates        | 202761   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.86e+03 |
    |    ep_rew_mean      | 540      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3772     |
    |    fps              | 236      |
    |    time_elapsed     | 3850     |
    |    total_timesteps  | 912381   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0149   |
    |    n_updates        | 203095   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.84e+03 |
    |    ep_rew_mean      | 534      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3776     |
    |    fps              | 236      |
    |    time_elapsed     | 3859     |
    |    total_timesteps  | 914727   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0233   |
    |    n_updates        | 203681   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.86e+03 |
    |    ep_rew_mean      | 535      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3780     |
    |    fps              | 237      |
    |    time_elapsed     | 3865     |
    |    total_timesteps  | 916087   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0128   |
    |    n_updates        | 204021   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.86e+03 |
    |    ep_rew_mean      | 536      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3784     |
    |    fps              | 236      |
    |    time_elapsed     | 3870     |
    |    total_timesteps  | 917374   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.115    |
    |    n_updates        | 204343   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.84e+03 |
    |    ep_rew_mean      | 539      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3788     |
    |    fps              | 237      |
    |    time_elapsed     | 3876     |
    |    total_timesteps  | 918871   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0206   |
    |    n_updates        | 204717   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.85e+03 |
    |    ep_rew_mean      | 543      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3792     |
    |    fps              | 236      |
    |    time_elapsed     | 3881     |
    |    total_timesteps  | 919909   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0155   |
    |    n_updates        | 204977   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.81e+03 |
    |    ep_rew_mean      | 537      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3796     |
    |    fps              | 237      |
    |    time_elapsed     | 3889     |
    |    total_timesteps  | 921840   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0257   |
    |    n_updates        | 205459   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.82e+03 |
    |    ep_rew_mean      | 539      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3800     |
    |    fps              | 237      |
    |    time_elapsed     | 3897     |
    |    total_timesteps  | 923699   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0293   |
    |    n_updates        | 205924   |
    ----------------------------------
    Eval num_timesteps=925000, episode_reward=603.00 +/- 90.14
    Episode length: 4109.20 +/- 690.16
    ----------------------------------
    | eval/               |          |
    |    mean_ep_length   | 4.11e+03 |
    |    mean_reward      | 603      |
    | rollout/            |          |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    total_timesteps  | 925000   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0191   |
    |    n_updates        | 206249   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.82e+03 |
    |    ep_rew_mean      | 539      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3804     |
    |    fps              | 236      |
    |    time_elapsed     | 3915     |
    |    total_timesteps  | 925159   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.00806  |
    |    n_updates        | 206289   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.82e+03 |
    |    ep_rew_mean      | 541      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3808     |
    |    fps              | 236      |
    |    time_elapsed     | 3919     |
    |    total_timesteps  | 926078   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0246   |
    |    n_updates        | 206519   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.81e+03 |
    |    ep_rew_mean      | 536      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3812     |
    |    fps              | 236      |
    |    time_elapsed     | 3926     |
    |    total_timesteps  | 927998   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0346   |
    |    n_updates        | 206999   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.81e+03 |
    |    ep_rew_mean      | 540      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3816     |
    |    fps              | 236      |
    |    time_elapsed     | 3933     |
    |    total_timesteps  | 929420   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0149   |
    |    n_updates        | 207354   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.82e+03 |
    |    ep_rew_mean      | 543      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3820     |
    |    fps              | 236      |
    |    time_elapsed     | 3946     |
    |    total_timesteps  | 932619   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0229   |
    |    n_updates        | 208154   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.82e+03 |
    |    ep_rew_mean      | 544      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3824     |
    |    fps              | 236      |
    |    time_elapsed     | 3950     |
    |    total_timesteps  | 933647   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0604   |
    |    n_updates        | 208411   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.82e+03 |
    |    ep_rew_mean      | 544      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3828     |
    |    fps              | 236      |
    |    time_elapsed     | 3955     |
    |    total_timesteps  | 935017   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.00931  |
    |    n_updates        | 208754   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.82e+03 |
    |    ep_rew_mean      | 544      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3832     |
    |    fps              | 236      |
    |    time_elapsed     | 3960     |
    |    total_timesteps  | 936144   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0241   |
    |    n_updates        | 209035   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.81e+03 |
    |    ep_rew_mean      | 543      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3836     |
    |    fps              | 236      |
    |    time_elapsed     | 3968     |
    |    total_timesteps  | 938023   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0188   |
    |    n_updates        | 209505   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.82e+03 |
    |    ep_rew_mean      | 543      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3840     |
    |    fps              | 236      |
    |    time_elapsed     | 3973     |
    |    total_timesteps  | 939322   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0242   |
    |    n_updates        | 209830   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.82e+03 |
    |    ep_rew_mean      | 542      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3844     |
    |    fps              | 236      |
    |    time_elapsed     | 3978     |
    |    total_timesteps  | 940537   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.047    |
    |    n_updates        | 210134   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.82e+03 |
    |    ep_rew_mean      | 539      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3848     |
    |    fps              | 236      |
    |    time_elapsed     | 3987     |
    |    total_timesteps  | 942598   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0344   |
    |    n_updates        | 210649   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.81e+03 |
    |    ep_rew_mean      | 537      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3852     |
    |    fps              | 236      |
    |    time_elapsed     | 3991     |
    |    total_timesteps  | 943750   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.033    |
    |    n_updates        | 210937   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.82e+03 |
    |    ep_rew_mean      | 538      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3856     |
    |    fps              | 236      |
    |    time_elapsed     | 3999     |
    |    total_timesteps  | 945723   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0148   |
    |    n_updates        | 211430   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.84e+03 |
    |    ep_rew_mean      | 541      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3860     |
    |    fps              | 236      |
    |    time_elapsed     | 4005     |
    |    total_timesteps  | 947084   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0187   |
    |    n_updates        | 211770   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.84e+03 |
    |    ep_rew_mean      | 539      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3864     |
    |    fps              | 236      |
    |    time_elapsed     | 4012     |
    |    total_timesteps  | 948998   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0176   |
    |    n_updates        | 212249   |
    ----------------------------------
    Eval num_timesteps=950000, episode_reward=354.00 +/- 85.64
    Episode length: 4006.20 +/- 647.03
    ----------------------------------
    | eval/               |          |
    |    mean_ep_length   | 4.01e+03 |
    |    mean_reward      | 354      |
    | rollout/            |          |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    total_timesteps  | 950000   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.022    |
    |    n_updates        | 212499   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.85e+03 |
    |    ep_rew_mean      | 541      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3868     |
    |    fps              | 235      |
    |    time_elapsed     | 4029     |
    |    total_timesteps  | 950204   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.00608  |
    |    n_updates        | 212550   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.85e+03 |
    |    ep_rew_mean      | 543      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3872     |
    |    fps              | 235      |
    |    time_elapsed     | 4037     |
    |    total_timesteps  | 952091   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.00693  |
    |    n_updates        | 213022   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.85e+03 |
    |    ep_rew_mean      | 543      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3876     |
    |    fps              | 235      |
    |    time_elapsed     | 4041     |
    |    total_timesteps  | 953218   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0315   |
    |    n_updates        | 213304   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.84e+03 |
    |    ep_rew_mean      | 539      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3880     |
    |    fps              | 235      |
    |    time_elapsed     | 4046     |
    |    total_timesteps  | 954388   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0207   |
    |    n_updates        | 213596   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.86e+03 |
    |    ep_rew_mean      | 544      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3884     |
    |    fps              | 235      |
    |    time_elapsed     | 4054     |
    |    total_timesteps  | 956202   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0302   |
    |    n_updates        | 214050   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.86e+03 |
    |    ep_rew_mean      | 543      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3888     |
    |    fps              | 235      |
    |    time_elapsed     | 4060     |
    |    total_timesteps  | 957632   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0253   |
    |    n_updates        | 214407   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.88e+03 |
    |    ep_rew_mean      | 548      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3892     |
    |    fps              | 235      |
    |    time_elapsed     | 4065     |
    |    total_timesteps  | 958890   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0515   |
    |    n_updates        | 214722   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.94e+03 |
    |    ep_rew_mean      | 557      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3896     |
    |    fps              | 235      |
    |    time_elapsed     | 4073     |
    |    total_timesteps  | 960996   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0214   |
    |    n_updates        | 215248   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.94e+03 |
    |    ep_rew_mean      | 557      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3900     |
    |    fps              | 235      |
    |    time_elapsed     | 4079     |
    |    total_timesteps  | 962378   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.013    |
    |    n_updates        | 215594   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.95e+03 |
    |    ep_rew_mean      | 560      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3904     |
    |    fps              | 235      |
    |    time_elapsed     | 4086     |
    |    total_timesteps  | 964128   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0479   |
    |    n_updates        | 216031   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.97e+03 |
    |    ep_rew_mean      | 560      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3908     |
    |    fps              | 235      |
    |    time_elapsed     | 4090     |
    |    total_timesteps  | 965259   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0349   |
    |    n_updates        | 216314   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.97e+03 |
    |    ep_rew_mean      | 559      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3912     |
    |    fps              | 235      |
    |    time_elapsed     | 4096     |
    |    total_timesteps  | 966591   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0623   |
    |    n_updates        | 216647   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.98e+03 |
    |    ep_rew_mean      | 564      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3916     |
    |    fps              | 235      |
    |    time_elapsed     | 4101     |
    |    total_timesteps  | 967963   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0326   |
    |    n_updates        | 216990   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.94e+03 |
    |    ep_rew_mean      | 555      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3920     |
    |    fps              | 235      |
    |    time_elapsed     | 4107     |
    |    total_timesteps  | 969150   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0284   |
    |    n_updates        | 217287   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.94e+03 |
    |    ep_rew_mean      | 554      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3924     |
    |    fps              | 235      |
    |    time_elapsed     | 4111     |
    |    total_timesteps  | 970247   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0845   |
    |    n_updates        | 217561   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.95e+03 |
    |    ep_rew_mean      | 557      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3928     |
    |    fps              | 236      |
    |    time_elapsed     | 4117     |
    |    total_timesteps  | 971672   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0101   |
    |    n_updates        | 217917   |
    ----------------------------------
    Eval num_timesteps=975000, episode_reward=592.00 +/- 125.80
    Episode length: 4027.60 +/- 657.56
    ----------------------------------
    | eval/               |          |
    |    mean_ep_length   | 4.03e+03 |
    |    mean_reward      | 592      |
    | rollout/            |          |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    total_timesteps  | 975000   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.00752  |
    |    n_updates        | 218749   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 3.99e+03 |
    |    ep_rew_mean      | 562      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3932     |
    |    fps              | 235      |
    |    time_elapsed     | 4143     |
    |    total_timesteps  | 975253   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0344   |
    |    n_updates        | 218813   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 4e+03    |
    |    ep_rew_mean      | 566      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3936     |
    |    fps              | 235      |
    |    time_elapsed     | 4147     |
    |    total_timesteps  | 976167   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0122   |
    |    n_updates        | 219041   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 4.01e+03 |
    |    ep_rew_mean      | 563      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3940     |
    |    fps              | 235      |
    |    time_elapsed     | 4152     |
    |    total_timesteps  | 977487   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0293   |
    |    n_updates        | 219371   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 4.01e+03 |
    |    ep_rew_mean      | 563      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3944     |
    |    fps              | 235      |
    |    time_elapsed     | 4158     |
    |    total_timesteps  | 979006   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.00837  |
    |    n_updates        | 219751   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 4.01e+03 |
    |    ep_rew_mean      | 562      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3948     |
    |    fps              | 235      |
    |    time_elapsed     | 4162     |
    |    total_timesteps  | 979922   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0747   |
    |    n_updates        | 219980   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 4.02e+03 |
    |    ep_rew_mean      | 568      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3952     |
    |    fps              | 235      |
    |    time_elapsed     | 4167     |
    |    total_timesteps  | 981286   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.008    |
    |    n_updates        | 220321   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 4.02e+03 |
    |    ep_rew_mean      | 567      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3956     |
    |    fps              | 235      |
    |    time_elapsed     | 4174     |
    |    total_timesteps  | 983011   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0119   |
    |    n_updates        | 220752   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 4.04e+03 |
    |    ep_rew_mean      | 575      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3960     |
    |    fps              | 235      |
    |    time_elapsed     | 4181     |
    |    total_timesteps  | 984595   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0259   |
    |    n_updates        | 221148   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 4.05e+03 |
    |    ep_rew_mean      | 575      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3964     |
    |    fps              | 235      |
    |    time_elapsed     | 4188     |
    |    total_timesteps  | 986256   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0209   |
    |    n_updates        | 221563   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 4.07e+03 |
    |    ep_rew_mean      | 584      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3968     |
    |    fps              | 235      |
    |    time_elapsed     | 4196     |
    |    total_timesteps  | 988316   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0167   |
    |    n_updates        | 222078   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 4.07e+03 |
    |    ep_rew_mean      | 584      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3972     |
    |    fps              | 235      |
    |    time_elapsed     | 4201     |
    |    total_timesteps  | 989531   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0155   |
    |    n_updates        | 222382   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 4.07e+03 |
    |    ep_rew_mean      | 587      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3976     |
    |    fps              | 235      |
    |    time_elapsed     | 4205     |
    |    total_timesteps  | 990633   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0174   |
    |    n_updates        | 222658   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 4.08e+03 |
    |    ep_rew_mean      | 590      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3980     |
    |    fps              | 235      |
    |    time_elapsed     | 4213     |
    |    total_timesteps  | 992641   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0164   |
    |    n_updates        | 223160   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 4.06e+03 |
    |    ep_rew_mean      | 590      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3984     |
    |    fps              | 235      |
    |    time_elapsed     | 4217     |
    |    total_timesteps  | 993621   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0287   |
    |    n_updates        | 223405   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 4.06e+03 |
    |    ep_rew_mean      | 592      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3988     |
    |    fps              | 235      |
    |    time_elapsed     | 4225     |
    |    total_timesteps  | 995561   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0106   |
    |    n_updates        | 223890   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 4.09e+03 |
    |    ep_rew_mean      | 598      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3992     |
    |    fps              | 235      |
    |    time_elapsed     | 4233     |
    |    total_timesteps  | 997387   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0208   |
    |    n_updates        | 224346   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 4.09e+03 |
    |    ep_rew_mean      | 598      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 3996     |
    |    fps              | 235      |
    |    time_elapsed     | 4236     |
    |    total_timesteps  | 998283   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0433   |
    |    n_updates        | 224570   |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 4.11e+03 |
    |    ep_rew_mean      | 601      |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    episodes         | 4000     |
    |    fps              | 235      |
    |    time_elapsed     | 4241     |
    |    total_timesteps  | 999570   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0203   |
    |    n_updates        | 224892   |
    ----------------------------------
    Eval num_timesteps=1000000, episode_reward=593.00 +/- 131.51
    Episode length: 3888.00 +/- 237.10
    ----------------------------------
    | eval/               |          |
    |    mean_ep_length   | 3.89e+03 |
    |    mean_reward      | 593      |
    | rollout/            |          |
    |    exploration_rate | 0.01     |
    | time/               |          |
    |    total_timesteps  | 1000000  |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.0237   |
    |    n_updates        | 224999   |
    ----------------------------------
    Saving to logs//dqn/SpaceInvadersNoFrameskip-v4_1



```python
# Eval
!python -m rl_zoo3.enjoy  --algo dqn  --env SpaceInvadersNoFrameskip-v4  --no-render  --n-timesteps 5000  --folder logs/
```

    2025-08-28 12:20:48.399917: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:467] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
    WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
    E0000 00:00:1756383648.422601   23312 cuda_dnn.cc:8579] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
    E0000 00:00:1756383648.429154   23312 cuda_blas.cc:1407] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
    W0000 00:00:1756383648.446607   23312 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
    W0000 00:00:1756383648.446636   23312 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
    W0000 00:00:1756383648.446642   23312 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
    W0000 00:00:1756383648.446647   23312 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
    2025-08-28 12:20:48.451537: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    Gym has been unmaintained since 2022 and does not support NumPy 2.0 amongst other critical functionality.
    Please upgrade to Gymnasium, the maintained drop-in replacement of Gym, or contact the authors of your software and request that they upgrade.
    See the migration guide at https://gymnasium.farama.org/introduction/migration_guide/ for additional information.
    Loading latest experiment, id=1
    Loading logs/dqn/SpaceInvadersNoFrameskip-v4_1/SpaceInvadersNoFrameskip-v4.zip
    A.L.E: Arcade Learning Environment (version 0.11.2+ecc1138)
    [Powered by Stella]
    Stacking 4 frames
    Atari Episode Score: 575.00
    Atari Episode Length 3226
    Atari Episode Score: 600.00
    Atari Episode Length 4069
    Atari Episode Score: 580.00
    Atari Episode Length 3521
    Atari Episode Score: 600.00
    Atari Episode Length 3745
    Atari Episode Score: 465.00
    Atari Episode Length 4083



```python
from huggingface_hub import login
login()
```


    VBox(children=(HTML(value='<center> <img\nsrc=https://huggingface.co/front/assets/huggingface_logo-noborder.sv…



```python
!python -m rl_zoo3.push_to_hub  --algo dqn  --env SpaceInvadersNoFrameskip-v4  --repo-name dqn-SpaceInvadersNoFrameskip-v4  -orga JpChi  -f logs/
```

    2025-08-28 12:22:46.612821: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:467] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
    WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
    E0000 00:00:1756383766.634304   23815 cuda_dnn.cc:8579] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
    E0000 00:00:1756383766.640790   23815 cuda_blas.cc:1407] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
    W0000 00:00:1756383766.658240   23815 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
    W0000 00:00:1756383766.658275   23815 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
    W0000 00:00:1756383766.658279   23815 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
    W0000 00:00:1756383766.658282   23815 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
    2025-08-28 12:22:46.663748: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    Gym has been unmaintained since 2022 and does not support NumPy 2.0 amongst other critical functionality.
    Please upgrade to Gymnasium, the maintained drop-in replacement of Gym, or contact the authors of your software and request that they upgrade.
    See the migration guide at https://gymnasium.farama.org/introduction/migration_guide/ for additional information.
    Loading latest experiment, id=1
    Loading logs/dqn/SpaceInvadersNoFrameskip-v4_1/SpaceInvadersNoFrameskip-v4.zip
    A.L.E: Arcade Learning Environment (version 0.11.2+ecc1138)
    [Powered by Stella]
    Stacking 4 frames
    Wrapping the env in a VecTransposeImage.
    Uploading to JpChi/dqn-SpaceInvadersNoFrameskip-v4, make sure to have the rights
    [38;5;4mℹ This function will save, evaluate, generate a video of your agent,
    create a model card and push everything to the hub. It might take up to some
    minutes if video generation is activated. This is a work in progress: if you
    encounter a bug, please open an issue.[0m
    Fetching 1 files:   0% 0/1 [00:00<?, ?it/s]
    .gitattributes: 1.52kB [00:00, 2.31MB/s]
    Fetching 1 files: 100% 1/1 [00:00<00:00,  1.33it/s]
    Saving model to: hub/dqn-SpaceInvadersNoFrameskip-v4/dqn-SpaceInvadersNoFrameskip-v4
    Saving video to /tmp/tmpjglx1lk_/-step-0-to-step-1000.mp4
    /usr/local/lib/python3.12/dist-packages/moviepy/config_defaults.py:47: SyntaxWarning: invalid escape sequence '\P'
      IMAGEMAGICK_BINARY = r"C:\Program Files\ImageMagick-6.8.8-Q16\magick.exe"
    Moviepy - Building video /tmp/tmpjglx1lk_/-step-0-to-step-1000.mp4.
    Moviepy - Writing video /tmp/tmpjglx1lk_/-step-0-to-step-1000.mp4
    
    Moviepy - Done !
    Moviepy - video ready /tmp/tmpjglx1lk_/-step-0-to-step-1000.mp4
    [38;5;1m✘ 'DummyVecEnv' object has no attribute 'video_recorder'[0m
    [38;5;1m✘ We are unable to generate a replay of your agent, the package_to_hub
    process continues[0m
    [38;5;1m✘ Please open an issue at
    https://github.com/huggingface/huggingface_sb3/issues[0m
    [38;5;4mℹ Pushing repo dqn-SpaceInvadersNoFrameskip-v4 to the Hugging Face
    Hub[0m
    Processing Files (0 / 0)                : |          |  0.00B /  0.00B            
    New Data Upload                         : |          |  0.00B /  0.00B            [A
    
      ...oFrameskip-v4/pytorch_variables.pth: 100% 1.26k/1.26k [00:00<?, ?B/s][A[A
    
    Processing Files (1 / 1)                :   0% 1.26k/54.3M [00:01<12:18:37, 1.22kB/s, 1.58kB/s  ]
    
      ...oFrameskip-v4/pytorch_variables.pth: 100% 1.26k/1.26k [00:00<?, ?B/s][A[A
    
      ...oFrameskip-v4/pytorch_variables.pth: 100% 1.26k/1.26k [00:00<?, ?B/s][A[A
    
    
      ...NoFrameskip-v4/policy.optimizer.pth:   1% 195k/13.5M [00:00<?, ?B/s][A[A[A
    
    
    
      ...ceInvadersNoFrameskip-v4/policy.pth:   1% 195k/13.5M [00:00<?, ?B/s][A[A[A[A
    
    
    
    
      ...dqn-SpaceInvadersNoFrameskip-v4.zip:   1% 393k/27.2M [00:00<?, ?B/s][A[A[A[A[A
    
    
    
    
    
      ...Frameskip-v4/train_eval_metrics.zip:   1% 532/36.8k [00:00<?, ?B/s][A[A[A[A[A[A
    
      ...oFrameskip-v4/pytorch_variables.pth: 100% 1.26k/1.26k [00:00<?, ?B/s][A[A
    
    
      ...NoFrameskip-v4/policy.optimizer.pth:   1% 195k/13.5M [00:00<?, ?B/s][A[A[A
    
    
    
      ...ceInvadersNoFrameskip-v4/policy.pth:   1% 195k/13.5M [00:00<?, ?B/s][A[A[A[A
    
    
    
    
      ...dqn-SpaceInvadersNoFrameskip-v4.zip:   1% 393k/27.2M [00:00<?, ?B/s][A[A[A[A[A
    
    
    
    
    
    Processing Files (1 / 5)                :   1% 785k/54.3M [00:01<01:29, 594kB/s,  561kB/s  ]     
    New Data Upload                         :   1% 779k/53.9M [00:01<01:51, 478kB/s,  557kB/s  ][A
    
      ...oFrameskip-v4/pytorch_variables.pth: 100% 1.26k/1.26k [00:00<?, ?B/s][A[A
    
    
      ...NoFrameskip-v4/policy.optimizer.pth:   1% 195k/13.5M [00:00<?, ?B/s][A[A[A
    
    
    
      ...ceInvadersNoFrameskip-v4/policy.pth:   1% 195k/13.5M [00:00<?, ?B/s][A[A[A[A
    
    
    
    
      ...dqn-SpaceInvadersNoFrameskip-v4.zip:   1% 393k/27.2M [00:00<?, ?B/s][A[A[A[A[A
    
    
    
    
    
      ...Frameskip-v4/train_eval_metrics.zip:   1% 532/36.8k [00:00<?, ?B/s][A[A[A[A[A[A
    
      ...oFrameskip-v4/pytorch_variables.pth: 100% 1.26k/1.26k [00:00<?, ?B/s][A[A
    
    
      ...NoFrameskip-v4/policy.optimizer.pth:   1% 195k/13.5M [00:00<?, ?B/s][A[A[A
    
    
    
      ...ceInvadersNoFrameskip-v4/policy.pth:   1% 195k/13.5M [00:00<?, ?B/s][A[A[A[A
    
    
    
    
      ...dqn-SpaceInvadersNoFrameskip-v4.zip:   1% 393k/27.2M [00:00<?, ?B/s][A[A[A[A[A
    
    
    
    
    
      ...Frameskip-v4/train_eval_metrics.zip:   1% 532/36.8k [00:00<?, ?B/s][A[A[A[A[A[A
    
      ...oFrameskip-v4/pytorch_variables.pth: 100% 1.26k/1.26k [00:01<?, ?B/s][A[A
    
    
      ...NoFrameskip-v4/policy.optimizer.pth:   1% 195k/13.5M [00:00<?, ?B/s][A[A[A
    
    
    
      ...ceInvadersNoFrameskip-v4/policy.pth:   1% 195k/13.5M [00:00<?, ?B/s][A[A[A[A
    
    
    
    
      ...dqn-SpaceInvadersNoFrameskip-v4.zip:   1% 393k/27.2M [00:00<?, ?B/s][A[A[A[A[A
    
    
    
    
    
      ...Frameskip-v4/train_eval_metrics.zip:   1% 532/36.8k [00:00<?, ?B/s][A[A[A[A[A[A
    
      ...oFrameskip-v4/pytorch_variables.pth: 100% 1.26k/1.26k [00:01<?, ?B/s][A[A
    
    
      ...NoFrameskip-v4/policy.optimizer.pth:   3% 390k/13.5M [00:00<00:53, 244kB/s][A[A[A
    
    
    
      ...ceInvadersNoFrameskip-v4/policy.pth:   3% 390k/13.5M [00:00<00:53, 244kB/s][A[A[A[A
    
    
    
    
      ...dqn-SpaceInvadersNoFrameskip-v4.zip:   3% 787k/27.2M [00:00<00:53, 492kB/s][A[A[A[A[A
    
    
    
    
    
    Processing Files (1 / 5)                :   3% 1.57M/54.3M [00:02<01:08, 774kB/s,  713kB/s  ]
    New Data Upload                         :   3% 1.56M/53.9M [00:02<01:16, 683kB/s,  708kB/s  ][A
    
      ...oFrameskip-v4/pytorch_variables.pth: 100% 1.26k/1.26k [00:01<?, ?B/s][A[A
    
    
      ...NoFrameskip-v4/policy.optimizer.pth:  12% 1.56M/13.5M [00:00<00:08, 1.37MB/s][A[A[A
    
    
    
      ...ceInvadersNoFrameskip-v4/policy.pth:  12% 1.56M/13.5M [00:00<00:08, 1.37MB/s][A[A[A[A
    
    
    
    
      ...dqn-SpaceInvadersNoFrameskip-v4.zip:  12% 3.15M/27.2M [00:00<00:08, 2.75MB/s][A[A[A[A[A
    
    
    
    
    
    Processing Files (1 / 5)                :  12% 6.28M/54.3M [00:02<00:11, 4.01MB/s, 2.61MB/s  ]
    New Data Upload                         :  12% 6.23M/53.9M [00:02<00:13, 3.60MB/s, 2.60MB/s  ][A
    
      ...oFrameskip-v4/pytorch_variables.pth: 100% 1.26k/1.26k [00:01<?, ?B/s][A[A
    
    
      ...NoFrameskip-v4/policy.optimizer.pth:  12% 1.56M/13.5M [00:01<00:10, 1.14MB/s][A[A[A
    
    
    
      ...ceInvadersNoFrameskip-v4/policy.pth:  12% 1.56M/13.5M [00:01<00:10, 1.14MB/s][A[A[A[A
    
    
    
    
      ...dqn-SpaceInvadersNoFrameskip-v4.zip:  12% 3.15M/27.2M [00:01<00:10, 2.29MB/s][A[A[A[A[A
    
    
    
    
    
      ...Frameskip-v4/train_eval_metrics.zip:  12% 4.26k/36.8k [00:01<00:10, 3.11kB/s][A[A[A[A[A[A
    
      ...oFrameskip-v4/pytorch_variables.pth: 100% 1.26k/1.26k [00:01<?, ?B/s][A[A
    
    
      ...NoFrameskip-v4/policy.optimizer.pth:  16% 2.15M/13.5M [00:01<00:08, 1.39MB/s][A[A[A
    
    
    
      ...ceInvadersNoFrameskip-v4/policy.pth:  16% 2.15M/13.5M [00:01<00:08, 1.39MB/s][A[A[A[A
    
    
    
    
      ...dqn-SpaceInvadersNoFrameskip-v4.zip:  16% 4.33M/27.2M [00:01<00:08, 2.81MB/s][A[A[A[A[A
    
    
    
    
    
    Processing Files (1 / 5)                :  16% 8.63M/54.3M [00:03<00:10, 4.55MB/s, 3.08MB/s  ]
    New Data Upload                         :  16% 8.57M/53.9M [00:03<00:10, 4.20MB/s, 3.06MB/s  ][A
    
      ...oFrameskip-v4/pytorch_variables.pth: 100% 1.26k/1.26k [00:02<?, ?B/s][A[A
    
    
      ...NoFrameskip-v4/policy.optimizer.pth:  25% 3.32M/13.5M [00:01<00:05, 1.95MB/s][A[A[A
    
    
    
      ...ceInvadersNoFrameskip-v4/policy.pth:  25% 3.32M/13.5M [00:01<00:05, 1.95MB/s][A[A[A[A
    
    
    
    
      ...dqn-SpaceInvadersNoFrameskip-v4.zip:  25% 6.69M/27.2M [00:01<00:05, 3.93MB/s][A[A[A[A[A
    
    
    
    
    
    Processing Files (1 / 5)                :  25% 13.3M/54.3M [00:03<00:05, 7.79MB/s, 4.44MB/s  ]
    New Data Upload                         :  25% 13.2M/53.9M [00:03<00:05, 7.28MB/s, 4.42MB/s  ][A
    
      ...oFrameskip-v4/pytorch_variables.pth: 100% 1.26k/1.26k [00:02<?, ?B/s][A[A
    
    
      ...NoFrameskip-v4/policy.optimizer.pth:  33% 4.49M/13.5M [00:01<00:03, 2.39MB/s][A[A[A
    
    
    
      ...ceInvadersNoFrameskip-v4/policy.pth:  33% 4.49M/13.5M [00:01<00:03, 2.39MB/s][A[A[A[A
    
    
    
    
      ...dqn-SpaceInvadersNoFrameskip-v4.zip:  33% 9.05M/27.2M [00:01<00:03, 4.81MB/s][A[A[A[A[A
    
    
    
    
    
    Processing Files (1 / 5)                :  33% 18.0M/54.3M [00:03<00:03, 10.9MB/s, 5.64MB/s  ]
    New Data Upload                         :  33% 17.9M/53.9M [00:03<00:03, 10.3MB/s, 5.60MB/s  ][A
    
      ...oFrameskip-v4/pytorch_variables.pth: 100% 1.26k/1.26k [00:02<?, ?B/s][A[A
    
    
      ...NoFrameskip-v4/policy.optimizer.pth:  42% 5.66M/13.5M [00:02<00:02, 2.73MB/s][A[A[A
    
    
    
      ...ceInvadersNoFrameskip-v4/policy.pth:  42% 5.66M/13.5M [00:02<00:02, 2.73MB/s][A[A[A[A
    
    
    
    
      ...dqn-SpaceInvadersNoFrameskip-v4.zip:  42% 11.4M/27.2M [00:02<00:02, 5.51MB/s][A[A[A[A[A
    
    
    
    
    
    Processing Files (1 / 5)                :  42% 22.7M/54.3M [00:03<00:02, 13.7MB/s, 6.69MB/s  ]
    New Data Upload                         :  42% 22.6M/53.9M [00:03<00:02, 13.0MB/s, 6.65MB/s  ][A
    
      ...oFrameskip-v4/pytorch_variables.pth: 100% 1.26k/1.26k [00:02<?, ?B/s][A[A
    
    
      ...NoFrameskip-v4/policy.optimizer.pth:  43% 5.86M/13.5M [00:02<00:02, 2.57MB/s][A[A[A
    
    
    
      ...ceInvadersNoFrameskip-v4/policy.pth:  43% 5.86M/13.5M [00:02<00:02, 2.57MB/s][A[A[A[A
    
    
    
    
      ...dqn-SpaceInvadersNoFrameskip-v4.zip:  43% 11.8M/27.2M [00:02<00:02, 5.19MB/s][A[A[A[A[A
    
    
    
    
    
    Processing Files (1 / 5)                :  43% 23.5M/54.3M [00:03<00:02, 11.3MB/s, 6.54MB/s  ]
    New Data Upload                         :  43% 23.4M/53.9M [00:03<00:02, 10.9MB/s, 6.49MB/s  ][A
    
      ...oFrameskip-v4/pytorch_variables.pth: 100% 1.26k/1.26k [00:02<?, ?B/s][A[A
    
    
      ...NoFrameskip-v4/policy.optimizer.pth:  51% 6.83M/13.5M [00:02<00:02, 2.77MB/s][A[A[A
    
    
    
      ...ceInvadersNoFrameskip-v4/policy.pth:  51% 6.83M/13.5M [00:02<00:02, 2.77MB/s][A[A[A[A
    
    
    
    
      ...dqn-SpaceInvadersNoFrameskip-v4.zip:  51% 13.8M/27.2M [00:02<00:02, 5.57MB/s][A[A[A[A[A
    
    
    
    
    
    Processing Files (1 / 5)                :  51% 27.4M/54.3M [00:04<00:01, 13.4MB/s, 7.22MB/s  ]
    New Data Upload                         :  51% 27.3M/53.9M [00:04<00:02, 13.1MB/s, 7.18MB/s  ][A
    
      ...oFrameskip-v4/pytorch_variables.pth: 100% 1.26k/1.26k [00:03<?, ?B/s][A[A
    
    
      ...NoFrameskip-v4/policy.optimizer.pth:  59% 8.00M/13.5M [00:02<00:01, 3.00MB/s][A[A[A
    
    
    
      ...ceInvadersNoFrameskip-v4/policy.pth:  59% 8.00M/13.5M [00:02<00:01, 3.00MB/s][A[A[A[A
    
    
    
    
      ...dqn-SpaceInvadersNoFrameskip-v4.zip:  59% 16.1M/27.2M [00:02<00:01, 6.05MB/s][A[A[A[A[A
    
    
    
    
    
    Processing Files (1 / 5)                :  59% 32.2M/54.3M [00:04<00:01, 16.1MB/s, 8.04MB/s  ]
    New Data Upload                         :  59% 31.9M/53.9M [00:04<00:01, 15.8MB/s, 7.99MB/s  ][A
    
      ...oFrameskip-v4/pytorch_variables.pth: 100% 1.26k/1.26k [00:03<?, ?B/s][A[A
    
    
      ...NoFrameskip-v4/policy.optimizer.pth:  68% 9.17M/13.5M [00:02<00:01, 3.21MB/s][A[A[A
    
    
    
      ...ceInvadersNoFrameskip-v4/policy.pth:  68% 9.17M/13.5M [00:02<00:01, 3.21MB/s][A[A[A[A
    
    
    
    
      ...dqn-SpaceInvadersNoFrameskip-v4.zip:  68% 18.5M/27.2M [00:02<00:01, 6.46MB/s][A[A[A[A[A
    
    
    
    
    
    Processing Files (1 / 5)                :  68% 36.9M/54.3M [00:04<00:00, 18.2MB/s, 8.78MB/s  ]
    New Data Upload                         :  68% 36.6M/53.9M [00:04<00:00, 17.8MB/s, 8.72MB/s  ][A
    
      ...oFrameskip-v4/pytorch_variables.pth: 100% 1.26k/1.26k [00:03<?, ?B/s][A[A
    
    
      ...NoFrameskip-v4/policy.optimizer.pth:  77% 10.3M/13.5M [00:02<00:00, 3.38MB/s][A[A[A
    
    
    
      ...ceInvadersNoFrameskip-v4/policy.pth:  77% 10.3M/13.5M [00:02<00:00, 3.38MB/s][A[A[A[A
    
    
    
    
      ...dqn-SpaceInvadersNoFrameskip-v4.zip:  77% 20.8M/27.2M [00:02<00:00, 6.82MB/s][A[A[A[A[A
    
    
    
    
    
    Processing Files (1 / 5)                :  77% 41.6M/54.3M [00:04<00:00, 19.7MB/s, 9.45MB/s  ]
    New Data Upload                         :  77% 41.3M/53.9M [00:04<00:00, 19.4MB/s, 9.39MB/s  ][A
    
      ...oFrameskip-v4/pytorch_variables.pth: 100% 1.26k/1.26k [00:03<?, ?B/s][A[A
    
    
      ...NoFrameskip-v4/policy.optimizer.pth:  85% 11.5M/13.5M [00:03<00:00, 3.54MB/s][A[A[A
    
    
    
      ...ceInvadersNoFrameskip-v4/policy.pth:  85% 11.5M/13.5M [00:03<00:00, 3.54MB/s][A[A[A[A
    
    
    
    
      ...dqn-SpaceInvadersNoFrameskip-v4.zip:  85% 23.2M/27.2M [00:03<00:00, 7.13MB/s][A[A[A[A[A
    
    
    
    
    
    Processing Files (1 / 5)                :  85% 46.3M/54.3M [00:04<00:00, 20.8MB/s, 10.1MB/s  ]
    New Data Upload                         :  85% 46.0M/53.9M [00:04<00:00, 20.5MB/s, 9.99MB/s  ][A
    
      ...oFrameskip-v4/pytorch_variables.pth: 100% 1.26k/1.26k [00:03<?, ?B/s][A[A
    
    
      ...NoFrameskip-v4/policy.optimizer.pth:  94% 12.7M/13.5M [00:03<00:00, 3.67MB/s][A[A[A
    
    
    
      ...ceInvadersNoFrameskip-v4/policy.pth:  94% 12.7M/13.5M [00:03<00:00, 3.67MB/s][A[A[A[A
    
    
    
    
      ...dqn-SpaceInvadersNoFrameskip-v4.zip:  94% 25.6M/27.2M [00:03<00:00, 7.40MB/s][A[A[A[A[A
    
    
    
    
    
    Processing Files (1 / 5)                :  94% 51.0M/54.3M [00:05<00:00, 21.6MB/s, 10.6MB/s  ]
    New Data Upload                         :  94% 50.6M/53.9M [00:05<00:00, 21.3MB/s, 10.6MB/s  ][A
    
      ...oFrameskip-v4/pytorch_variables.pth: 100% 1.26k/1.26k [00:04<?, ?B/s][A[A
    
    
      ...NoFrameskip-v4/policy.optimizer.pth: 100% 13.5M/13.5M [00:03<00:00, 3.69MB/s][A[A[A
    
    
    
      ...ceInvadersNoFrameskip-v4/policy.pth: 100% 13.5M/13.5M [00:03<00:00, 3.69MB/s][A[A[A[A
    
    
    
    
      ...dqn-SpaceInvadersNoFrameskip-v4.zip: 100% 27.1M/27.2M [00:03<00:00, 7.43MB/s][A[A[A[A[A
    
    
    
    
    
    Processing Files (1 / 5)                : 100% 54.1M/54.3M [00:05<00:00, 19.8MB/s, 10.8MB/s  ]
    New Data Upload                         : 100% 53.8M/53.9M [00:05<00:00, 19.7MB/s, 10.8MB/s  ][A
    
      ...oFrameskip-v4/pytorch_variables.pth: 100% 1.26k/1.26k [00:04<?, ?B/s][A[A
    
    
      ...NoFrameskip-v4/policy.optimizer.pth: 100% 13.5M/13.5M [00:03<00:00, 3.49MB/s][A[A[A
    
    
    
      ...ceInvadersNoFrameskip-v4/policy.pth: 100% 13.5M/13.5M [00:03<00:00, 3.49MB/s][A[A[A[A
    
    
    
    
      ...dqn-SpaceInvadersNoFrameskip-v4.zip: 100% 27.1M/27.2M [00:03<00:00, 7.04MB/s][A[A[A[A[A
    
    
    
    
    
      ...Frameskip-v4/train_eval_metrics.zip: 100% 36.7k/36.8k [00:03<00:00, 9.52kB/s][A[A[A[A[A[A
    
      ...oFrameskip-v4/pytorch_variables.pth: 100% 1.26k/1.26k [00:04<?, ?B/s][A[A
    
    
      ...NoFrameskip-v4/policy.optimizer.pth: 100% 13.5M/13.5M [00:04<00:00, 3.32MB/s][A[A[A
    
    
    
      ...ceInvadersNoFrameskip-v4/policy.pth: 100% 13.5M/13.5M [00:04<00:00, 3.32MB/s][A[A[A[A
    
    
    
    
      ...dqn-SpaceInvadersNoFrameskip-v4.zip: 100% 27.1M/27.2M [00:04<00:00, 6.69MB/s][A[A[A[A[A
    
    
    
    
    
      ...Frameskip-v4/train_eval_metrics.zip: 100% 36.7k/36.8k [00:04<00:00, 9.05kB/s][A[A[A[A[A[A
    
      ...oFrameskip-v4/pytorch_variables.pth: 100% 1.26k/1.26k [00:04<?, ?B/s][A[A
    
    
      ...NoFrameskip-v4/policy.optimizer.pth: 100% 13.5M/13.5M [00:04<00:00, 3.16MB/s][A[A[A
    
    
    
      ...ceInvadersNoFrameskip-v4/policy.pth: 100% 13.5M/13.5M [00:04<00:00, 3.16MB/s][A[A[A[A
    
    
    
    
      ...dqn-SpaceInvadersNoFrameskip-v4.zip: 100% 27.1M/27.2M [00:04<00:00, 6.37MB/s][A[A[A[A[A
    
    
    
    
    
      ...Frameskip-v4/train_eval_metrics.zip: 100% 36.7k/36.8k [00:04<00:00, 8.62kB/s][A[A[A[A[A[A
    
      ...oFrameskip-v4/pytorch_variables.pth: 100% 1.26k/1.26k [00:04<?, ?B/s][A[A
    
    
      ...NoFrameskip-v4/policy.optimizer.pth: 100% 13.5M/13.5M [00:04<00:00, 3.03MB/s][A[A[A
    
    
    
      ...ceInvadersNoFrameskip-v4/policy.pth: 100% 13.5M/13.5M [00:04<00:00, 3.03MB/s][A[A[A[A
    
    
    
    
      ...dqn-SpaceInvadersNoFrameskip-v4.zip: 100% 27.2M/27.2M [00:04<00:00, 6.10MB/s][A[A[A[A[A
    
    
    
    
    
    Processing Files (5 / 5)                : 100% 54.3M/54.3M [00:06<00:00, 7.54MB/s, 9.36MB/s  ]
    New Data Upload                         : 100% 53.9M/53.9M [00:06<00:00, 7.48MB/s, 9.30MB/s  ][A
    
      ...oFrameskip-v4/pytorch_variables.pth: 100% 1.26k/1.26k [00:05<?, ?B/s][A[A
    
    
      ...NoFrameskip-v4/policy.optimizer.pth: 100% 13.5M/13.5M [00:04<00:00, 2.89MB/s][A[A[A
    
    
    
      ...ceInvadersNoFrameskip-v4/policy.pth: 100% 13.5M/13.5M [00:04<00:00, 2.89MB/s][A[A[A[A
    
    
    
    
      ...dqn-SpaceInvadersNoFrameskip-v4.zip: 100% 27.2M/27.2M [00:04<00:00, 5.83MB/s][A[A[A[A[A
    
    
    
    
    
      ...Frameskip-v4/train_eval_metrics.zip: 100% 36.8k/36.8k [00:04<00:00, 7.89kB/s][A[A[A[A[A[A
    
      ...oFrameskip-v4/pytorch_variables.pth: 100% 1.26k/1.26k [00:05<?, ?B/s][A[A
    
    
      ...NoFrameskip-v4/policy.optimizer.pth: 100% 13.5M/13.5M [00:04<00:00, 2.85MB/s][A[A[A
    
    
    
      ...ceInvadersNoFrameskip-v4/policy.pth: 100% 13.5M/13.5M [00:04<00:00, 2.85MB/s][A[A[A[A
    
    
    
    
      ...dqn-SpaceInvadersNoFrameskip-v4.zip: 100% 27.2M/27.2M [00:04<00:00, 5.73MB/s][A[A[A[A[A
    
    
    
    
    
      ...Frameskip-v4/train_eval_metrics.zip: 100% 36.8k/36.8k [00:04<00:00, 7.76kB/s][A[A[A[A[A[A
    
      ...oFrameskip-v4/pytorch_variables.pth: 100% 1.26k/1.26k [00:05<?, ?B/s][A[A
    
    
      ...NoFrameskip-v4/policy.optimizer.pth: 100% 13.5M/13.5M [00:04<00:00, 2.77MB/s][A[A[A
    
    
    
      ...ceInvadersNoFrameskip-v4/policy.pth: 100% 13.5M/13.5M [00:04<00:00, 2.77MB/s][A[A[A[A
    
    
    
    
      ...dqn-SpaceInvadersNoFrameskip-v4.zip: 100% 27.2M/27.2M [00:04<00:00, 5.59MB/s][A[A[A[A[A
    
    
    
    
    
    Processing Files (5 / 5)                : 100% 54.3M/54.3M [00:06<00:00, 8.44MB/s, 8.75MB/s  ]
    New Data Upload                         : 100% 53.9M/53.9M [00:06<00:00, 8.39MB/s, 8.70MB/s  ]
      ...oFrameskip-v4/pytorch_variables.pth: 100% 1.26k/1.26k [00:05<?, ?B/s]
      ...NoFrameskip-v4/policy.optimizer.pth: 100% 13.5M/13.5M [00:04<00:00, 2.77MB/s]
      ...ceInvadersNoFrameskip-v4/policy.pth: 100% 13.5M/13.5M [00:04<00:00, 2.77MB/s]
      ...dqn-SpaceInvadersNoFrameskip-v4.zip: 100% 27.2M/27.2M [00:04<00:00, 5.59MB/s]
      ...Frameskip-v4/train_eval_metrics.zip: 100% 36.8k/36.8k [00:04<00:00, 7.56kB/s]
    [38;5;4mℹ Your model is pushed to the hub. You can view your model here:
    https://huggingface.co/JpChi/dqn-SpaceInvadersNoFrameskip-v4[0m



```python

```
