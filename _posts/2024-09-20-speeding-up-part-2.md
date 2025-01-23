First few things to check:

1. What hardware is available
2. What's the capability
3. Is the capability used to max

In this blog am using A100 80GB SXM from https://www.runpod.io.

## Table of Contents

*Maximizing Hardware*
* What are Tensor Cores?
    * TF32
    * Float16
* torch.compile
    * High level overview of Memory architecture
    * Flash Attention
    * Nice/Ugly Numbers

*Algorithmic Speed Up*
* Optimizer and Gradient Clipping
* Cosine Decay Scheduler
* Batch size
* Data Sampling
* Weight Decay
* Gradient Accumulation
* Loss normalization intuition

*Distributed Data Parallel*
* How DDP Works?
    * Implementation in code
    * DataLoader Update
    * Model Update
    * Training Loop Update
* DDP Materials

*Dataset Update*
* Dataset Creation
* DataLoader Update
* Training Loop Update

*Text Generation*
* Validation loss
* HellaSwag Evaluation
    * Render sample
    * Evaluation
* Training loop update
*

## Maximizing Hardware

Sample GPU specs(*diag1*):
<img src="/images/reproduce-gpt/ax100sxm.png">

* As we reduce the size of datatype used by parameters in a deep network model number of TFLOPS(matrix operations) increases.
* Neural network training can work with lower precision types FP16, FP32 etc.
* Due to Memory Bandwidth limitations, most memory will be idle. Because the tensors or operations has to be moved inside memory, perform operation and move them out. With reduced datatype this access speed increases.
* Lesser datatype precision, weights, biases, activations requires less memory and faster access.

### What are TensorCore?

TensorCore is an architecture to speed up matrix multiplication by Nvidia. 
* They came into existence from Votla series, Turing, Ampere, Hopper(Lates as of 2024)
* They perform a 4* 4 matrix multiplication and addition. Assume a matrix multiplication of dot product. It goes like summation of rows and columns. This operation is done in parallel for all rows and columns using TensorCore. [Checkout this visualization](https://blog.paperspace.com/understanding-tensor-cores/#how-do-tensor-cores-work). 
* Mixed precision(FP16 for lower layers and higher precision for top layers), these are like little knobs(FP16, FP32, TF16, TF32) that can be used.

### TF32
*diag2*
<img src="/images/reproduce-gpt/a100-architecture-precison-types.png">



* If we use TF32, 19 bits is truncated increasing the number of mat mul operations.
* We lose some precision but this is kind of sweet spot to train neural networks.
* Output is TF32, Output is FP32, Accumulator is FP32 only the internal operations are in TF32, we get a 8X increase in performance.
* The code won't see this change or has to do this change everything happens internally and we can reap 8X speed rewards.
* This change is local to the operation itself

To see the improvement in performance with GPU we've added timing training loop after a single step and more important metric like tokens per second.
* `torch.cuda.synchronize()` is ran after optimizer step to wait until all instruction sent from cpu to gpu are completed before getting the end time.

*Timing sample:*
```Python
# Initialize training loop
for i in range(50):
	t0 = time.time()
	x, y = data_loader.next_batch()
	x, y = x.to(device), y.to(device)
	# Optimizer zero grad
	optim.zero_grad(set_to_none=True)
	# Forward pass
	logits, loss = model(x, y)
	# Backward pass
	loss.backward()
	# Update parameters
	optim.step()
	# Wait until all instruction sent from cpu to gpu are completed
	torch.cuda.synchronize()
	t1 = time.time()
	dt = (t1 - t0)*1000 # ms
	print(f"Step {i}, loss: {loss.item():.4f}, dt: {dt:.2f}ms, tokens/sec: {B*T/t1-t0:.2f}")
	losses.append(loss.item())
print(losses)
```

*Logs without TF32 on A100SXM GPU* - Baseline
```logs
Step 0, loss: 10.9723, dt: 1173.49ms, tokens/sec: 13961.74
Step 1, loss: 9.4105, dt: 1006.55ms, tokens/sec: 16277.31
Step 2, loss: 8.9783, dt: 1005.89ms, tokens/sec: 16288.03
Step 3, loss: 8.7954, dt: 1008.09ms, tokens/sec: 16252.60
Step 4, loss: 8.4695, dt: 1003.95ms, tokens/sec: 16319.58
Step 5, loss: 8.4083, dt: 1005.01ms, tokens/sec: 16302.33
Step 6, loss: 8.2694, dt: 1004.14ms, tokens/sec: 16316.50
Step 7, loss: 8.0184, dt: 1005.21ms, tokens/sec: 16299.10
Step 8, loss: 7.7258, dt: 1003.80ms, tokens/sec: 16322.00
Step 9, loss: 7.5069, dt: 1005.11ms, tokens/sec: 16300.69
Step 10, loss: 7.3261, dt: 1004.34ms, tokens/sec: 16313.14
```

*How to set float32 tensorcore precision*: https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
* `torch.set_float32_matmul_precision("high")` with this line of code above training loop TF32 will be enabled. 
* TF32 is available only depends on GPU used.

*Logs with TF32 on A100SXM GPU*
```logs
Step 0, loss: 10.9861, dt: 506.31ms, tokens/sec: 32359.73
Step 1, loss: 9.4392, dt: 334.88ms, tokens/sec: 48924.73
Step 2, loss: 9.0096, dt: 334.27ms, tokens/sec: 49013.71
Step 3, loss: 8.8273, dt: 334.90ms, tokens/sec: 48921.84
Step 4, loss: 8.4655, dt: 335.23ms, tokens/sec: 48874.34
Step 5, loss: 8.4109, dt: 335.33ms, tokens/sec: 48859.37
Step 6, loss: 8.2472, dt: 335.73ms, tokens/sec: 48800.87
Step 7, loss: 8.0185, dt: 334.70ms, tokens/sec: 48950.87
Step 8, loss: 7.7368, dt: 334.71ms, tokens/sec: 48950.45
Step 9, loss: 7.5093, dt: 335.37ms, tokens/sec: 48853.15
```

From baseline 33% decrease in time per step and 33% increase in tokens per sec.
### Float16

* Even with TF32, we've still not using GPU to it's max, because the parameter size are still Float32 and requires higher memory bandwidth to move them in and out of memory.
* We can improve this by moving more parameters to FLOAT16 or BFLOAT16 in diag1.
	* There are two options for Float16 in A100 GPU
		* If we use FP16, we require Gradient autoscalers. Because in diag2 you can see the range(exponent) gets reduces, this adds an additional layer in training loop. *FP16 cannot represent entire range of FP32*. FP16 came first and BF16 came later in ampere. This is the reason behind usage of Gradient scalers in FP16. (Refer diag2)
		* With BF16, Gradient scalers are not requires as range of FP32 is preserved and Precision bits are reduced further. This eliminates the need for Gradient Scalers but has reduced precision.
		* The changes in BF16 are not local like FP32, it affects the tensors in PyTorch.
* We'll use BF16 in our code.
	* This can be implemented with https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html
	* torch.autocast can be done with autocast context manager and should be applied only to logits and loss. This doesn't require casting model or inputs to BF16 dtype. https://pytorch.org/docs/stable/amp.html#torch.autocast
	* https://pytorch.org/docs/stable/amp.html#id8 shows which operations uses BF16 and others that uses FP32. PyTorch implements this mixed precision in the background and documentation is not that clear.
	* Operations like matmul, matadd(less susceptible to reduced precision) are in BF16 and other susceptible layers like Linear, LayerNorm, activations are still in FP32.
* With BF16 we'll trade accuracy/precision decrease for increased operations.
```Python
torch.set_float32_matmul_precision("high")
print("Running all data overfit training loop")
# create dataloader
data_loader = DataLoaderLite(input_file="input.txt", B=B, T=T)
# Initialize model
losses = []
model = GPT2(GPTConfig).to(device)
# Initialize optimizer
optim = torch.optim.AdamW(
	params=model.parameters(), # Parameters for backprop
	lr=3e-4, # This is good initial learning rate
)

# Initialize training loop
for i in range(50):
	t0 = time.time()
	x, y = data_loader.next_batch()
	x, y = x.to(device), y.to(device)
	# Optimizer zero grad
	optim.zero_grad(set_to_none=True)
	# Forward pass
	with torch.autocast(device_type=device, dtype=torch.bfloat16):
		logits, loss = model(x, y)
	# Backward pass
	loss.backward()
	# Update parameters
	optim.step()
	# Wait until all instruction sent from cpu to gpu are completed
	if device == "cuda":
	torch.cuda.synchronize()
	t1 = time.time()
	dt = (t1 - t0)*1000 # ms
	print(f"Step {i}, loss: {loss.item():.4f}, dt: {dt:.2f}ms, tokens/sec: {(B*T)/(t1-t0):.2f}")
	losses.append(loss.item())
print(losses)
```

```logs
Step 0, loss: 10.9590, dt: 519.29ms, tokens/sec: 31550.67
Step 1, loss: 9.3671, dt: 304.54ms, tokens/sec: 53799.90
Step 2, loss: 9.1510, dt: 303.77ms, tokens/sec: 53935.40
Step 3, loss: 8.7488, dt: 304.13ms, tokens/sec: 53872.36
Step 4, loss: 8.4597, dt: 303.66ms, tokens/sec: 53954.25
Step 5, loss: 8.3837, dt: 303.54ms, tokens/sec: 53976.20
Step 6, loss: 8.2138, dt: 304.74ms, tokens/sec: 53763.37
Step 7, loss: 7.9817, dt: 304.20ms, tokens/sec: 53858.77
Step 8, loss: 7.7324, dt: 304.27ms, tokens/sec: 53847.67
Step 9, loss: 7.4918, dt: 304.10ms, tokens/sec: 53877.13
Step 10, loss: 7.2896, dt: 305.09ms, tokens/sec: 53702.19
```

10% decrease in time per step and 10x increase in tokens per sec
### torch.compile

*torch.compile significantly increases speedup from reducing Python overhead and GPU read/writes.*

```Python
model = torch.compile(model)
```

* Without compile, Python interpreter runs the forward pass of GPT2 model line by line.
* With torch.compile:
	* Python interpereter:
		* PyTorch understand the entirety of operations that python compiler doesn't understand and then optimizes these process
		* It takes out python interpreter of the entire forward pass, it knows what to run and runs them in efficient code.
	* GPU Read/Writes:
		<img src="./images/reproduce-gpt/gpu-read-writes.png">

	
		```Python
def gelu(x): 
	""" Implements the GELU activation function. Args: x: Input tensor. Returns: Output tensor after applying GELU activation. """ 
	# Approximate GELU using a polynomial approximation 
	cdf = 0.5 * (1.0 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0)))) return x * cdf
		```
		* x is stored in gpu memory(HBM)
		* First the torch.pow from right hand is performed. x travels to GPU(cores) calculates power saves it back to memory. 
		* Next x + 0.044715 will be performed in GPU core and then the value will be moved to be stored in HBM.
		* *This travel time costs lot's of issues. Everything is extremely fast inside GPU but the travel time takes extremely long amount of time*
	* Without torch.compile PyTorch doesn't know the single line is just a bunch of element wise operations on a single input(x) and performs a lot's of round trips to compute cdf.
	* With torch.compile, PyTorch has the benefit of hindsight(knows what operations needs to be performed on input). When x is moved to GPU it performs all the operations and sends the result back to HBM.
--> Above steps from function is a brief example of what PyTorch does to reduce GPU read/writes when model compile occurs.

*It breaks down the operations that's gonna happen on code execution into subgraphs. Compiles and flattens them, assign each PyTorch operations into their chosen backend-specific kernels. And the kernels call their corresponding low-level device operations*

After adding:
```Python
model = torch.compile(model)
```
```logs
Step 0, loss: 10.9333, dt: 35959.98ms, tokens/sec: 455.62
Step 1, loss: 9.4089, dt: 174.39ms, tokens/sec: 93951.33
Step 2, loss: 9.0968, dt: 173.50ms, tokens/sec: 94434.19
Step 3, loss: 8.7862, dt: 173.54ms, tokens/sec: 94410.84
Step 4, loss: 8.5005, dt: 174.61ms, tokens/sec: 93829.33
Step 5, loss: 8.4456, dt: 173.76ms, tokens/sec: 94290.75
Step 6, loss: 8.2564, dt: 173.78ms, tokens/sec: 94279.50
Step 7, loss: 8.0227, dt: 173.56ms, tokens/sec: 94402.15
Step 8, loss: 7.7540, dt: 173.59ms, tokens/sec: 94383.09
Step 9, loss: 7.5099, dt: 174.44ms, tokens/sec: 93923.72
```

17% reduction in time taken per step and  increase tokens/sec

#### High level overview of Memory architecture
*diag4* - Indepth view of single GPu core in diag3
<img src="/images/reproduce-gpt/ga100-full-gpu-128-sms.png">


* HBM and above GPU core are different chips.
*diag5* - single SM inside diag4 GPU core
<img src="/images/reproduce-gpt/single-sm-a100.png">


* There is some memory but not a lot inside the GPU Chip
* The memory inside SM is really really fast to access but it goes from ram to cpu to gpu
* with torch compile when input lives on GPU, we perform all the operations using operator fusion/kernel fusion and reduces the round trips. It uses the small but extremely fast memory inside GPU's.
* ***Kernel fusion***. The most common approach to accelerate memory-bound operations is kernel fusion: if there are multiple operations applied to the same input, the input can be loaded once from HBM, instead of multiple times for each operation.[Materials 5]

In General, Compute(processings inside GPU cores/SM) is faster but access to Compute is limited by memory. As data has to go something like below:

*diag6* - Memory Hierarchy with Bandwidth speeds.
<img src="/images/reproduce-gpt/mem-hirearchy-with-bandwidth-size.png">


### Flash Attention

[Flash Attention Paper has lot's of curious info](https://arxiv.org/pdf/2205.14135)

Here's a short brief on the paper:

There are lot's of operation torch compile can find but flash attention is not one of them. Flash attention is an algorithm to perform attention lot faster.

Bottlenecks of Attention:

Excerpt from Flash Attention Paper:
<img src="/images/reproduce-gpt/standard-attention-implementation.png">


Normal attention has many read writes for very large N * N matrix multiplications in attention mechanism. Here N * N referes to Key, Query, Value, attention scores calculation, Attention weights calculation with softmax and final attention values. ***Flash attention avoid the creation of N *  N matrices.*** 

FlashAttention Overcomes this by performing an algorithmic rewrite of attention mechanism in memory efficient manner. Below Excerpt from Flash Attention Paper:

***Implementation details: Kernel fusion. Tiling enables us to implement our algorithm in one CUDA kernel, loading input from HBM, performing all the computation steps (matrix multiply, softmax, optionally masking and dropout, matrix multiply), then write the result back to HBM (masking and dropout in Appendix B). This avoids repeatedly reading and writing of inputs and outputs from and to HBM.***

In code we can achieve this by replacing QKV matrix operations with `torch.nn.functional.scaled_dot_product_attention(q,k,v,is_causal=True)`.

```logs
Step 0, loss: 10.9507, dt: 25867.14ms, tokens/sec: 633.39
Step 1, loss: 9.4273, dt: 144.73ms, tokens/sec: 113207.00
Step 2, loss: 9.0978, dt: 144.29ms, tokens/sec: 113548.00
Step 3, loss: 8.7908, dt: 144.18ms, tokens/sec: 113636.25
Step 4, loss: 8.5107, dt: 145.06ms, tokens/sec: 112944.65
Step 5, loss: 8.4376, dt: 144.71ms, tokens/sec: 113217.25
Step 6, loss: 8.2681, dt: 145.24ms, tokens/sec: 112804.11
Step 7, loss: 8.0488, dt: 144.47ms, tokens/sec: 113405.03
Step 8, loss: 7.7589, dt: 145.16ms, tokens/sec: 112866.92
Step 9, loss: 7.5214, dt: 145.00ms, tokens/sec: 112995.35
```

12x reduction in time taken per step and  increase tokens/sec

### Nice/Ugly Numbers

* Since Neural Networks are trained with CUDA. CUDA call's kernels to execute computations on tiles or blocks GPU's, these blocks perform computations in powers of 32, 16, 64 etc. and not odd numbers. Any parameters in neural networks like n_layers, n_heads, vocab_size must be in  powers or of 2 for improved computations. If odd numbers are used, kernels will call these blocks to execute nice computations until powers of 2 and other computation will be executed by additional kernels.
* We can update 50257 vocab to 50304, this is divisible by 16,32,64. This will lead to improvement in performance even though we're adding additional tokens. Like unicode logits are pushed to -inf these additional dummy tokens logits will also be learned by the model.

## Algorithmic Speed Up

In this section, we'll use model training parameters from GPT3 to improve our model.

### Optimizer and Gradient Clipping

In Appendix B Details of Model Training of GPT3 paper:
* Uses Adam optimizer with betas 0.9 and 0.95. 
* epsilon - $10^{-8}$
* clip global norm of gradient to 1.0. Global norm is $\sqrt{\sum_{i=1}^{n} g_i^2}$ where $g_i$ is gradients of all parameters in the model. This prevents gradients from exploding. If you get a unlucky batch which leads to high loss and in turn high gradients. This prevents the model from shock due to high gradients. 

```logs
Step    0 | loss: 10.952483 | norm: 2.2098 | dt: 28304.86ms | tokens/sec: 578.84
Step    1 | loss: 9.361493 | norm: 2.3879 | dt: 113.24ms | tokens/sec: 144684.77
Step    2 | loss: 9.096659 | norm: 3.5505 | dt: 112.44ms | tokens/sec: 145708.49
Step    3 | loss: 8.781656 | norm: 2.5740 | dt: 113.52ms | tokens/sec: 144331.35
Step    4 | loss: 8.523396 | norm: 2.7177 | dt: 112.22ms | tokens/sec: 145999.18
Step    5 | loss: 8.461228 | norm: 2.2171 | dt: 112.62ms | tokens/sec: 145476.22
Step    6 | loss: 8.283797 | norm: 1.5492 | dt: 112.47ms | tokens/sec: 145678.22
Step    7 | loss: 8.063981 | norm: 1.9874 | dt: 112.85ms | tokens/sec: 145183.32
Step    8 | loss: 7.794333 | norm: 2.1738 | dt: 112.51ms | tokens/sec: 145617.41
Step    9 | loss: 7.555257 | norm: 1.6877 | dt: 112.40ms | tokens/sec: 145761.03
```
 20x reduction in time per step and 23x increase in tokens per sec
### Cosine Decay Scheduler

* Linear warm up over first 375M tokens and a cosine decay until it reaches 10% of learning rate and continue at 10% lr. We'll set a similar scheme for our training:
	* Linear warm up until 10 steps
	* cosine decay until a certain limit
	* and continue
	* max_lr from GPT3 paper for GPT-small is $6 * 10^{-4}$, our's is right now $3 * 10^{-4}$

```Python
import math
max_lr = 10
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 50
lrs = []
coeffs = []
drs = []

def get_lr(it):
	if it < warmup_steps:
		# Starts from min_lr to max_lr during warmup
		lr = max_lr * (it + 1) / warmup_steps
		print(lr)
		lrs.append(lr)
		return lr

	if it > max_steps:
		# After max steps use min_lr
		print(min_lr)
		lrs.append(min_lr)
		return min_lr

	# Betweem warmup_steps and max_steps use cosine annealing
	# Decay ratio increases from 0 to 1

	decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
	drs.append(decay_ratio)
	print(f"Decay ratio: {decay_ratio}")
	# math.cos(math.pi * decay_ratio) products value from -1 to 1 based on decay_ratio
	# 1.0 is added to shift value from -1 to 1 to 1 to 2
	# 0.5 is multiplied to shift value from 1 to 2 to 0 to 1, coeff starts from 1 and decreases until 0
	# As decay ration increases coeff decreases, thus reducing learning rate
	coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
	coeffs.append(coeff)
	print(f"Coeff: {coeff}")
	lr = min_lr + (max_lr - min_lr) * coeff
	print(f"Lr calculation: {min_lr} + ({max_lr} - {min_lr}) * {coeff} = {lr}")
	print(lr)
	lrs.append(lr)
	return lr
```

<img src="/images/reproduce-gpt/cosine-learning-rate-scheduler.png">


* warmup until max learning rate with step
* Coefficient decreases with decay increase. Coefficient decreases in a cosine manner with respect to decay.

```logs
Step    0 | loss: 10.988498 | lr: 6.0000e-05 | norm: 2.1887 | dt: 26797.96ms | tokens/sec: 611.39
Step    1 | loss: 10.380230 | lr: 1.2000e-04 | norm: 2.6468 | dt: 113.89ms | tokens/sec: 143857.59
Step    2 | loss: 9.592928 | lr: 1.8000e-04 | norm: 2.7015 | dt: 113.40ms | tokens/sec: 144473.97
Step    3 | loss: 9.329897 | lr: 2.4000e-04 | norm: 6.0671 | dt: 113.08ms | tokens/sec: 144893.43
Step    4 | loss: 8.835960 | lr: 3.0000e-04 | norm: 2.0631 | dt: 112.60ms | tokens/sec: 145511.03
Step    5 | loss: 8.772915 | lr: 3.6000e-04 | norm: 2.5251 | dt: 112.87ms | tokens/sec: 145157.26
Step    6 | loss: 8.536877 | lr: 4.2000e-04 | norm: 1.8342 | dt: 112.65ms | tokens/sec: 145446.97
Step    7 | loss: 8.236061 | lr: 4.8000e-04 | norm: 1.9982 | dt: 112.69ms | tokens/sec: 145391.27
Step    8 | loss: 7.816847 | lr: 5.4000e-04 | norm: 1.7764 | dt: 112.64ms | tokens/sec: 145451.59
Step    9 | loss: 7.463247 | lr: 6.0000e-04 | norm: 1.3857 | dt: 112.65ms | tokens/sec: 145447.28
```

* No improvement with learning rate schedule.

### Batch size

Excerpt from GPT3 paper
Gradually increased batch size from a small value(32k tokens) to the full value over the first 4-12 billion tokens of training, depending on model size.

We're gonna skip this due to below reasons:
* It complicates the arithmetic as it changes the batch size for every step of training.
* The model learning what tokens come next in the initial stages of training is the same if we use a small batch size or a huge batch size. Hence there's not much advantage in increasing the batch size to learn the same thing that can be learned by the neural net over small batch sizes.
* The gradients of different sequences at initial stages are correlated due to the nature of neural net learning to predict what tokens comes next. Once this is learned then at the later stages gradients become de-correlated as more details from sequences are learned.

### Data Sampling

Excerpt from paper:
Data are sampled without replacement during training(until an epoch boundary is reached) to minimize overfitting.

* This means an sequence is not returned to the pool to be used in the same step. If we use the same sample in a single step multiple times, the neural net might overfit(memorize the sequence)
* We're already doing this with our DataLoader as we're loading chunks of tokens and same chunk won't come in until the next step.

### Weight Decay

Excerpt from paper:
All models use weight decay of 0.1 to provide a small amount of regularization.

What's weight decay: Pull down the weights by weight decay param to make optimization to use all of the weights and not allow any of the individual weights to be way too large.

How Regularization is applied:
1. Applied to weights > 2dim, embeddings
2. Not applied to weights < 1dim and biases

 Fused optimizer can call a single fused kernel to update all the parameters in the network.

We'll implement a function to return an AdamW optimizer with above functionalities.

```gist
https://gist.github.com/JpChii/d8baf0ef276a1e6d6d40f14d8650e154
```


```logs
sing device: cuda
Running all data overfit training loop
Number of tokens: 338025
Total number of batches per epoch: 20
Weights intializations started
Weights intiailization complete
Number of decayed parameter tensors: 50, (124354560)
Number of non-decayed parameter tensors: 98, (121344)
Step    0 | loss: 11.003080 | lr: 6.0000e-05 | norm: 2.1814 | dt: 32620.00ms | tokens/sec: 502.27
Step    1 | loss: 10.401819 | lr: 1.2000e-04 | norm: 2.6339 | dt: 119.02ms | tokens/sec: 137655.35
Step    2 | loss: 9.607843 | lr: 1.8000e-04 | norm: 2.4026 | dt: 109.62ms | tokens/sec: 149465.87
Step    3 | loss: 9.368017 | lr: 2.4000e-04 | norm: 6.2099 | dt: 109.31ms | tokens/sec: 149879.56
Step    4 | loss: 8.888070 | lr: 3.0000e-04 | norm: 2.5425 | dt: 109.66ms | tokens/sec: 149403.16
Step    5 | loss: 8.792170 | lr: 3.6000e-04 | norm: 2.4993 | dt: 109.78ms | tokens/sec: 149249.03
Step    6 | loss: 8.584518 | lr: 4.2000e-04 | norm: 1.6879 | dt: 109.05ms | tokens/sec: 150236.39
Step    7 | loss: 8.266123 | lr: 4.8000e-04 | norm: 2.0413 | dt: 109.80ms | tokens/sec: 149212.73
Step    8 | loss: 7.880139 | lr: 5.4000e-04 | norm: 1.5634 | dt: 109.75ms | tokens/sec: 149281.78
Step    9 | loss: 7.526327 | lr: 6.0000e-04 | norm: 1.4583 | dt: 109.70ms | tokens/sec: 149358.67
```

### Gradient Accumulation

<img src="/images/reproduce-gpt/gpt3-hyperparameters.png">


The batch size of 125M model is 0.5M tokens. All above training iterations were performed with 16(Batch size) x 1024(Tokens) = 16,384 tokens. To increase the number of tokens to 0.5M, we've to increase the batch size by 0.5M/1024 = 488. B,T of (488, 1024) is required to load 0.5M tokens. This is not feasible due to the limited Memory(GPU) or resources available. To overcome this and achieve a batch size of 0.5M we can use Gradient Accumulation.

***What's gradient accumulation?

***With the above example it's to calculate gradients until 0.5M tokens are reached and then perform the weights upgrade with optimizer.

***How can this be implemented via code?***
* Find a nice number that is in powers of 2 nearest to 0.5M and divisible by B * T. 5,24,288
* Now loop for steps of 5,24,288/B * T = 32steps.
* Gradients will just be added up at every step
* CrossEntropyLoss uses a mean reduction. Normalize the loss accumulated for 32steps(for loop of 32 steps of 16 * 1024 Tokens) by diving accumulated loss by steps(32).

#### Loss normalization intution

Let's calculate loss for four inputs over a single forward pass and in gradient accumulation manner and compare them.

```Python
# simple neural net
import torch
net = torch.nn.Sequential(
	torch.nn.Linear(16, 32),
	torch.nn.GELU(),
	torch.nn.Linear(32, 1)
)
torch.random.manual_seed(42)
x = torch.rand(4, 16)
y = torch.randn(4, 1)
```

```Python
# Single Forward pass loss
# Loss Objective
# loss = 1/4 * [
# (y_hat[0] - y[0])**2 + (y_hat[1] - y[1])**2 + (y_hat[2] - y[2])**2 + (y_hat[3] - y[3])**2
#]
net.zero_grad()
y_hat = net(x)
loss = torch.nn.functional.mse_loss(y_hat, y)
loss.backward()
net[0].weight.grad.view(-1)[:10]
# tensor([ 4.4395e-03, 5.4900e-06, -3.4201e-04, 1.8907e-03, 2.0486e-03, 9.6243e-03, 6.5651e-03, 1.3026e-03, 8.0891e-03, 4.3802e-03])
```

```Python
# Gradient accumulated loss
# Loss Objective
# acumualtion in gradient --> SUM in loss
# L0 = (y_hat[0] - y[0])**2
# L1 = (y_hat[1] - y[1])**2
# L2 = (y_hat[2] - y[2])**2
# L3 = (y_hat[3] - y[3])**2
# loss = (L0 + L1 + L2 + L3)
net.zero_grad()
for input_ in range(len(x)):
	y_hat = net(x[input_])
	loss = torch.nn.functional.mse_loss(y_hat, y[input_])
	loss.backward()
net[0].weight.grad.view(-1)[:10]
# tensor([ 1.7758e-02, 2.1960e-05, -1.3680e-03, 7.5628e-03, 8.1944e-03, 3.8497e-02, 2.6260e-02, 5.2104e-03, 3.2356e-02, 1.7521e-02])
```

We can see the weights are not the same, they are different. How can this be fixed?
```Python
# To achieve the same loss objective as before gradient accumulation we perform the mean over loss accumulated manually
# L0 = (y_hat[0] - y[0])**2
# L1 = (y_hat[1] - y[1])**2
# L2 = (y_hat[2] - y[2])**2
# L3 = (y_hat[3] - y[3])**2
# loss = 1/4 * (L0 + L1 + L2 + L3)
net.zero_grad()
for input_ in range(len(x)):
	y_hat = net(x[input_])
	loss = torch.nn.functional.mse_loss(y_hat, y[input_])
	loss = loss/4
	loss.backward()
net[0].weight.grad.view(-1)[:10]
# tensor([ 4.4395e-03, 5.4900e-06, -3.4201e-04, 1.8907e-03, 2.0486e-03, 9.6243e-03, 6.5651e-03, 1.3026e-03, 8.0891e-03, 4.3802e-03])
```

Now the losses are same in single step and gradient accumulation step.

We'll add this to the training loop via for loop.

```Python
B = 16
T = 1024
max_steps = 50
max_lr = 6e-4 # From GPT2 paper
min_lr = max_lr * 0.1
warmup_steps = 10
total_batch_size = 524288 # 2**19 and divisible by B*T
assert total_batch_size % (B*T) == 0, f"Total batch size {total_batch_size} must be divisible by B*T {B*T}"
grad_accum_steps = total_batch_size // (B*T)
print(f"Total batch size: {total_batch_size}")
print(f"Grad accum steps:=> {total_batch_size} // {B} * {T} = {grad_accum_steps}")

torch.set_float32_matmul_precision("high")

# create dataloader
data_loader = DataLoaderLite(input_file="input.txt", B=B, T=T)
  
# Initialize model
losses = []
model = GPT2(GPTConfig).to(device)
model = torch.compile(model)
# Initialize optimizer

optim = model.configure_optimizers(
	weight_decay=0.1,
	learning_rate=6e-4,
	device=device,
)

# Initialize training loop
for step in range(max_steps):
	loss_accum = 0
	t0 = time.time()
	x, y = data_loader.next_batch()
	x, y = x.to(device), y.to(device)
	# Optimizer zero grad
	optim.zero_grad(set_to_none=True)
	# Forward pass
	# Gradient accumulation
	for micro_step in range(grad_accum_steps):
		with torch.autocast(device_type=device, dtype=torch.bfloat16):
			logits, loss = model(x, y)
		loss = loss / grad_accum_steps
		# detach to remove this step from gradient calculation
		loss_accum += loss.detach()
		# Backward pass
		loss.backward()
	# Clip gradient norm
	norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
	# determine and set learning rate for this iteration
	lr = get_lr(step)
	# There's a notion in PyTorch where multiple param_groups might exist,
	# hence we're looping through them and setting the lr in below fashion
	for param_group in optim.param_groups:
		param_group['lr'] = lr
	# Update parameters
	optim.step()
	
	# Wait until all instruction sent from cpu to gpu are completed
	if device == "cuda":
		torch.cuda.synchronize()
	t1 = time.time()
	dt = (t1 - t0)*1000 # ms
	tokens_processed = B * T * grad_accum_steps
	tokens_per_sec = tokens_processed / (t1-t0)
	print(f"Step {step:4d} | loss: {loss_accum.item():.6f} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt:.2f}ms | tokens/sec: {tokens_per_sec:.2f}")
	losses.append(loss.item())
print(losses)
```

```logs
Total batch size: 524288
Grad accum steps:=> 524288 // 16 * 1024 = 32
Using device: cuda
Running all data overfit training loop
Number of tokens: 338025
Total number of batches per epoch: 20
Weights intializations started
Weights intiailization complete
Number of decayed parameter tensors: 50, (124354560)
Number of non-decayed parameter tensors: 98, (121344)
Using fused AdamW: True
Step    0 | loss: 10.956725 | lr: 6.0000e-05 | norm: 2.0819 | dt: 35187.63ms | tokens/sec: 14899.78
Step    1 | loss: 10.395347 | lr: 1.2000e-04 | norm: 2.5778 | dt: 3408.19ms | tokens/sec: 153831.58
Step    2 | loss: 9.628510 | lr: 1.8000e-04 | norm: 2.6722 | dt: 3415.39ms | tokens/sec: 153507.47
Step    3 | loss: 9.332164 | lr: 2.4000e-04 | norm: 4.9852 | dt: 3410.96ms | tokens/sec: 153706.75
Step    4 | loss: 8.826386 | lr: 3.0000e-04 | norm: 2.5916 | dt: 3413.74ms | tokens/sec: 153581.86
Step    5 | loss: 8.796247 | lr: 3.6000e-04 | norm: 2.7638 | dt: 3413.34ms | tokens/sec: 153599.91
Step    6 | loss: 8.531341 | lr: 4.2000e-04 | norm: 1.9492 | dt: 3413.70ms | tokens/sec: 153583.67
Step    7 | loss: 8.222630 | lr: 4.8000e-04 | norm: 1.8840 | dt: 3412.05ms | tokens/sec: 153657.79
Step    8 | loss: 7.844775 | lr: 5.4000e-04 | norm: 1.9959 | dt: 3412.03ms | tokens/sec: 153658.63
Step    9 | loss: 7.483783 | lr: 6.0000e-04 | norm: 1.4544 | dt: 3415.39ms | tokens/sec: 153507.57
```

Time has increased 32x times from previous dt.

## Distributed Data Parallel

Until now all the code were executed on a single GPU. Next let's use N GPUs to perform training.

[Our model fits in a single GPU but to scale up training using multiple GPUs](https://pytorch.org/tutorials/beginner/dist_overview.html), we'll use [torch.nn.parallel.DistributedDataParallel](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html).

*processes and gpus are use interchangeably.*

### How DDP Works?

* [Basics](https://pytorch.org/docs/stable/distributed.html#basics)
* Launches N processes for N GPU's.
* Each of these process run the same training loop as before from train.py but with different chunks of data.
* Once Backpropgation is completed, gradients across N GPUs are averaged before optimization step is performed on respective processes.
* Each process has it's own python interpreter avoiding GIL - thrashing comes from driving several execution threads from a single process.

### Implementation in code

Initialization of DDP:
*  Requires initializtion `torch.distributed` by calling `torch.distributed.init_process_group`.
	* `init_process_group('backend')` specify the backend to be used for distributed training. Multiple backends are available - GLOO, NCCL, UCC, MPI.
	* We'll use NCCL(NVIDIA Collective Communications Library)
* Training will be executed using `torchrun` instead of `python train.py`.
* `torchrun` makes sure of below things:
	* It runs N process in parallel.
	* Creates `RANK, LOCAL_RANK, WORLD_SIZE` environment variables to identify the processes.
		* RANK - rank of current process.
		* WORLD_SIZE - Number of process participating in the job or Number of GPUs
		* LOCAL_RANK - Local rank within a node(useful for multi node setup). 
	* Each process differs only with RANK.
	* master process(RANK=0).
		* performs logging, print, checkpoints.


* Now we've to update `grad_accum_steps` from `B * T` to `B * T * ddp_world_size`, where `ddp_world_size` is num GPUs.
* grad_accum_steps = 5,24,288/B * T * ddp_world_size = 4steps. where B = 16, T = 1024.\
* To avoid multiple prints, print only on master process.

### DataLoader Update

We've to update the DataLoader to give different chunk of data for each GPU.

`self.current_postition`  will be updated from `B * T` to `B * T * ddp_rank`. With ddp_rank range from 0 to N, we'll be selecting a new chunk for each GPU and start from beginning when we run out of tokens.

### Model Update

We've to wrap the model with `torch.distributed` and pass the `ddp_local_rank`.
```Python
model = torch.distributed(model. device_ids=[ddp_local_rank])
raw_model = model.module 
```

With DDP:
* Forward pass remains the same across processes.
* Average gradients across processes for all model parameters and synchronize the average to all processes
* During backward pass, DDP sends communication between processes(probably from master process) as soon as the gradients are calculated for parameters creating an overlap before average of gradients and synchronize the averaged gradients across processes.

### Training Loop Update

* To avoid average and synchronize gradients for every gradient accumulation step. To avoid this we can call `loss.backward()` just in the last micro_step of gradient accumulation.
	* To do this we can use [no_sync context manager](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel.no_sync). In the [source code](https://pytorch.org/docs/stable/_modules/torch/nn/parallel/distributed.html#DistributedDataParallel.no_sync) this set's self.require_backward_grad_sync to False for no sync of gradients.
		```Python
		#old
		loss.backward()
		# Update
		if ddp:
			# This makes sure grads are not synchronized until last microstep
			model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
		loss.backward()
		```
* Update token size with ddp_world_size. This will be number of tokens processed across N processes.
* Perform an average of loss across processes before printing it out.
* Gradients are averaged, then we've to average losses.
* Shutdown of processes to clean up resources, use `destroy_process_group()`. At the end of script.

*This run is on 2 A100SXM GPUs single node compute*

```logs
Is running on DDP: True
Dist initialization: True
Dist initialization: True
Total batch size: 524288
Grad accum steps:=> 524288 // 16 * 1024 = 16
Number of tokens processing in parallel: 32768
Number of tokens: 338025
Total number of batches per epoch: 20
Step 0 | loss: 10.980519 | lr: 6.0000e-05 | norm: 2.2486 | dt: 31183.66ms | tokens/sec: 16812.91
Step 1 | loss: 10.382103 | lr: 1.2000e-04 | norm: 2.5440 | dt: 1418.98ms | tokens/sec: 369481.78
Step 2 | loss: 9.519885 | lr: 1.8000e-04 | norm: 2.4928 | dt: 1421.23ms | tokens/sec: 368896.48
Step 3 | loss: 9.134990 | lr: 2.4000e-04 | norm: 5.4789 | dt: 1419.17ms | tokens/sec: 369432.55
Step 4 | loss: 8.850931 | lr: 3.0000e-04 | norm: 2.8440 | dt: 1420.94ms | tokens/sec: 368972.80
Step 5 | loss: 8.603592 | lr: 3.6000e-04 | norm: 2.9302 | dt: 1421.49ms | tokens/sec: 368829.84
Step 6 | loss: 8.266810 | lr: 4.2000e-04 | norm: 2.0269 | dt: 1423.09ms | tokens/sec: 368415.83
Step 7 | loss: 7.935740 | lr: 4.8000e-04 | norm: 2.5958 | dt: 1421.07ms | tokens/sec: 368938.13
Step 8 | loss: 7.568612 | lr: 5.4000e-04 | norm: 1.9012 | dt: 1423.92ms | tokens/sec: 368201.10
Step 9 | loss: 7.199800 | lr: 6.0000e-04 | norm: 1.6810 | dt: 1422.85ms | tokens/sec: 368478.06
```

Tokens per sec is doubled and dt is reduced by a factor of 2, due to increase in number of compute to 2.

Do not indent destroy_process_group inside training loop lol. I was running in circles for sometime why process group was destroyed after first step.
### DDP Materials
1. [Basics](https://pytorch.org/docs/stable/distributed.html#basics)
2. [DDP internal working](https://pytorch.org/docs/stable/notes/ddp.html)
3. [ZeroRedundancyOptimizer](https://pytorch.org/tutorials/recipes/zero_redundancy_optimizer.html)

## Dataset Update

### Dataset Creation

Alright now we're done with speeding up and our training setup is no longer suited for tiny shakespeare dataset. Next let's take a look at the datasets history in GPT series models and open source datasets available for pretraining objective.

* GPT2 - Scraped all reddit outbound links with at least 3 karma(heuristic indicator for whether the article was useful or not). The resulting dataset WebText contains the text subset of these 45 million links. [openwebtext is the opensource alternative](https://openwebtext2.readthedocs.io/en/latest/background/) as the dataset and source code was not made public.
* GPT3 - Filtered version of CommonCrawl and fuzzy deduplication at the document level across datasets, plus high-quality corpora to the training mix to augment CommonCrawl to increase its diversity. *GPT3 data mixture below*:
	<img src="/images/reproduce-gpt/gpt3-data-mixture.png">


*Open source datasets:*
1. [OpenWebText2](https://openwebtext2.readthedocs.io/en/latest/background/)
2. [RedPajama](https://github.com/togethercomputer/RedPajama-Data/tree/mainslim)
3. [SlimPajama](https://cerebras.ai/blog/slimpajama-a-627b-token-cleaned-and-deduplicated-version-of-redpajama). Clean deduplicated scaled version of RedPajama
4. [finweb](https://huggingface.co/datasets/HuggingFaceFW/fineweb). To read upon data mixture used to create this dataset refer this [blogpost by huggingface](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1)
5. [fineweb edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)

We're gonna use the fineweb 10B Tokens dataset for our use case. Preprocess, tokenize, shard the dataset.

```gist
https://gist.github.com/JpChii/e8dccd125a950f983da57895c3006e29
```

We're gonna use the fineweb script above to load sample 10T data from huggingface and convert them to 100 shards with each shard holding 100M tokens and size of 200MB. This dataset is uploaded to huggingface datasets to avoid token processing on every iteration. [Dataset here](https://huggingface.co/datasets/JpChi/finewebedu10BT-tokenized-gpt2)

### DataLoader Update

* Added `load_tokens()` function to load tokens from shards
* `reset()` - function to set initial position with respect to rank and shards
* updated `next_batch()` to load tokens from shard and update to load tokens from next shard when one shard is finished.

### Training Loop Update

* Update max_steps to 19072 => 10e9(10 billion tokens) / 524288(tokens per step)
* warmup_steps until 375M tokens form gpt3 paper. 375M/524288 => 715 steps

## Text Generation

We're gonna sample from the model for every N steps and generate sequences with a fixed prompt to see the model improvement.

```Python
        model.eval()
        num_return_sequences = 4
        max_length = 32 # End text generation at sequence length of 32
        # Encode tokens
        tokens = val_loader.encoder.encode("Hello, I'm a a language model")
        # Create reperating sequence of 4, 1
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)

        # Set sample seed unique for generation alone outside of training loop
        rng = torch.Generator(device=device)
        rng.manual_seed(42 + ddp_rank)

        while xgen.size(1) < max_length:
            # Forward pass
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(xgen)
                
                # Take logits of last token in the batch
                last_token_logits = logits[:, -1, :]
                # Convert to probs
                probs = torch.softmax(last_token_logits, dim=-1)
                # top_k, 50 by default in huggingface pipeline
                top_k_probs, top_k_indices = torch.topk(probs, k=50, dim=-1)

                # Sample
                next_token_ix = torch.multinomial(probs, num_samples=1, generator=rng) # (B, 1)

                # Gather the corresponding indices
                xcol = torch.gather(top_k_indices, -1, next_token_ix) # (B, 1)

                # Append
                xgen = torch.cat((xgen, xcol), dim=-1)

        # print the generated text
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = val_loader.encoder.decode(tokens)
            print(f"rank {ddp_rank} sample {i}: {decoded}")
```

* Feed a prompt that's the first token
* convert it to token ids and pass it to the model
* Slice and obtain logits for last token
* Calculate probability along embedding dimension
* get top_k_probs and top_k_indices from probabilities
* select one sample from top_k probs - token index
* Use the index to fetch token id
* append it to the existing tokens tensor
* repeat it until max length is reached
* finally print the sequence.

## Validation loss

Evaluate validation loss every n steps

```Python
	    model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                    loss = loss / val_loss_steps
                    val_loss_accum += loss.detach()
        
        if ddp:
            dist.all_reduct(val_loss_accum, op=dist.ReduceOp.AVG)

        if master_process:
            print(f"Valiation loss: {val_loss_accum.item():.4f}")

```

## HellaSwag Evaluation

 [Hellaswag](https://arxiv.org/pdf/1905.07830) evaluation is used to determine text generation capability of the model apart from validation loss.

* The dataset has a context and then four possible generations to complete the context.
* Out of the four options only one option is correct and others are wrong and easily identifiable by humans but difficult for Language Models(at the time of this paper release). The wrong answers were created with adversarial generation(created by LLM and).

### Render sample

* Accepts input sample that contains label(correct option from four generations), context, four options
* Loop through four options and create four options of context + ending tokens
* Mask - set mask to 0 for context tokens and 1 for ending tokens(generations). 
* This mask is to make sure loss is calculated for generated tokens and not the context.

### Evaluation

* For eval we'll use the prediction from huggingface GPT2 model our target as truth label.
* We'll perform below steps to get predictions from trained model and HF model:
	* Get logits
	* Slice and exclude last logit as it's not prediciotn
	* In tokens ignore first token as it's prompt
	* Flatten logits along token dimension
	* Flatten tokens along token dimension
	* Calculate cross entropy loss
	* unflatten losses to original shape
	* average losses where mask == 1
	* get prediction norm
* Then compare both preds and calculate stats

```Python
@torch.no_grad()
def evaluate(model_type, device):

    torch.set_float32_matmul_precision("high") # use tf32
    model = GPT2LMHeadModel.from_pretrained(model_type)
    model.to(device)

    # Variables to tract metrics
    num_correct_norm = 0 # Normalized corred preds
    num_correct = 0 # Unnormalized corred preds
    num_total = 0 # total number of predictions

    for example in iterate_examples("val"):

        data, tokens, mask, label = render_example(example)
        # Move to device
        tokens = tokens.to(device)
        mask = mask.to(device)

        # get logits
        logits = model(tokens).logits

        # Evaluate the autoregressive loss at all positions
        shift_logits = (logits[..., :-1, :]).contiguous() # Exclude last logit
        shift_tokens = (tokens[..., 1:]).contiguous() # Exclude first token as it's prompt
        flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1)) # Flatten logits along token dimension
        flat_shift_tokens = shift_tokens.view(-1) # Flatten tokens along token dimension

        # # Print original shapes
        # print(f"Logits: {logits.shape}")
        # print(f"Tokens: {tokens.shape}")

        # # print unflatten shift shapes
        # print(f"Shift logits: {shift_logits.shape}")
        # print(f"Shift tokens shape: {shift_tokens.shape}")

        # # Print shapes
        # print(f"flat_shift_tokens: {flat_shift_tokens.shape}")
        # print(f"flat_shift_logits: {flat_shift_logits.shape}")


        # Calculate loss
        shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction="none")
        shift_losses = shift_losses.view(shift_tokens.size(0), -1) # Unflatten

        # Average losses where mask == 1 in each row
        shift_mask = (mask[..., 1:]).contiguous()
        masked_shift_losses = shift_losses * shift_mask # Set losses to 0 where mask == 0

        # Sum and dive by total number of 1s in mask
        sum_loss = masked_shift_losses.sum()
        avg_loss = sum_loss / shift_mask.sum(dim=1)

        # Sample with lowest loss is most likley completion by model
        pred = sum_loss.argmin().item()
        pred_norm = avg_loss.argmin().item()

        # accumulate stats
        num_total += 1
        num_correct_norm += int(pred_norm == label)
        num_correct += int(pred == label)
        print(f"{num_total} acc_norm: {num_correct_norm}/{num_total}={num_correct_norm/num_total:.4f}")

        break

    # debug: pretty print a few examples, and the losses in each case
    if num_total < 10:
        print("---")
        print(f"Context:\n {example['ctx']}")
        print(f"Endings:")
        for i, end in enumerate(example["endings"]):
            print(f"{i} (loss: {avg_loss[i].item():.4f}) {end}")
        print(f"predicted: {pred_norm}, actual: {label}")
```

[full hellaswag code](https://github.com/JpChii/gpt2-reproduced/blob/main/hellaswag.py)

### Training loop update

* Add a modified version of evaluate in training loop.
* get_most_likely_row is from evaluate.

```Python
        num_correct_norm = 0
        num_total = 0
        for i, example in enumerate(iterate_examples(split="val")):

            # DDP setting, unqiue samples for each process
            if i % ddp_world_size != ddp_rank:
                continue

            # Render example to get the most proabale row
            _, tokens, mask, label = render_example(example)

            tokens, logits = tokens.to(device), logits.to(device)
            
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.float16):
                    logits, loss = model(tokens)
                pred_norm = get_most_likey_row(tokens, mask, logits)

            num_total += 1
            num_correct_norm += (label == pred_norm)

        # Reduce stats across all processes
        if ddp:
            num_total = torch.tensor(num_total, dtype=torch.long, device=device)
            num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
            dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
            num_total = num_total.item()
            num_correct_norm = num_correct_norm.item()
        acc_norm = num_correct_norm / num_total
        if master_process:
            print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")

```
