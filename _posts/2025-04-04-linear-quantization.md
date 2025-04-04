# Linear Quantization

[Source](https://www.deeplearning.ai/short-courses/quantization-in-depth/)

Linear Quantization. This notebook covers:
1. Linear Quantization theory
2. Quantizer
3. Quantization challenges(weights packing, llm quantization)

## Quantization Theory

For basic introduction to quantization [check this out](https://github.com/JpChii/nlp-with-hugging-face/blob/main/notebooks/8-Making-transformers-efficient-in-production.ipynb).

1. Assymetric Quantization

$f = \left(\frac{{f_{\text{max}} - f_{\text{min}}}}{{q_{\text{max}} - q_{\text{min}}}}\right) (q - Z) = S(q - Z)$ --> DeQuantization formula.

$f$ - tensor/channel/matrix to be quantized.

$S$ - scale factor, $Z$ - Zero Point, $q$ --> quantized value

$q = \frac{f}{S} + z$ - Quantize value

$q = int(round(q))$ --> Rounded quantize value

2. Symmetric Quantization

This is the quantization in Symmetric Quantization. $(-f_{max}, f_{max})$ --> $(-q_{max}, q_{max})$.

We don't need zero point in this case, because zero point is symmetric between original values and quantized values.

$S = \frac{f_{max}}{q_{max}}$

$q = int(round(\frac{r}{s}))$


```python
!pwd
```

    /Users/j.chinnarajii



```python
from dotenv import load_dotenv
load_dotenv("Documents/projects/env/env.local")
```




    True



## Asymmetric Quantizer


```python
import torch

def linear_q_with_scale_and_zero_point(
    tensor, scale, zero_point, dtype=torch.int8
):
    scaled_and_shifted_tensor = (tensor / scale) + zero_point
    # Round to the nearest integer
    rounded_tensor = torch.round(scaled_and_shifted_tensor)
    # clamp to the range of the dtype
    q_min = torch.iinfo(dtype).min
    q_max = torch.iinfo(dtype).max
    clamped_tensor = torch.clamp(rounded_tensor, q_min, q_max).to(dtype)
    # Cast to the dtype
    return clamped_tensor
```


```python
# Define a test tensor of shape (3,3) of FP32, and calculate scale, zeropoint to compare quantizatoin function above
test_tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
scale = 10
zero_point = -90

quantized_tensor = linear_q_with_scale_and_zero_point(test_tensor, scale, zero_point)
```


```python
def linear_dequantization(quantized_tensor, scale, zero_point):
    return scale * (quantized_tensor.float() - zero_point)
```


```python
dequantized_tensor = linear_dequantization(quantized_tensor, scale, zero_point)
dequantized_tensor
```




    tensor([[ 0.,  0.,  0.],
            [ 0.,  0., 10.],
            [10., 10., 10.]])



Quantization Error is the difference between Dequantized tensor and Original tensor. They should be as close as possible.


```python
# Quantization error
(dequantized_tensor - test_tensor).square().mean()
```




    tensor(9.4444)



The difference between quantization and dequantization is large. This is due to the random scale. Let's find scale and zero point next.

## Find Scale and Zero Point

$S = \frac{f_{max}-f_{min}}{q_{max}-q_{min}}$

$Z = {q_{min} - \frac{f_{min}}{S}}$




```python
def find_scale_and_zero_point(tensor, dtype=torch.int8):
    q_min = torch.iinfo(dtype).min
    q_max = torch.iinfo(dtype).max
    scale = (tensor.max() - tensor.min()) / (q_max - q_min)
    zero_point = (q_min - tensor.min() / scale).item()
    zero_point = int(round(zero_point))
    return scale, zero_point
```


```python
scale, zero_point = find_scale_and_zero_point(tensor=test_tensor)
scale, zero_point
```




    (tensor(0.0314), -160)



Edge Case for zero point.
If zero point is less than q_min use q_min if it's greater than q_max use q_max.


```python
def find_scale_and_zero_point_edge(tensor, dtype=torch.int8):
  q_min = torch.iinfo(dtype).min
  q_max = torch.iinfo(dtype).max
  scale = (tensor.max() - tensor.min()) / (q_max - q_min)
  zero_point = (q_min - tensor.min() / scale).item()
  print(f"Zero point: {zero_point}")
  if zero_point < q_min:
    zero_point = q_min
  elif zero_point > q_max:
    zero_point = q_max
  zero_point = int(round(zero_point))
  return scale, zero_point
```


```python
scale, zero_point = find_scale_and_zero_point_edge(tensor=test_tensor)
scale, zero_point
```

    Zero point: -159.875





    (tensor(0.0314), -128)




```python
# Quantize with scale and zero point
quantized_tensor = linear_q_with_scale_and_zero_point(test_tensor, scale, zero_point)
quantized_tensor
```




    tensor([[-96, -64, -32],
            [ -1,  31,  63],
            [ 95, 127, 127]], dtype=torch.int8)




```python
# Find the difference
(dequantized_tensor - test_tensor).square().mean()
```




    tensor(9.4444)




```python
# Put everything into a function
def linear_q(tensor, dtype=torch.int8):
    scale, zero_point = find_scale_and_zero_point(tensor)
    return linear_q_with_scale_and_zero_point(tensor, scale, zero_point, dtype)
```


```python
# Quantize and evaluate
def linear_q_with_eval(tensor, dtype=torch.int8):
    scale, zero_point = find_scale_and_zero_point(tensor)
    quantized_tensor = linear_q_with_scale_and_zero_point(tensor, scale, zero_point, dtype)
    dequantized_tensor = linear_dequantization(quantized_tensor, scale, zero_point)
    quantization_error = (dequantized_tensor - test_tensor).square().mean()
    return {
        "Quantized Tensor": quantized_tensor,
        "Dequantized Tensor": dequantized_tensor,
        "Quantization Error": quantization_error,
    }
```


```python
print(linear_q_with_eval(tensor=test_tensor))
```

    {'Quantized Tensor': tensor([[-128,  -96,  -64],
            [ -33,   -1,   31],
            [  63,   95,  127]], dtype=torch.int8), 'Dequantized Tensor': tensor([[1.0039, 2.0078, 3.0118],
            [3.9843, 4.9882, 5.9922],
            [6.9961, 8.0000, 9.0039]]), 'Quantization Error': tensor(7.6893e-05)}


## Symmetric Quantizer


```python
test_tensor = torch.randn((4, 4))
```


```python
def get_q_scale_symmetric(tensor, dtype=torch.int8):
  q_max = torch.iinfo(dtype).max
  return tensor.abs().max().item() / q_max
```


```python
scale = get_q_scale_symmetric(tensor=test_tensor)
scale
```




    0.015789092056394564




```python
def linear_q_symmetric(tensor, dtype=torch.int8):
    scale = get_q_scale_symmetric(tensor)
    zero_point = 0
    quantized_tensor = linear_q_with_scale_and_zero_point(tensor, scale, zero_point, dtype)
    return quantized_tensor, scale
```


```python
# assymetric quantization
quantized_tensor, scale = linear_q_symmetric(tensor=test_tensor)
quantized_tensor, scale
```




    (tensor([[  40, -122,  -15,    3],
             [ -22,  -43,  115,  -23],
             [  10,  127, -107,   56],
             [   1,   -9,   35,    6]], dtype=torch.int8),
     0.015789092056394564)



### Trade-Off

* Assymetric uses entire quantization range.
* Symmetric, float range is biased towards one side. Only half of quantization range is utilized. We can use this scheme for RELU(and other similar activations).
* Symmetric is simpler and doesn't need storage for zero-point.

## Channel Quantization

Channel Quantization is quantization along row, column or desired dimension.

1. Traverse in row or column dimension.
2. Find scale per channel.
3. Reshape scale to match test tensor to perform matrix division.
4. Return channel quantized tensor, scale.


```python
test_tensor=torch.tensor(
    [[191.6, -13.5, 728.6],
     [92.14, 295.5,  -184],
     [0,     684.6, 245.5]]
)
```


```python
# Channel quantize along row dimension
dim = 0
output_dim = test_tensor.shape[dim]
output_dim
```




    3




```python
test_tensor.shape[0]
```




    3




```python
# Scale tensor is not a single tensor anymore. This should match the dimension of quantization
scale_tensor = torch.zeros(output_dim)
```


```python
scale_tensor.shape, scale_tensor
```




    (torch.Size([3]), tensor([0., 0., 0.]))




```python
# Calculate scale
for idx in range(output_dim):
  # Select corresponding row
  channel_tensor = test_tensor[idx, :]
  scale_tensor[idx] = get_q_scale_symmetric(tensor=channel_tensor)
```


```python
scale_tensor
```




    tensor([5.7370, 2.3268, 5.3906])




```python
scale_tensor.shape, test_tensor.shape
```




    (torch.Size([3]), torch.Size([3, 3]))




```python
# With these shapes each row in test_tensor will be divded by three elements in scale tensor
# But scale tensor was calculated as one tensor per row
# Each tensor needs to be scaled by respective scale tensor. To do this, we've to change the shape of scale tensor.
# We'll use torch.view() to avoid additional memory overhead.
scale_shape = [1] * test_tensor.dim() # Match target tensor shape
scale_shape[dim] = -1
scale_shape
```




    [-1, 1]




```python
# Use view to reshape the scale tensor
scale = scale_tensor.view(scale_shape)
scale
```




    tensor([[5.7370],
            [2.3268],
            [5.3906]])






```python
# Quantization
quantized_tensor = test_tensor / scale
quantized_tensor = torch.clamp(quantized_tensor, -128, 127).to(torch.int8)
```


```python
dequantized_tensor = scale * quantized_tensor
dequantized_tensor
```




    tensor([[ 189.3213,  -11.4740,  728.6000],
            [  90.7441,  293.1732, -183.8150],
            [   0.0000,  684.6000,  242.5748]])




```python
(dequantized_tensor - test_tensor).abs().square().mean()
```




    tensor(2.8056)




```python
# Now we've all we need to perfrom channel quantization, let's put this together into a function
def linear_q_symmetric_per_channel(input_tensor, dim=0, dtype=torch.int8):

  # Select quantization dim
  output_dim = input_tensor.shape[dim]

  # Get Scale per channel and reshape for quantization
  scale_tensor = torch.zeros(output_dim)
  for idx in range(output_dim):
    channel_tensor = input_tensor.select(dim, idx)
    scale_tensor[idx] = get_q_scale_symmetric(tensor=channel_tensor)

  # Reshape
  scale_shape = [1] * input_tensor.dim()
  scale_shape[dim] = -1
  scale = scale_tensor.view(scale_shape)

  # Quantize
  quantized_tensor = linear_q_with_scale_and_zero_point(input_tensor, scale, 0, dtype)

  return quantized_tensor, scale
```


```python
# Test channel quantizer along row
quantized_tensor_0, scale_0 = linear_q_symmetric_per_channel(input_tensor=test_tensor, dim=0)
quantized_tensor_0, scale_0
```




    (tensor([[ 33,  -2, 127],
             [ 40, 127, -79],
             [  0, 127,  46]], dtype=torch.int8),
     tensor([[5.7370],
             [2.3268],
             [5.3906]]))




```python
# Test quantization along column
quantized_tensor_1, scale_1 = linear_q_symmetric_per_channel(input_tensor=test_tensor, dim=1)
quantized_tensor_1, scale_1
```




    (tensor([[127,  -3, 127],
             [ 61,  55, -32],
             [  0, 127,  43]], dtype=torch.int8),
     tensor([[1.5087, 5.3906, 5.7370]]))




```python
# Dequantized tensor
dequantized_tensor_0 = scale_0 * quantized_tensor_0
dequantized_tensor_0
```




    tensor([[ 189.3213,  -11.4740,  728.6000],
            [  93.0709,  295.5000, -183.8150],
            [   0.0000,  684.6000,  247.9653]])




```python
# Quantization error
(dequantized_tensor_0 - test_tensor).abs().square().mean()
```




    tensor(1.8084)



## Group Quantization

Per-Group Quantization requires lot more memory. How?

Example:

* Quantize 4-bit tensor.
* Group Size: 32(Group Size is in multiples of 2)
* Scale Per Group: 16bit/32elements = 0.5bit
* Quantization requires: 4.0 + 0.5 bits --> 4.5 bits.

For simplicity, we'll implement quantization with two dimension tensor.


```python
def linear_q_symmetric_per_group(tensor, group_size, dtype=torch.int8):

  # Get tensor shape
  t_shape = tensor.shape
  # To perform quantization along rows, each row must match group_size
  # This ensures, we can reshape the tensor with group_size for quantization
  assert t_shape[1] % group_size == 0
  # Ensure tensor is two dimension
  assert tensor.dim() == 2

  # Reshape tensor
  tensor = tensor.view(-1, group_size)

  # Groups are created, use per channel to quantize each row
  quantized_tensor, scale = linear_q_symmetric_per_channel(
      input_tensor=tensor, dim=0, dtype=dtype
  )

  return quantized_tensor.view(t_shape), scale

```


```python
test_tensor = torch.rand((4,128))
```


```python
test_tensor.shape
```




    torch.Size([4, 128])




```python
test_tensor.view(-1, 32).shape
```




    torch.Size([16, 32])




```python
# Dequantization function to verify results
def linear_dequantization_per_group(quantized_tensor, scale, group_size):
  """
  Quantization was done per group size, it makes sense to dequantize per group size.
  """

  t_shape = quantized_tensor.shape
  quantized_tensor = quantized_tensor.view(-1, group_size)
  dequantized_tensor = scale * quantized_tensor
  return dequantized_tensor.view(t_shape)
```


```python
# Test Quantization and calculate the error
test_tensor = torch.rand((4,128))
group_size = 32
quantized_tensor, scale = linear_q_symmetric_per_group(tensor=test_tensor, group_size=group_size)
dequantized_tensor = linear_dequantization_per_group(quantized_tensor, scale, group_size)
(dequantized_tensor - test_tensor).abs().square().mean()
```




    tensor(4.8609e-06)



The quantization error is very very low.

## Linear Inference Quantization

In Neural Networks both weights and activations can be quantized. Depending on what is quantized storage and computation differs.

***Case1:***
Storage: W8A32 - Weights 8-bit, Activation - 32-bit
Computation: FP32, FP16(Floating point arithmentics)
Remark: Weights need to be Dequantized to perfrom Floating point computation. Because INT8 can't be used for FLOAT computations.

***Case2:***
Storage: W8A8 - Weights 8-bit, Activation - 32-bit
Computation: INT8, INT4(Integer Arithmetics)
Remark: Isn't available in all hardware.

Let's implement linear layer quantization in W8A32.


```python
def quantized_linear_WA832_without_bias(input, q_w, s_w, z_w):
  """
  Quantized Linear Layer.

  Args:
    input: Input Tensor to Linear layer
    q_w: Quantized Weights
    s_w: Scale of Weights
    z_w: Zero Point of Weights

  """
  assert input.dtype == torch.float32 # From previous activation
  assert q_w.dtype == torch.int8

  dequantized_tensor = q_w.to(torch.float32) * s_w - z_w
  output = torch.nn.functional.linear(input, dequantized_tensor)
  return output
```


```python
input = torch.rand((3,3), dtype=torch.float32)
weight = torch.rand((3,3), dtype=torch.float32)
q_w, s_w = linear_q_symmetric(tensor=weight, dtype=torch.int8)
```


```python
%%timeit
output = quantized_linear_WA832_without_bias(input, q_w, s_w, 0)
```

    4.18 μs ± 60.6 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)



```python
# Let's try this linear layer without quantization
%%timeit
output = torch.nn.functional.linear(input, weight)
```

    836 ns ± 35.8 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops each)


Output tensor are really close to each other and latency of linear quantization layer is larger than FP32 multiplication.

Quantized models will have lesser memory footprint compared to thei larger counterparts but latency decrease due to quantization overhead is a cost to bear.

Advantages:

To understand the advantages, we've to remind ourselves of model is loaded.
Model Resides on Disk -> Layer, Configurations, Weights loaded to memory(cache will be used internally) --> Weights are sent to Processor for computation -> output is sent to memory.

* For LLMs this reduces the weights by 75% leading to lower memory footprint. This is crucial for devices with limited RAM.
* Reduces the pressure on memory bandwidth. (Memory bandwidth is used to move weights from memory to processor). This indirectly improves the performance.
* Smaller weights mean more more of the model can fit into CPU/GPU cache, reducing latency caused by memory fetches.
* LLM's are not always persisted in memory. Latency associated with frequent loading of weights into memory will be reduced. This latency benefit is more prominent with LLM's due to their memory constraint.
* Deployment's easier with small model size.

## Building Quantizer to Quantize models

Let's build a quantizer to quantize any model in 8-bit quantization.
This quantizer can quantize Audio, Video, Text.(Model Agnostic)

1. Create W8A16 Linear Layer
2. Replace LinearLayers with W8A16Layer

### 1. Create W8A16 Linear Layer

Create a linear layer to sotre 8-bit weights and 16 bit activations.

***w8a16 forward pass method***
- inputs: linear layer input(16bit), weights(8bit), scales, bias(optional).
- Functionality: Dequantize weights, change dtype, perform linear operation.

***quantize method***
- quantize weights


```python
import torch
import torch.nn as nn
import torch.nn.functional as F
```


```python
# Random inputs
random_weights = torch.randint(low=-127, high=127, size=(32, 16), dtype=torch.int8) # 32 inputs of size 16
random_hs = torch.randn((1, 16), dtype=torch.float16) # 16 Neurons
scales = torch.randn((1, 32), dtype=torch.float16) # match output shape of weight matrix. linear(random_random_hsactivations, random_weights) --> (1,16) x (16, 32) --> (1, 32)
bias = torch.randn((1, 32), dtype=torch.float16)
```


```python
F.linear(random_hs, random_weights.to(random_hs.dtype)), F.linear(random_hs, random_weights.to(random_hs.dtype)).shape
```




    (tensor([[-434.7500, -311.2500,   94.2500, -211.3750, -146.8750, -166.0000,
               241.3750,   -8.4922, -517.5000, -144.3750,   20.9219, -166.3750,
                65.5000,  367.2500,  360.7500,  140.0000,  124.0625,   46.4062,
               236.2500,  202.0000,  -85.3750,   87.5000,  -14.9453, -168.3750,
               -41.1250,   51.8438,  167.0000,   85.4375,  -50.7188, -447.5000,
                40.5312, -159.0000]], dtype=torch.float16),
     torch.Size([1, 32]))




```python
# Let's put this together in a function
def w8_a16_forward(weights, inputs, scales, bias=None):
  """
  Forward pass of W8A16 Linear Layer. This function accepts weights, inputs(hidden state activations), scales and optional bias.
  """

  assert weights.dtype == torch.int8

  casted_weights = weights.to(inputs.dtype)
  # Linear operation on input and weights
  output = F.linear(inputs, casted_weights) * scales
  if bias is not None:
    output = output + bias
  return output
```


```python
# Test the function without bias
output = w8_a16_forward(weights=random_weights, inputs=random_hs, scales=scales)
output.shape, output
```




    (torch.Size([1, 32]),
     tensor([[ 367.2500,   37.6875,  -56.3750,  165.0000,   20.9844, -329.7500,
              -140.8750,    4.6562, 1384.0000, -138.6250,  -25.8594,   13.6250,
                46.2188,  465.2500,   33.7812,  182.0000,  -27.5781,   14.8984,
              -197.6250, -150.3750,   25.4844, -117.9375,    8.5781,   99.0000,
               -43.8438,  -79.8125,  128.3750,   16.1719,   35.4688,  153.3750,
                12.7969,   -7.4141]], dtype=torch.float16))




```python
# Test the function with bias
output = w8_a16_forward(weights=random_weights, inputs=random_hs, scales=scales, bias=bias)
output.shape, output
```




    (torch.Size([1, 32]),
     tensor([[ 368.2500,   38.1562,  -57.3750,  164.7500,   22.5000, -329.0000,
              -141.1250,    6.3242, 1384.0000, -139.7500,  -26.7344,   16.0781,
                46.1250,  465.5000,   35.3125,  183.1250,  -26.8906,   14.0781,
              -196.5000, -151.2500,   23.7344, -116.8125,    9.4062,   97.9375,
               -43.2812,  -79.5000,  128.3750,   16.0469,   35.5312,  154.2500,
                13.2188,   -7.7500]], dtype=torch.float16))



### Create Linear Layer Class

Leverage the above method and create a LinearLayer class with PyTorch. This class has to meet nn.Linear signature to replace the classes in any model to perform quantization.

This class holds in8 weights, scale, bias.

1. Implement Forward pass
2. Implement Quantize method

Quantize method will convert the weights to Int8.


```python
import inspect
inspect.signature(nn.Linear)
```




    <Signature (in_features: int, out_features: int, bias: bool = True, device=None, dtype=None) -> None>



Weights are held in nn.Parameter to perform backprogation with gradients. But torch doesn't support gradients with int8.


```python
class W8A16LinearLayer(nn.Linear):
  def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=torch.float32) -> None:
     super().__init__(in_features, out_features, bias, device, dtype)

     # Weights
     self.int8_weights = nn.Parameter(torch.Tensor([0, 1]).to(dtype=torch.int8))
```


```python
# This'll fail
# W8A16LinearLayer(
#     in_features=10,
#     out_features=10,
#     bias=True,
#     device=None,
#     dtype=torch.float32
# )
```

We've to store the weights in torch.int8 within the LinearLayer without require_grad. To do this we can use register_buffer from nn.Module. With this we can store the weights in any datatype.

***For inference required_grad is not needed***


```python
class W8A16LinearLayerV1(nn.Module):

  def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=torch.float32) -> None:
    super().__init__()
    # Weights
    self.register_buffer(
        "int8_weights",
        torch.randint(low=-127, high=127, size=(out_features, in_features), dtype=torch.int8)
    )
    # Scales
    self.register_buffer(
        "scales",
        torch.randn(out_features, dtype=dtype)
    )
    if bias:
      self.register_buffer("bias", torch.randn(1, out_features, dtype=dtype))
    else:
      self.bias = None

    # Forward pass
  def forward(self, inputs):
      return w8_a16_forward(
          weights=self.int8_weights,
          inputs=inputs,
          scales=self.scales,
          bias=self.bias
      )

  def quantize(self, weights):
      w_fp32 = weights.clone().to(torch.float32)
      # Create scales
      scales = w_fp32.abs().max(dim=-1).values / 127 # Per channel scales
      scales = scales.to(weights.dtype)

      int8_weights = torch.round(w_fp32 / scales.unsqueeze(1)).to(torch.int8) # Unsqueeze is to resize scales as row vector

      self.int8_weights = int8_weights
      self.scales = scales

      return int8_weights, scales
```


```python
# Test LinearLayer
layer = W8A16LinearLayerV1(
    in_features=16,
    out_features=32,
)
```


```python
# Pass dummy hidden states from previous layer.
# 1 datapoint of sequence length 8, out features of 16. 16 matches in features of linear layer for matrix multiplication.
hidden_states = torch.randn((1, 8, 16), dtype=torch.float16)
module = W8A16LinearLayerV1(16, 32)
output = module(hidden_states)
```


```python
output.shape, output.dtype
```




    (torch.Size([1, 8, 32]), torch.float32)



Forward pass takes care of type conversion, Linear Matrix Multiplication, Multiply sclaes, add bias(optional) and returns output.


```python
torch.unsqueeze(torch.randn(16, 32).abs().max(dim=-1).values, 1).shape
```




    torch.Size([16, 1])




```python
# Let's try out Linear Layer Class
module = W8A16LinearLayerV1(4, 8)
# Random intialized weights
print(f"Linear Layer Weights: {module.int8_weights}")
```

    Linear Layer Weights: tensor([[ 122,   78,  -23,  -95],
            [  24,  -76,   -8,   24],
            [   2,   83,  -54,   17],
            [ -88,    3,  -85,  -26],
            [  58, -118,   18,   -1],
            [  10,  -65, -110,  -27],
            [ -20,   65,  -62,  -50],
            [ 119,  126,   74,   56]], dtype=torch.int8)



```python
# Create some weights and quantize them
weights = torch.randn(4, 8)
int8_weights, scales = module.quantize(weights)
```


```python
module.quantize(weights)
```




    (tensor([[  78,   33,    1,   89, -127,  -55,  112,  -58],
             [  13,  -91,  -51,  -94,  -92,    1, -127,   -5],
             [  55,  -30, -115,   41,  127,  125,  -76,    1],
             [-127,    0,    6,  -80,  -15,   30,   56,  -33]], dtype=torch.int8),
     tensor([0.0118, 0.0119, 0.0125, 0.0139]))




```python
module.int8_weights.shape
```




    torch.Size([4, 8])




```python
module.scales.shape
```




    torch.Size([4])




```python
module.int8_weights.dtype, module.scales.dtype
```




    (torch.int8, torch.float32)



Let's check dequantized weights against original weights.


```python
# Dequantized weights
dequantized_weights = int8_weights.to(torch.float32) * scales.unsqueeze(1)
dequantized_weights
```




    tensor([[ 0.9168,  0.3879,  0.0118,  1.0461, -1.4928, -0.6465,  1.3164, -0.6817],
            [ 0.1553, -1.0868, -0.6091, -1.1226, -1.0987,  0.0119, -1.5167, -0.0597],
            [ 0.6863, -0.3743, -1.4350,  0.5116,  1.5847,  1.5597, -0.9483,  0.0125],
            [-1.7686,  0.0000,  0.0836, -1.1141, -0.2089,  0.4178,  0.7798, -0.4595]])




```python
weights
```




    tensor([[ 9.2089e-01,  3.8608e-01,  6.1918e-03,  1.0515e+00, -1.4928e+00,
             -6.4184e-01,  1.3146e+00, -6.8212e-01],
            [ 1.5292e-01, -1.0915e+00, -6.0975e-01, -1.1176e+00, -1.0999e+00,
              1.5369e-02, -1.5167e+00, -5.5801e-02],
            [ 6.8585e-01, -3.8032e-01, -1.4372e+00,  5.0744e-01,  1.5847e+00,
              1.5542e+00, -9.4373e-01,  1.2526e-02],
            [-1.7686e+00, -1.2403e-03,  7.9305e-02, -1.1200e+00, -2.0776e-01,
              4.1425e-01,  7.8020e-01, -4.5932e-01]])




```python
torch.isclose(weights, dequantized_weights, atol=1e-02)
```




    tensor([[True, True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True, True]])




```python
(dequantized_weights - weights).abs().sum()
```




    tensor(0.0846)



### 2. Replace Linear Layers with W8A16LinearLayer


```python
from typing import List

def replace_linear_layer_with_target(
    module: nn.Module,
    target_class: W8A16LinearLayerV1,
    exclude: List
):
  """
  Accept a model and replace nn.Linear Layers in module with target_calss(W8A16LinearLayer)

  Args:
    module(nn.Module): Model.
    target_class(nn.Module): Target class to replace nn.Linear.
    exclude(List): List of modules to exclude from replacement.
  """

  for name, layer in module.named_children():
    if isinstance(layer, nn.Linear) and not any([x == name for x in exclude]):

      # Get bias from layer
      old_bias = layer.bias

      # Create target class to replace
      new_module = target_class(
          in_features=layer.in_features,
          out_features=layer.out_features,
          bias=layer.bias is not None,
          dtype=layer.weight.dtype,
      )

      # Replace
      setattr(module, name, new_module) # Replace name in module with new_module

      # Explicitly set bias
      if old_bias is not None:
        getattr(module, name).bias = old_bias

      # Recursive call for Nested Modules(Ex: Multi-Attention-Head)
    else:
        replace_linear_layer_with_target(
            layer,
            target_class,
            exclude,
        )
```


```python
# Create a Dummy Model to test linear layer replacement function
class DummyModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.emb = nn.Embedding(1, 1)
    self.linear_1 = nn.Linear(1, 1)
    self.linear_2 = nn.Linear(1, 1, bias=False)
    self.lm_head = nn.Linear(1, 1 , bias = False)
```


```python
model_1 = DummyModel()
replace_linear_layer_with_target(
    model_1,
    target_class=W8A16LinearLayerV1,
    exclude=[],
)
```


```python
print(model_1)
```

    DummyModel(
      (emb): Embedding(1, 1)
      (linear_1): W8A16LinearLayerV1()
      (linear_2): W8A16LinearLayerV1()
      (lm_head): W8A16LinearLayerV1()
    )


All nn.Linear aLayers are replaced with W8A16LinearLayerV1.

Exclude lm_head from. Why? Because it's the layer where token logits are calculated to predict next token. With higher precision datatype provides numerical stability and better performance.


```python
model_2 = DummyModel()
replace_linear_layer_with_target(
    model_2,
    target_class=W8A16LinearLayerV1,
    exclude=["lm_head"],
)
```


```python
print(model_2)
```

    DummyModel(
      (emb): Embedding(1, 1)
      (linear_1): W8A16LinearLayerV1()
      (linear_2): W8A16LinearLayerV1()
      (lm_head): Linear(in_features=1, out_features=1, bias=False)
    )



```python
# Let's add quantization as well to the function
from typing import List
def replace_linear_layer_with_target_and_quantize(
    module: nn.Module,
    target_class: W8A16LinearLayerV1,
    exclude: List
):
  """
  Accept a model and replace nn.Linear Layers in module with target_calss(W8A16LinearLayer)
  """

  for name, layer in module.named_children():
    if isinstance(layer, nn.Linear) and not any([x == name for x in exclude]):
      old_bias = layer.bias
      old_weight = layer.weight

      new_module = target_class(
          in_features=layer.in_features,
          out_features=layer.out_features,
          bias=layer.bias is not None,
          dtype=layer.weight.dtype,
      )

      setattr(module, name, new_module)

      getattr(module, name).quantize(old_weight)

      if old_bias is not None:
        getattr(module, name).bias = old_bias

    else:
        replace_linear_layer_with_target_and_quantize(
            layer,
            target_class,
            exclude,
        )
```


```python
model_3 = DummyModel()
replace_linear_layer_with_target_and_quantize(
    model_3,
    target_class=W8A16LinearLayerV1,
    exclude=["lm_head"],
)
```


```python
vars(model_3.linear_1)
```




    {'training': True,
     '_parameters': {'bias': Parameter containing:
      tensor([0.2250], requires_grad=True)},
     '_buffers': {'int8_weights': tensor([[-127]], dtype=torch.int8),
      'scales': tensor([0.0016], grad_fn=<DivBackward0>)},
     '_non_persistent_buffers_set': set(),
     '_backward_pre_hooks': OrderedDict(),
     '_backward_hooks': OrderedDict(),
     '_is_full_backward_hook': None,
     '_forward_hooks': OrderedDict(),
     '_forward_hooks_with_kwargs': OrderedDict(),
     '_forward_hooks_always_called': OrderedDict(),
     '_forward_pre_hooks': OrderedDict(),
     '_forward_pre_hooks_with_kwargs': OrderedDict(),
     '_state_dict_hooks': OrderedDict(),
     '_state_dict_pre_hooks': OrderedDict(),
     '_load_state_dict_pre_hooks': OrderedDict(),
     '_load_state_dict_post_hooks': OrderedDict(),
     '_modules': {}}




```python
model_3.linear_1.int8_weights
```




    tensor([[-127]], dtype=torch.int8)



Now we've built the Quantizer and replaced layers with DummyModels. Next, let's try it out on OpenSource Models.

## Quantize Open Source Models

### Salesforce/codegen-350M-mono


```python
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
model_path = "Salesforce/codegen-350M-mono"
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
```

    Some weights of the model checkpoint at Salesforce/codegen-350M-mono were not used when initializing CodeGenForCausalLM: ['transformer.h.0.attn.causal_mask', 'transformer.h.1.attn.causal_mask', 'transformer.h.10.attn.causal_mask', 'transformer.h.11.attn.causal_mask', 'transformer.h.12.attn.causal_mask', 'transformer.h.13.attn.causal_mask', 'transformer.h.14.attn.causal_mask', 'transformer.h.15.attn.causal_mask', 'transformer.h.16.attn.causal_mask', 'transformer.h.17.attn.causal_mask', 'transformer.h.18.attn.causal_mask', 'transformer.h.19.attn.causal_mask', 'transformer.h.2.attn.causal_mask', 'transformer.h.3.attn.causal_mask', 'transformer.h.4.attn.causal_mask', 'transformer.h.5.attn.causal_mask', 'transformer.h.6.attn.causal_mask', 'transformer.h.7.attn.causal_mask', 'transformer.h.8.attn.causal_mask', 'transformer.h.9.attn.causal_mask']
    - This IS expected if you are initializing CodeGenForCausalLM from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
    - This IS NOT expected if you are initializing CodeGenForCausalLM from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).



```python
# Count number of linear layers
count = 0
for name, layer in model.named_modules():
  if isinstance(layer, nn.Linear):
    count += 1
print(f"Number of Linear Layers: {count}")
```

    Number of Linear Layers: 81



```python
# Time and test the pipeline
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
pipe("def hello_world():", max_new_tokens=20, do_sample=False)
```

    Device set to use mps:0
    Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.





    [{'generated_text': 'def hello_world():\n    print("Hello World")\n\nhello_world()\n\n# 파'}]




```python
model.get_memory_footprint()
```




    1426849792




```python
# Run quantization
replace_linear_layer_with_target_and_quantize(
    model,
    target_class=W8A16LinearLayerV1,
    exclude=["lm_head"],
)
```


```python
torch.mps.is_available()
```




    True




```python
pipe("def hello_world():", max_new_tokens=20, do_sample=False)
```

    Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.





    [{'generated_text': 'def hello_world():\n    print("Hello World")\n\nhello_world()\n\n# 파'}]




```python
model.get_memory_footprint()
```




    672612352



### Alibaba-NLP/gte-Qwen2-1.5B-instruct


```python
from sentence_transformers import SentenceTransformer
embedding_model_path = "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
embedding_model = SentenceTransformer(embedding_model_path)
```


    Loading checkpoint shards:   0%|          | 0/2 [00:00<?, ?it/s]



```python
%%timeit
embedding_model.encode("Hello")
```

    65.9 ms ± 1.21 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)



```python
import os
def get_memory_footprint(model):
  state_dict = model.state_dict()
  tmp_path = "model.pt"
  torch.save(state_dict, tmp_path)
  size = os.path.getsize(tmp_path)
  os.remove(tmp_path)
  return size
```


```python
print(get_memory_footprint(embedding_model))
```

    6173193278



```python
replace_linear_layer_with_target_and_quantize(embedding_model, W8A16LinearLayerV1, exclude=["lm_head"])
```


```python
print(get_memory_footprint(embedding_model))
```

    2245228242



```python
%%timeit
embedding_model.encode("Moshi")
```

    151 ms ± 487 μs per loop (mean ± std. dev. of 7 runs, 1 loop each)


We've saved like 3x space with quantization.

## Memory Effiecient 8-bit loading from HuggingFace Hub

Current design: Load Original model in higher precision, replace LinearLayer with QuantizationLayer, Quantize the weights.

Better Design:

1. Avoid loading original model everytime into RAM for quantization.
Run Quantization on a large machine and store the weights in cloud. We'll use HuggingFaceHub.
2. Then we'll use `meta` from PyTorch. First load skeleton(layers, modules etc) onto RAM. Replace Linear Layers with Quantized Layers. Load weights from hub, load the weights to the skeleton.


```python
import os
from dotenv import load_dotenv
from huggingface_hub import login, HfApi, create_repo
```


```python
# HuggingFaceHub Repo details
HF_USERNAME = "JpChi"
repo_id = f"{HF_USERNAME}/codegen-350M-mono-quantized"
```


```python
# Save quantized weights
torch.save(model.state_dict() ,"quantized_state_dict.pth")
```


```python
# Create hub on huggingface
create_repo(repo_id)
api = HfApi()
# Upload weights
api.upload_file(
    path_or_fileobj="quantized_state_dict.pth",
    path_in_repo="quantized_state_dict.pth",
    repo_id=repo_id,
)
```


    quantized_state_dict.pth:   0%|          | 0.00/673M [00:00<?, ?B/s]





    CommitInfo(commit_url='https://huggingface.co/JpChi/codegen-350M-mono-quantized/commit/2db238a8ba3d0b6e72bb34f5ad6053e0bef9de1f', commit_message='Upload quantized_state_dict.pth with huggingface_hub', commit_description='', oid='2db238a8ba3d0b6e72bb34f5ad6053e0bef9de1f', pr_url=None, repo_url=RepoUrl('https://huggingface.co/JpChi/codegen-350M-mono-quantized', endpoint='https://huggingface.co', repo_type='model', repo_id='JpChi/codegen-350M-mono-quantized'), pr_revision=None, pr_num=None)



Now the quantized weights are successfully pushed to HfHub.


```python
# 1. Load model skeleton without weights
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained(model_path)
config = AutoConfig.from_pretrained(model_path)
with torch.device("meta"):
  model = AutoModelForCausalLM.from_config(config)
```


```python
for param in model.parameters():
  print(param)
  break
```

    Parameter containing:
    tensor(..., device='meta', size=(51200, 1024), dtype=torch.float16,
           requires_grad=True)



```python
# 2. Replace layers
replace_linear_layer_with_target(model, W8A16LinearLayerV1, exclude=["lm_head"])
```


```python
model
```




    CodeGenForCausalLM(
      (transformer): CodeGenModel(
        (wte): Embedding(51200, 1024)
        (drop): Dropout(p=0.0, inplace=False)
        (h): ModuleList(
          (0-19): 20 x CodeGenBlock(
            (ln_1): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
            (attn): CodeGenAttention(
              (attn_dropout): Dropout(p=0.0, inplace=False)
              (resid_dropout): Dropout(p=0.0, inplace=False)
              (qkv_proj): W8A16LinearLayerV1()
              (out_proj): W8A16LinearLayerV1()
            )
            (mlp): CodeGenMLP(
              (fc_in): W8A16LinearLayerV1()
              (fc_out): W8A16LinearLayerV1()
              (act): NewGELUActivation()
              (dropout): Dropout(p=0.0, inplace=False)
            )
          )
        )
        (ln_f): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
      )
      (lm_head): Linear(in_features=1024, out_features=51200, bias=True)
    )



`...` indicates tensor is empty.


```python
# 3. Download weights
from huggingface_hub import hf_hub_download
state_dict_cache = hf_hub_download(repo_id=repo_id, filename="quantized_state_dict.pth")
```


    quantized_state_dict.pth:  83%|########2 | 556M/673M [00:00<?, ?B/s]



```python
# 4. Load weights
state_dict = torch.load(state_dict_cache)
model.load_state_dict(state_dict, strict=True, assign=True)
```




    <All keys matched successfully>



All keys matched successfully -- weights are loaded into skeleton successfully.

## Weights Packing

Problem: We can't store 2-bit or 4-bit PyTorch. Hence quantizing model parameters to 2-bit, 4-bit is wasteful because we'll to store the quantized parameters as `uint8`(the least memory datatype). This causes a overhead 0f 6-bits for a 2-bit parameter.


```python
torch.tensor([1], dtype=torch.int4)
```


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    <ipython-input-4-1df724b7a44d> in <cell line: 0>()
    ----> 1 torch.tensor([1], dtype=torch.int4)
    

    /usr/local/lib/python3.11/dist-packages/torch/__init__.py in __getattr__(name)
       2560             return importlib.import_module(f".{name}", __name__)
       2561 
    -> 2562         raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
       2563 
       2564 


    AttributeError: module 'torch' has no attribute 'int4'



```python
torch.tensor([1], dtype=torch.uint8)
```




    tensor([1], dtype=torch.uint8)



This is where packing comes in, we can pack binaries of [0,1,2,3] into a single 8-bit tensor as follows.


```python
# prompt: create 0's 1's representation for 1 in 8-bit

def binary_representation(number, bits):
  """
  Generates the binary representation of a number with the specified number of bits.

  Args:
    number: The number to convert to binary.
    bits: The number of bits in the representation.

  Returns:
    A string representing the binary form of the number.
  """
  binary = bin(number)[2:]  # Convert to binary and remove the "0b" prefix
  padding = '0' * (bits - len(binary)) #calculate padding
  return padding + binary
```


```python
# Below tensor can be packed into a single uint8
tensor = torch.tensor([0,1,2,3], dtype=torch.uint8)
```


```python
for te in tensor:
  print(f"Number: {te.item()}, binary: {binary_representation(te.item(), 8)}")
```

    Number: 0, binary: 00000000
    Number: 1, binary: 00000001
    Number: 2, binary: 00000010
    Number: 3, binary: 00000011


By excluding the leading zeros, the above 4 parameters stored in 32-bits can be stored in a single uint8 of 8-bits.

11100100 - is the packed tensor - 228.

Advantages:

- It reflects true memory footprint of quantization

Disadvantages:

- Weights have to be unpacked for inference.
- Unpacked tensor needs to be in shape of 8 // nbits.(Check if the inputs fit into 8-bit datatype).


```python
# Using uint8 to avoid handling sign in int8.
# TODO: Try packing int8. 1st bit occupies sign
def pack_weights(uint8tensor, bits):

  """
  Pack quantized weights into a 8-bit tenosr.

  Args:
    uint8tensor: Input tensor
    bits: bits per tensor in uint8tensor
  """

  # Check if the inputs can fit into uint8
  if uint8tensor.shape[0] * bits % 8 != 0: # Total Bits % target datatype bit size
    # The above condition checks if input can be packed into uint8 datatype.
    # If not raise error
    raise ValueError(
        f"The input shape needs to be multiple of {8 / bits}- got {torch.uint8.shape[0]}"
    )

  # Check number of inputs to set the buffer
  num_values = uint8tensor.shape[0] * bits // 8
  # buffer
  packed_tensor = torch.zeros((num_values), dtype=torch.uint8)

  # Number of steps to pack
  num_steps = 8 // bits # Target bits // tensor

  """
  Core Logic:

  We'll loop through num_values
    4 steps per value
      In each step we'll shift the value with step and perfom an Binary OR Operation

  Buffer ==> [0000 0000]
  num_values - 1:
    (bits * j) - (2 * 0) - No Shift
    0 - [0000 0000]
    buffer ==> [0000 0000]

  num_values - 2:
    (bits * j) - (2 * 0) - 2 shifts
    current input 1 - [0000 0001]
    Post two shifts - [0000 0100]
    [0000 0000] OR [0000 0100] - buffer OR current value
    buffer ==> [0000 0100]

  This repeats!
  """
  unpacked_idx = 0
  print(f"Num values: {num_values}")
  for i in range(num_values):
    print(f"Current value: {i}")
    for j in range(num_steps):
      print(f"Current step: {j}")
      # |= BitwiswOR << Shift
      packed_tensor[i] |= uint8tensor[unpacked_idx] << (j * bits)
      unpacked_idx += 1
  return packed_tensor
```


```python
pack_weights(tensor, 2)
```




    tensor([228], dtype=torch.uint8)



Correct value of 228 is obtained after packing  `tensor`.


```python
tensor1 = torch.tensor([0,1,2,3,0,1,2,3], dtype=torch.int8)
```


```python
for te in tensor1:
  print(f"Number: {te.item()}, binary: {binary_representation(te.item(), 8)}")
```

    Number: 0, binary: 00000000
    Number: 1, binary: 00000001
    Number: 2, binary: 00000010
    Number: 3, binary: 00000011
    Number: 0, binary: 00000000
    Number: 1, binary: 00000001
    Number: 2, binary: 00000010
    Number: 3, binary: 00000011



```python
pack_weights(tensor1, 2)
```

    Num values: 2
    Current value: 0
    Current step: 0
    Current step: 1
    Current step: 2
    Current step: 3
    Current value: 1
    Current step: 0
    Current step: 1
    Current step: 2
    Current step: 3





    tensor([228, 228], dtype=torch.uint8)



### UnPack Weights

Unpacking, logic will be covered in code.

Input: [228, 228]
Output: [0, 1, 2, 3, 0, 1, 2, 3]


```python
def unpack_weights(uint8tensor, bits):

  # Number of values for unpacked tensor
  num_values = uint8tensor.shape[0] * 8 // bits # For our input 2*8 // 2 = 8
  # buffer
  unpacked_tensor = torch.zeros((num_values), dtype=torch.uint8)

  # num steps per packed tensor
  num_steps = 8 // bits

  """
  Core logic:

  We'll loop through num_values
    4 steps per value
      In each step we'll shift the value to right with step to get the 2 bits for the respective tensor
  uint8 tensor = 11100100
  Buffer ==> [00000000, 00000000, 00000000,...] # 8 values
  num_values - 1:
    (bits * j) - (2 * 0) - No Shift
    0 - []
    buffer ==> [11100100, 00000000, 00000000,...]
  num_values - 2:
    (bits * j) - (2 * 0) - 2 shifts
    11100100 (2 shifts) --> 00111001
    Post two shifts - [11100100, 00111001, 00000000,...]
  num_values - 3:
    (bits * j) - (2 * 0) - 4 shifts
    11100100 (4 shifts) --> 00001110
    Post four shifts - [11100100, 00111001, 00001110,...]

  This repeats!

  Finally we'll use a mask and run logical AND with buffer.
  Mask: [0000 0011]
  In AND with this mask only last two bits from input will be retained.

  Input: [11100100, 00111001, 00001110,...]
  Mask: [00000011, 00000011, 00000011,...]
  Output of AND: [00000000, 00000001, 00000011]
  """

  unpacked_idx = 0

  print(f"Number of values: {uint8tensor.shape[0]}")
  print(f"Number of steps: {num_steps}")
  for i in range(uint8tensor.shape[0]):
    print(f"Current value: {i}")
    for j in range(num_steps):
      print(f'Step: {j}')
      unpacked_tensor[unpacked_idx] |= uint8tensor[i] >> (bits * j)
      unpacked_idx += 1

  # mask
  mask = 2 ** bits - 1 # 2 - 0000 0011

  # binary AND
  unpacked_tensor &= mask
  return unpacked_tensor
```


```python
print(f"Unpacked tensor: {tensor1}")
packed_tensor = pack_weights(tensor1, 2)
packed_tensor
```

    Unpacked tensor: tensor([0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.int8)
    Num values: 2
    Current value: 0
    Current step: 0
    Current step: 1
    Current step: 2
    Current step: 3
    Current value: 1
    Current step: 0
    Current step: 1
    Current step: 2
    Current step: 3





    tensor([228, 228], dtype=torch.uint8)




```python
# Unpack
unpacked_tensor = unpack_weights(packed_tensor, 2)
unpacked_tensor
```

    Number of values: 2
    Number of steps: 4
    Current value: 0
    Step: 0
    Step: 1
    Step: 2
    Step: 3
    Current value: 1
    Step: 0
    Step: 1
    Step: 2
    Step: 3





    tensor([0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.uint8)




```python
unpacked_tensor == tensor1
```




    tensor([True, True, True, True, True, True, True, True])



Unpacked Tensor matches with original Tensor.

Limitations:

- This implementation can only quantize 0,1,2,3 to 2-bits larger numbers require higher bits.
- Doesn't work with arbitrary shapes.(Inputs have to be multiples of 8).
- Works only for 2-bit
- Naive Algorithm with two loops.

