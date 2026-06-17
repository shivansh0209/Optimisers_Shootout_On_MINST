# Optimizer Shootout on Fashion-MNIST

A comparative study of popular optimization algorithms and learning rate scheduling techniques on the Fashion-MNIST dataset using TensorFlow/Keras.

---

## Project Overview

The objective of this project is to understand how different optimization algorithms affect neural network training dynamics, convergence speed, final accuracy, and computational efficiency.

The same neural network architecture is trained multiple times while varying:

* Optimizer
* Learning Rate
* Learning Rate Scheduler

The project also includes a manual NumPy implementation of SGD with Momentum to understand the underlying mathematics before using framework implementations.

---

## Dataset

Dataset: Fashion-MNIST

Fashion-MNIST is a drop-in replacement for MNIST consisting of 70,000 grayscale images of clothing items belonging to 10 classes.

### Classes

| Label | Class       |
| ----- | ----------- |
| 0     | T-shirt/Top |
| 1     | Trouser     |
| 2     | Pullover    |
| 3     | Dress       |
| 4     | Coat        |
| 5     | Sandal      |
| 6     | Shirt       |
| 7     | Sneaker     |
| 8     | Bag         |
| 9     | Ankle Boot  |

---

## Neural Network Architecture

```text
Input Layer (784)

Dense(256, ReLU)
BatchNormalization
Dropout(0.3)

Dense(128, ReLU)
BatchNormalization
Dropout(0.3)

Dense(64, ReLU)
BatchNormalization
Dropout(0.3)

Dense(10, Softmax)
```

### Loss Function

```python
SparseCategoricalCrossentropy
```

### Metric

```python
Accuracy
```

### Training Configuration

```python
Epochs = 20
Batch Size = 128
Validation Split = 0.2
EarlyStopping(patience=4)
```

---

# Optimizers Compared

The following optimizers were evaluated:

## 1. SGD

Standard Stochastic Gradient Descent

Update rule:

```text
w = w - η∇L
```

Characteristics:

* Simple
* Memory efficient
* Slow convergence
* Sensitive to learning rate

---

## 2. SGD + Momentum

Adds velocity to gradient updates.

Update rule:

```text
v = βv - η∇L
w = w + v
```

Characteristics:

* Faster convergence
* Reduced oscillations
* Better navigation through ravines

---

## 3. Nesterov Accelerated Gradient (NAG)

Computes gradients after taking a look-ahead step.

Characteristics:

* More informed updates
* Often converges faster than Momentum

---

## 4. AdaGrad

Adapts learning rates individually for each parameter.

Characteristics:

* Works well for sparse features
* Learning rate continually shrinks

---

## 5. RMSprop

Improves AdaGrad by using an exponentially weighted average of squared gradients.

Characteristics:

* Stable learning
* Faster convergence
* Common choice for deep learning

---

## 6. Adam

Combines Momentum and RMSprop.

Characteristics:

* Adaptive learning rate
* Fast convergence
* Generally strong default optimizer

---

# Manual NumPy Implementation

Before using Keras implementations, SGD with Momentum was implemented manually using NumPy.

### Algorithm

```python
velocity = beta * velocity - lr * gradient
weights += velocity
```

Purpose:

* Understand optimizer internals
* Visualize momentum accumulation
* Verify mathematical intuition

---

# Learning Rate Experiments

The following initial learning rates were tested:

```text
0.1
0.01
0.001
0.0001
```

Metrics observed:

* Convergence speed
* Stability
* Final validation accuracy
* Final validation loss

### Observations

#### LR = 0.1

* Fastest convergence
* Worked surprisingly well with schedulers
* Best performance with cosine annealing

#### LR = 0.01

* Stable baseline
* Good balance between speed and accuracy

#### LR = 0.001

* Slow convergence
* Often required more epochs

#### LR = 0.0001

* Extremely slow learning
* Underfitting observed

---

# Learning Rate Scheduling

Three scheduling techniques were implemented.

---

## Step Decay

Learning rate reduced by 50% every 5 epochs.

Example:

```text
0.0100
0.0100
0.0100
0.0100
0.0100
0.0050
0.0050
...
```

Characteristics:

* Simple
* Stable convergence
* Sudden drops in learning rate

---

## Exponential Decay

Learning rate decreases continuously.

Formula:

```text
lr = lr₀ × e^(-k·epoch)
```

Characteristics:

* Smooth decay
* More exploration early in training
* Slightly noisier validation curves

---

## Cosine Annealing

Learning rate follows a cosine curve.

Formula:

```text
lr = lr_min + 0.5(lr_max - lr_min)
     × (1 + cos(π·epoch/T))
```

Characteristics:

* Very smooth convergence
* Large learning rates early
* Fine-tuning near the end

Example:

```text
Epoch 1  : 0.100000
Epoch 10 : 0.058240
Epoch 20 : 0.000161
```

---

# Metrics Collected

For every experiment:

### Training Metrics

* Training Loss
* Validation Loss
* Training Accuracy
* Validation Accuracy

### Computational Metrics

* Wall-clock Time per Epoch

Custom callback:

```python
class TimeHistory(tf.keras.callbacks.Callback):
    ...
```

used to measure epoch durations.

---

# Key Observations

## Optimizers

* SGD converged the slowest.
* Momentum significantly accelerated SGD.
* NAG provided slightly smoother convergence than Momentum.
* AdaGrad plateaued earlier due to aggressive learning rate reduction.
* RMSprop converged rapidly and remained stable.
* Adam consistently provided strong performance with minimal tuning.

---

## Learning Rate Schedulers

### Step Decay

* Stable and predictable.
* Produced strong validation performance.

### Exponential Decay

* Slightly noisier curves.
* Similar final performance to Step Decay.

### Cosine Annealing

* Most graceful convergence.
* Performed best when paired with a larger initial learning rate.
* Small initial learning rates caused premature convergence.

---

# Challenges Encountered

### Binary Cross Entropy Shape Error

Initially:

```python
metrics=['accuracy', 'binary_crossentropy']
```

caused:

```text
ValueError:
target.shape=(batch_size,1)
output.shape=(batch_size,10)
```

because Fashion-MNIST is a multiclass classification problem.

Solution:

```python
metrics=['accuracy']
```

---

### Scheduler Visualization

Learning rate schedulers were functioning correctly, but their effect was not always visually obvious in loss/accuracy plots.

To verify scheduler behavior, learning rates were logged each epoch and plotted separately.

---

# Conclusions

1. Adam and RMSprop achieved the fastest convergence.
2. SGD benefited significantly from Momentum and NAG.
3. Learning rate scheduling improved convergence stability.
4. Cosine Annealing delivered the strongest results when paired with a sufficiently large initial learning rate.
5. Choosing an appropriate learning rate had a larger impact than switching between many optimizers.

---

# Future Improvements

* Add SGD with Warm Restarts (SGDR)
* Compare optimizers on CIFAR-10
* Add Weight Decay (AdamW)
* Perform Hyperparameter Optimization
* Compare Batch Normalization vs No Batch Normalization
* Analyze gradient norms during training

---

## Technologies Used

* Python
* NumPy
* TensorFlow
* Keras
* Matplotlib
* Fashion-MNIST

---

## Author

**Shivansh Pandey**

Project completed as part of a hands-on exploration of optimization algorithms and learning rate scheduling techniques in Deep Learning.
