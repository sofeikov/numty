"""README showcase: plain numpy, no annotations, two classic bugs.

Nothing here imports numty. The shapes come from the constructors and
flow through reshape, matmul and the reduction on their own.
"""
import numpy as np

images = np.zeros((64, 28, 28))
weights = np.zeros((10, 784))  # transposed: should be (784, 10)
bias = np.zeros(10)

flat = images.reshape(64, -1)
logits = flat @ weights + bias  # E: matmul inner dimensions disagree

scores = np.zeros((64, 10))
exp = np.exp(scores)
probs = exp / exp.sum(axis=1)  # E: forgot keepdims=True
