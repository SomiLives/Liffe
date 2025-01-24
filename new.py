import torch
import torch.nn as nn
import mitdeeplearning as mdl
import numpy as np
import matplotlib.pyplot as plt

integer = torch.tensor(1234)
decimal = torch.tensor(3.14159265359)

print(f"`integer` is a {integer.ndim}-d Tensor: {integer}")
print(f"`decimal` is a {decimal.ndim}-d Tensor: {decimal}")