import numpy
import torch
import sklearn
import os

print("NumPy configuration:")
numpy.__config__.show()

print("\nPyTorch OpenMP info:")
print(torch.__config__.parallel_info())

print("\nScikit-learn configuration:")
sklearn.__config__.show()
