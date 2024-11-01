#Function to check for proper 0-1 range.
from logging import exception
def check_range(x):
    x = float(x)
    if (x<0 or x>1):
        raise Exception("Error, number falls out of valid range")
    else:
        return x

#Calculate nodes
def calculate(obj, node1, node2=None, node3=None):
    if node3 is None:
        if node2 is None:
            result = (obj.a1 * node1) + obj.bias
        else:
            result = (obj.a1 * node1) + (obj.a2 * node2) + obj.bias
    else:
        result = (obj.a1 * node1) + (obj.a2 * node2) + (obj.a3 * node3) + obj.bias
    return result
#Sigmoid function
import math
import numpy as np
def sigmoid(x):
    try:
        return 1/(1+(np.exp(-x)))
    except Exception:
        raise Exception(f"error with {x}")
