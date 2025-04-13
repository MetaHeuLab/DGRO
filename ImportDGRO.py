from DGRO import DGRO
import numpy as np
import math


def func(X):
    dim = len(X)
    fit = (
        -20 * np.exp(-0.2 * np.sqrt(np.sum(X ** 2) / dim))
        - np.exp(np.sum(np.cos(2 * math.pi * X)) / dim)
        + 20
        + np.exp(1)
    )

    return fit

no_runs = 3
count = 0
dim= 10
pop = 30
itera = 3000
lb = -100
ub = 100

for i in range(no_runs):
    DGRO(func, lb, ub, dim, pop, itera)
