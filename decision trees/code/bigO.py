import numpy as np

# O(N)
def func1(N):
    for i in range(N):
        print("Hello!")

# O(N)
def func2(N):
    x = np.zeros(N)
    x += 1000
    return x

# O(1000) = O(1)
def func3(N):
    x = np.zeros(1000)
    x = x * N
    return x

# O(N^2)
def func4(N):
    x = 0
    for i in range(N):
        for j in range(i, N):
            x += i * j
    return x
