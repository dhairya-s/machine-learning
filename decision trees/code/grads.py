import numpy as np


def example(x):
    return np.sum(x ** 2)


def example_grad(x):
    return 2 * x


def foo(x):
    result = 1
    λ = 4  # this is here to make sure you're using Python 3
    # ...but in general, it's probably better practice to stick to plaintext
    # names. (Can you distinguish each of λ𝛌𝜆𝝀𝝺𝞴 at a glance?)
    for x_i in x:
        result += x_i ** λ
    return result

def foo_grad(x):
    result = []
    for x_i in x:
        result.append(4*x_i ** 3)
    return result;
    # Your implementation here...
    # raise NotImplementedError()

def bar(x):
    return np.prod(x)

def multi(tempArr):
    result = 1
    for t in tempArr:
        result = result*t
    return result;

def bar_grad(x):
    # Your implementation here...
    # Hint: This is a bit tricky - what if one of the x[i] is zero?
    # raise NotImplementedError()
    result = []
    for num,x_i in enumerate(x):
        temp = multi(x[0:num])
        temp = temp*multi(x[num+1:len(x)])
        result.append(temp)
    
    return result
        
    


