import random
from Value import Value  


class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

class Neuron(Module):

    def __init__(self, nin, nonlin=True): 
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]  # all weights random
        self.b = Value(random.uniform(-1,1))   # all biases random
        self.nonlin = nonlin  # non linearity=true by default
        
    def __call__(self, x):
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)  # w * x + b
        return act.relu() if self.nonlin else act 

    def parameters(self):
        return self.w + [self.b]


class Layer(Module):

    def __init__(self, nin, nout, nonlin=True): 
        self.neurons = [Neuron(nin, nonlin=nonlin) for _ in range(nout)] #list of neurons, nin=dimensionality, nout=how many neurons OR number of outputs

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class MLP(Module):

    def __init__(self, nin, nouts):  # nin is same as Layer but nouts is list of all the layers we want in our MLP
        sz = [nin] + nouts    
        self.layers = [Layer(sz[i], sz[i+1], nonlin=(i != len(nouts) - 1)) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

