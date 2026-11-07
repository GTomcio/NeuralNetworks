#neural network for XOR problem
import numpy as np

#training set
N0=np.array([[0, 0], 
            [0, 1], 
            [1, 0], 
            [1, 1]
            ])
#solutions
ans=np.array([[0], 
            [1], 
            [1], 
            [0]
            ])

#define functions
def relu(x):
      return np.maximum(0, x)

def relu_derivative(x):
      return (x > 0).astype(float)

def sigmoid(x):
      x=np.clip(x, -500, 500)
      return 1/(1+np.exp(-x))

#structure definition
rng = np.random.default_rng(42)
W1 = rng.normal(0, 0.1, (2, 2))
B1= np.zeros((1, 2))
W2 = rng.normal(0, 0.1, (2, 1))
B2= np.zeros((1, 1))

#forward pass
def forward(N0, W1, B1, W2, B2):
      L1 = N0 @ W1 + B1
      N1 = relu(L1)
      L2 = N1 @ W2 + B2
      N2 = sigmoid(L2)
      return L1, N1, L2, N2
L1, N1, L2, N2 = forward(N0, W1, B1, W2, B2)

#loss function
def loss(ans, N2):
      return np.mean((ans - N2)**2)
CurrentLoss = loss(ans, N2)

print("N2 = ", N2)
print("loss = ", CurrentLoss)

#BACKPROPAGATION

#backpropagation: output layer
dN2 = (2 / ans.size) * (N2 - ans)
dL2 = dN2 * N2 * (1 - N2)

print("dN2 =", dN2)
print("dL2 =", dL2)

print("weights: ", W1, W2)

#gradients for the output-layer parameters
dW2 = N1.T @ dL2
dB2 = np.sum(dL2, axis=0, keepdims=True)

print("dW2 =", dW2)
print("dB2 =", dB2)




