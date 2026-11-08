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
def lrelu(x):
      return np.maximum(0.1*x, x)

def lrelu_derivative(x):
      return (x>0).astype(float) + 0.1*(x<=0).astype(float)

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
      N1 = lrelu(L1)
      L2 = N1 @ W2 + B2
      N2 = sigmoid(L2)
      return L1, N1, L2, N2

#loss function
def loss(ans, N2):
      return np.mean((ans - N2)**2)

def perceptual_loss(current_loss):
      # Display-only score: 0 means ideal; 1 means very poor/uncertain.
      # MSE is 0.25 when every prediction is 0.5, so this maps that baseline to 1.
      return min(1.0, 2 * np.sqrt(current_loss))

def backpropagate(N0, ans, W2, L1, N1, N2):
      # Backpropagation: output layer
      dN2 = (2 / ans.size) * (N2 - ans)
      dL2 = dN2 * N2 * (1 - N2)

      # Gradients for the output-layer parameters
      dW2 = N1.T @ dL2
      dB2 = np.sum(dL2, axis=0, keepdims=True)

      # Carry the output gradient back through the hidden layer and ReLU
      dN1 = dL2 @ W2.T
      dL1 = dN1 * lrelu_derivative(L1)

      # Gradients for the input-to-hidden parameters
      dW1 = N0.T @ dL1
      dB1 = np.sum(dL1, axis=0, keepdims=True)
      return {"W1": dW1, "B1": dB1, "W2": dW2, "B2": dB2}

def adjust_parameters(parameters, gradients, optimizer, learning_rate, state, step):
      if optimizer == "SGD":
            for name in parameters:
                  parameters[name] -= learning_rate * gradients[name]

      elif optimizer == "Momentum":
            momentum = 0.9
            for name in parameters:
                  state[name] = momentum * state[name] + gradients[name]
                  parameters[name] -= learning_rate * state[name]

def main():
      global W1, B1, W2, B2

      # Ask how many full-batch training cycles to run
      iterations = int(input("Number of training iterations: "))

      print("\nChoose an optimization algorithm:")
      print("1. SGD (simple gradient descent)")
      print("2. Momentum (smooths updates across steps)")
      choice = input("Enter 1, or 2:")
      optimizer_names = {"1": "SGD", "2": "Momentum"}
      if choice not in optimizer_names:
            raise ValueError("Choose 1 or 2. ")
      optimizer = optimizer_names[choice]

      learning_rate = float(input("Starting learning rate: "))

      parameters = {"W1": W1, "B1": B1, "W2": W2, "B2": B2}
      # Momentum keeps one velocity array for each parameter.
      state = {name: np.zeros_like(value) for name, value in parameters.items()}

      print(f"\nTraining with {optimizer}, learning rate {learning_rate}\n")
      for step in range(1, iterations + 1):
            # One cycle: forward -> measure -> backpropagate -> adjust -> forward again
            L1, N1, _, N2 = forward(N0, W1, B1, W2, B2)
            gradients = backpropagate(N0, ans, W2, L1, N1, N2)
            adjust_parameters(parameters, gradients, optimizer, learning_rate, state, step)

            # Show measurements after this update, so each row reflects the new parameters.
            _, _, _, N2 = forward(N0, W1, B1, W2, B2)
            current_loss = loss(ans, N2)
            print(f"Step {step}/{iterations}")
            print("Predictions:")
            print(np.round(N2, 4))
            print(f"Loss (MSE): {current_loss:.8f}")
            print(f"Perceptual score: {perceptual_loss(current_loss):.4f}\n")
if __name__ == "__main__":
      main()




