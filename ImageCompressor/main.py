import numpy as np
import struct
from pathlib import Path
import matplotlib.pyplot as plt

#define MNIST data path
def load_images(filename):
    data_dir = Path(__file__).parent / "data" / "MNIST" / "raw"
    with (data_dir / filename).open("rb") as file:
            magic, num, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError(f"Invalid magic number {magic} in MNIST image file: {filename}")
            
            images = np.frombuffer(file.read(), dtype=np.uint8)
    return images.reshape(num, rows * cols).astype(np.float32) / 255.0

#define images
images = load_images("train-images-idx3-ubyte")
test_images = load_images("t10k-images-idx3-ubyte")

####################################################################

#define widths
i_size = 784
n1_size = 576
n2_size = 484
o_size = 256

#define input vector
I = images[1]

#define weight matrices
W1 = np.random.randn(i_size, n1_size).astype(np.float32) * np.sqrt(2 / i_size)
W2 = np.random.randn(n1_size, n2_size).astype(np.float32) * np.sqrt(2 / n1_size)
W3 = np.random.randn(n2_size, o_size).astype(np.float32) * np.sqrt(2 / n2_size)

#define bias vectors
B1 = np.zeros(n1_size).astype(np.float32)
B2 = np.zeros(n2_size).astype(np.float32)
B3 = np.zeros(o_size).astype(np.float32)

#define ReLu
def LReLu(x):
    return np.maximum(0.1*x, x)

#define ReLu derivative
def LReLu_derivative(x):
    return np.where(x > 0, 1.0, 0.1).astype(np.float32)

#define forward pass
def forward(I, W1, B1, W2, B2, W3, B3):
    L1 = I @ W1 + B1
    N1 = LReLu(L1)
    L2 = N1 @ W2 + B2
    N2 = LReLu(L2)
    L3 = N2 @ W3 + B3
    N3 = LReLu(L3)

    return L1, N1, L2, N2, L3, N3

##############################################################

#define decoder widths

DI_size = 256
DN1_size = 484
DN2_size = 576
DN3_size = 784

#define decoder weight matrices
AW1 = np.random.randn(DI_size, DN1_size).astype(np.float32) * np.sqrt(2 / DI_size)
AW2 = np.random.randn(DN1_size, DN2_size).astype(np.float32) * np.sqrt(2 / DN1_size)
AW3 = np.random.randn(DN2_size, DN3_size).astype(np.float32) * np.sqrt(1 / DN2_size)

#define decoder bias vectors
AB1 = np.zeros(DN1_size).astype(np.float32)
AB2 = np.zeros(DN2_size).astype(np.float32)
AB3 = np.zeros(DN3_size).astype(np.float32)

#define sigmoid
def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

#define decoder forward pass
def decoder_forward(DI, AW1, AB1, AW2, AB2, AW3, AB3):
     AL1 = DI @ AW1 + AB1
     AN1 = LReLu(AL1)
     AL2 = AN1 @ AW2 + AB2
     AN2 = LReLu(AL2)
     AL3 = AN2 @ AW3 + AB3
     AN3 = sigmoid(AL3)

     return AL1, AN1, AL2, AN2, AL3, AN3

###############################################################

#define loss function
def loss(AN3, I):
    return np.mean((AN3 - I) ** 2)

#define backpropagation
def backpropagate(I, N1, N2, N3, L1, L2, L3,
                  AN1, AN2, AN3, AL1, AL2, AL3,
                  W1, W2, W3, AW1, AW2, AW3):
    
    dAL3 = (2 / I.size) * (AN3 - I) * AN3 * (1 - AN3)

   # Decoder layer 3
    dAW3 = np.outer(AN2, dAL3)
    dAB3 = dAL3
    dAN2 = dAL3 @ AW3.T

    # Decoder layer 2: ReLU
    dAL2 = dAN2 * LReLu_derivative(AL2)
    dAW2 = np.outer(AN1, dAL2)
    dAB2 = dAL2
    dAN1 = dAL2 @ AW2.T

    # Decoder layer 1: ReLU
    dAL1 = dAN1 * LReLu_derivative(AL1)
    dAW1 = np.outer(N3, dAL1)
    dAB1 = dAL1
    dN3 = dAL1 @ AW1.T

    # Encoder layer 3: ReLU
    dL3 = dN3 * LReLu_derivative(L3)
    dW3 = np.outer(N2, dL3)
    dB3 = dL3
    dN2 = dL3 @ W3.T

    # Encoder layer 2: ReLU
    dL2 = dN2 * LReLu_derivative(L2)
    dW2 = np.outer(N1, dL2)
    dB2 = dL2
    dN1 = dL2 @ W2.T

    # Encoder layer 1: ReLU
    dL1 = dN1 * LReLu_derivative(L1)
    dW1 = np.outer(I, dL1)
    dB1 = dL1

    return {
        "AW1": dAW1, "AB1": dAB1,
        "AW2": dAW2, "AB2": dAB2,
        "AW3": dAW3, "AB3": dAB3,
        "W1": dW1, "B1": dB1,
        "W2": dW2, "B2": dB2,
        "W3": dW3, "B3": dB3,
    }

#define parameters
parameters = {
    "W1": W1, "B1": B1,
    "W2": W2, "B2": B2,
    "W3": W3, "B3": B3,
    "AW1": AW1, "AB1": AB1,
    "AW2": AW2, "AB2": AB2,
    "AW3": AW3, "AB3": AB3,
}

#define parameter update
def parameter_update(parameters, gradients, learning_rate):
    for name in parameters:
        parameters[name] -= learning_rate * gradients[name]

#####################################################################

#training loop
training_repetitions = int(input("training_repetitions = "))
learning_rate = float(input("learning_rate = "))
number_of_images = int(input("number_of_images = "))

i=1

for repetition in range(training_repetitions):
    for I in images[:number_of_images]:
        #call forward pass
        L1, N1, L2, N2, L3, N3 = forward(I, W1, B1, W2, B2, W3, B3)

        #call decoder forward pass
        AL1, AN1, AL2, AN2, AL3, AN3 = decoder_forward(N3, AW1, AB1, AW2, AB2, AW3, AB3)

        #call loss
        current_loss = loss(AN3, I)

        #call backpropagation
        gradients = backpropagate(
        I, N1, N2, N3, L1, L2, L3,
        AN1, AN2, AN3, AL1, AL2, AL3,
        W1, W2, W3, AW1, AW2, AW3
        )
        #call parameter update
        parameter_update(parameters, gradients, learning_rate)
    
        #print update
        print("iteration", i, "loss:", current_loss)

    #update counter
    i=i+1

############################################################################

#count average pixel error
print("training complete")

test_count = 10_000
test_total_loss = 0.0

for I in test_images[:test_count]:
    L1, N1, L2, N2, L3, N3 = forward(I, W1, B1, W2, B2, W3, B3)
    AL1, AN1, AL2, AN2, AL3, AN3 = decoder_forward(
        N3, AW1, AB1, AW2, AB2, AW3, AB3
    )
    test_total_loss += loss(AN3, I)

test_mse = test_total_loss / test_count
test_rmse = np.sqrt(test_mse)

print("test MSE:", test_mse)
print("test RMSE:", test_rmse)

I = test_images[0]

L1, N1, L2, N2, L3, N3 = forward(I, W1, B1, W2, B2, W3, B3)
AL1, AN1, AL2, AN2, AL3, AN3 = decoder_forward(
    N3, AW1, AB1, AW2, AB2, AW3, AB3
)

#plot before and after
plt.subplot(1, 2, 1)
plt.imshow(I.reshape(28, 28), cmap="gray")
plt.title("Original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(AN3.reshape(28, 28), cmap="gray")
plt.title("Reconstruction")
plt.axis("off")

plt.show()





