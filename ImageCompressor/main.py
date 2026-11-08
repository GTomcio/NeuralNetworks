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

####################################################################

#define widths
i_size = 784
n1_size = 576
n2_size = 484
o_size = 256

#define input vector
I = images[0]

#define weight matrices
W1 = np.random.randn(i_size, n1_size).astype(np.float32) * 0.01
W2 = np.random.randn(n1_size, n2_size).astype(np.float32) * 0.01
W3 = np.random.randn(n2_size, o_size).astype(np.float32) * 0.01

#define bias vectors
B1 = np.zeros(n1_size).astype(np.float32)
B2 = np.zeros(n2_size).astype(np.float32)
B3 = np.zeros(o_size).astype(np.float32)

#define ReLu
def ReLu(x):
    return np.maximum(0, x)

#define ReLu derivative
def Relu_derivative(x):
    return (x > 0).astype(np.float32)

#define forward pass
def forward(I, W1, B1, W2, B2, W3, B3):
    L1 = I @ W1 + B1
    N1 = ReLu(L1)
    L2 = N1 @ W2 + B2
    N2 = ReLu(L2)
    L3 = N2 @ W3 + B3
    N3 = ReLu(L3)

    return L1, N1, L2, N2, L3, N3

#call forward pass
L1, N1, L2, N2, L3, N3 = forward(I, W1, B1, W2, B2, W3, B3)

##############################################################

#define decoder widths

DI_size = 256
DN1_size = 484
DN2_size = 576
DN3_size = 784

#define decoder weight matrices
DW1 = np.random.randn(DI_size, DN1_size).astype(np.float32) * 0.01
DW2 = np.random.randn(DN1_size, DN2_size).astype(np.float32) * 0.01
DW3 = np.random.randn(DN2_size, DN3_size).astype(np.float32) * 0.01

#define decoder bias vectors
DB1 = np.zeros(DN1_size).astype(np.float32)
DB2 = np.zeros(DN2_size).astype(np.float32)
DB3 = np.zeros(DN3_size).astype(np.float32)

#define sigmoid
def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

#define decoder forward pass
def decoder_forward(DI, DW1, DB1, DW2, DB2, DW3, DB3):
     DL1 = DI @ DW1 + DB1
     DN1 = ReLu(DL1)
     DL2 = DN1 @ DW2 + DB2
     DN2 = ReLu(DL2)
     DL3 = DN2 @ DW3 + DB3
     DN3 = sigmoid(DL3)

     return DL1, DN1, DL2, DN2, DL3, DN3

#call decoder forward pass
DL1, DN1, DL2, DN2, DL3, DN3 = decoder_forward(N3, DW1, DB1, DW2, DB2, DW3, DB3)

###############################################################

#define loss function
def loss(DN3, I):
    return np.mean((DN3 - I) ** 2)

#define backpropagation
def backpropagate(I, N1, N2, N3, L1, L2, L3,
                  DN1, DN2, DN3, DL1, DL2, DL3,
                  W1, W2, W3, DW1, DW2, DW3):
    
    dDL3 = (2 / I.size) * (DN3 - I) * DN3 * (1 - DN3)

    dDW3 = np.outer(DN2, dDL3)
    dDB3 = dDL3
    dDN2 = dDL3 @ DW3.T

