import numpy as np
import torch
import struct
from pathlib import Path
import matplotlib.pyplot as plt
import torch.nn as nn

#define device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

#define MNIST image loader
def load_images(filename):
    data_dir = Path(__file__).parent / "data" / "MNIST" / "raw"

    with (data_dir / filename).open("rb") as file:
        magic, num, rows, cols = struct.unpack(">IIII", file.read(16))
        if magic != 2051:
            raise ValueError(
                f"Invalid magic number {magic} in MNIST image file: {filename}"
            )

        image_bytes = np.frombuffer(file.read(), dtype=np.uint8).copy()

    images = torch.from_numpy(image_bytes).reshape(num, rows * cols)
    return images.to(device=DEVICE, dtype=torch.float32).div_(255.0)

#define images
images = load_images("train-images-idx3-ubyte")
test_images = load_images("t10k-images-idx3-ubyte")

#define encoder widths
i_size = 784
n1_size = 576
n2_size = 484
o_size = 256

DI_size = 256
DN1_size = 484
DN2_size = 576
DN3_size = 784

#define autoencoder model
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(i_size, n1_size),
            nn.LeakyReLU(0.1),
            nn.Linear(n1_size, n2_size),
            nn.LeakyReLU(0.1),
            nn.Linear(n2_size, o_size),
            nn.LeakyReLU(0.1),
        )

        self.decoder = nn.Sequential(
            nn.Linear(DI_size, DN1_size),
            nn.LeakyReLU(0.1),
            nn.Linear(DN1_size, DN2_size),
            nn.LeakyReLU(0.1),
            nn.Linear(DN2_size, DN3_size),
            nn.Sigmoid(),
        )

        #define initial weights and biases
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                scale = 1.0 if layer is self.decoder[4] else 2.0
                std = np.sqrt(scale / layer.in_features)
                nn.init.normal_(layer.weight, mean=0.0, std=std)
                nn.init.zeros_(layer.bias)

    #define forward pass
    def forward(self, x):
        return self.decoder(self.encoder(x))

#initialize model on device
model = Autoencoder().to(DEVICE)

#get training parameters
training_repetitions = int(input("training_repetitions = "))
learning_rate = float(input("learning_rate = "))
number_of_images = int(input("number_of_images = "))
momentum = float(input("momentum = "))

#define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=learning_rate,
    momentum=momentum,
)

#define batch size and training images
batch_size = 128
training_images = images[:number_of_images]

#training loop
for epoch in range(training_repetitions):
    model.train()
    total_loss = 0.0

    for start in range(0, len(training_images), batch_size):
        batch = training_images[start : start + batch_size]

        optimizer.zero_grad(set_to_none=True)
        reconstruction = model(batch)
        loss = criterion(reconstruction, batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(batch)

    average_loss = total_loss / len(training_images)
    print(f"epoch {epoch + 1}, loss: {average_loss:.6f}")

#evaluate model on test images
test_count = min(10_000, len(test_images))
test_total_loss = 0.0

model.eval()
with torch.no_grad():
    for start in range(0, test_count, batch_size):
        batch = test_images[start : start + batch_size]
        reconstruction = model(batch)
        test_total_loss += criterion(reconstruction, batch).item() * len(batch)

test_mse = test_total_loss / test_count
test_rmse = test_mse ** 0.5

print("test MSE:", test_mse)
print("test RMSE:", test_rmse)

#plot original image and reconstruction
with torch.no_grad():
    original = test_images[2]
    reconstruction = model(original.unsqueeze(0))[0]

plt.subplot(1, 2, 1)
plt.imshow(original.cpu().reshape(28, 28).numpy(), cmap="gray")
plt.title("Original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(reconstruction.cpu().reshape(28, 28).numpy(), cmap="gray")
plt.title("Reconstruction")
plt.axis("off")

plt.show()

#ask whether to export the trained model
export_model = input("Export model? (yes/no): ").strip().lower()
if export_model in {"yes", "y"}:
    model_path = Path(__file__).parent / "autoencoder.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model exported to: {model_path}")
else:
    print("Model not exported")
