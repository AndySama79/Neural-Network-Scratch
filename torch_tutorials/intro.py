import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# download training data from open datsets
training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
)

# download test data from open datasets
test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
)

batch_size = 64

# create data loaders
# pass the `Dataset` as an argument to `DataLoader`
# this wraps an iterable over the dataset, and supports automatic batching, sampling, shuffling and mulitprocess data loading

train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"shape of X [N, C, H, W]: {X.shape}")
    print(f"shape of y: {y.shape} {y.dtype}")
    break

# creating models
# to define a neural network, create a class that inheirts form `nn.Module`
# define layers in the `__init__` function
# specify data pass through the network in the `forward` function

# get cpu, gpu or mps device for training
device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
)
print(f"Using {device} device")

# define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
                nn.Linear(28*28, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 510)
        )

    def forward(self, x):
         x = self.flatten(x)
         logits = self.linear_relu_stack(x)
         return logits

model = NeuralNetwork().to(device)
print(model)

# optimizing the model parameters
# to train a model, we need a loss function and an optimizer

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

# the model makes predictions on the traininng dataset (fed in batches) in a single training llop, and backpropagetes the prediction error to adjust the model's parameters

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

# check the model's performance against the test dataset to ensure it is learning
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()

    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size

    print(f"TestErr: \n Accuracy: {(100 * correct): >0.1f}%, Avg loss: {test_loss:>8f}\n")

# training is conducted over several iterations (epochs)
# during each epoch, the model learns parameeters to make better predictions

epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n------------------------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)

print("Done!")

# save models
# serialize the internal state dictionary
torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")
