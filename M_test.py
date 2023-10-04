import snntorch as snn
from snntorch import surrogate
from snntorch import backprop
from snntorch import functional as SF
from snntorch import utils
from snntorch import spikeplot as splt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
import itertools

batch_size = 128
data_path='/data/mnist'

dtype = torch.float
device = torch.device("cuda:0") 
print(device)
transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))])

mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)

# Create DataLoaders
train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=True)

spike_grad = surrogate.fast_sigmoid(slope=25)
beta = 0.5
num_steps = 20

class LSTM(nn.Module):
    def __init__(self):
        super().__init__()

        num_inputs = 64
        num_hidden1 = 128

        spike_grad_lstm = snn.surrogate.straight_through_estimator()

        # initialize layers
        self.slstm1 = snn.SLSTM(num_inputs, num_hidden1,
        spike_grad=spike_grad_lstm)

    def forward(self, x):
        # Initialize hidden states and outputs at t=0
        syn1, mem1 = self.slstm1.init_slstm()

        # Record the final layer
        spk2_rec = []
        mem2_rec = []

        print("size : ",x.flatten(1).size())
        spk1, syn1, mem1 = self.slstm1(x.flatten(1), syn1, mem1)

        return torch.stack(spk), torch.stack(mem2_rec)

def forward_pass(net, num_steps, data):
  mem_rec = []
  spk_rec = []
  utils.reset(net)  # resets hidden states for all LIF neurons in net

#   for step in range(num_steps):
  spk_out, mem_out = net(data)
  spk_rec.append(spk_out)
  mem_rec.append(mem_out)
  return torch.stack(spk_rec).squeeze(), torch.stack(mem_rec)

def batch_accuracy(train_loader, net, num_steps):
  with torch.no_grad():
    total = 0
    acc = 0
    net.eval()
    
    train_loader = iter(train_loader)
    for data, targets in train_loader:
      data = data.to(device)
      targets = targets.to(device)
      spk_rec, _ = forward_pass(net, num_steps, data)

      acc += SF.accuracy_rate(spk_rec, targets) * spk_rec.size(1)
      total += spk_rec.size(1)

  return acc/total

loss_fn = SF.ce_rate_loss()
net = LSTM().to(device)
# data, targets = next(iter(train_loader))
# data = data.to(device)
# targets = targets.to(device)
# print(net)
# for step in range(num_steps):
#     spk_out, mem_out = net(data)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-2, betas=(0.9, 0.999))
num_epochs = 128
loss_hist = []
test_acc_hist = []
counter = 0

# Outer training loop
for epoch in range(num_epochs):

    # Training loop
    for data, targets in iter(train_loader):
        data = data.to(device)
        print("data : ",data.size())
        targets = targets.to(device)
        mem_rec = []
        spk_rec = []
        # forward pass
        net.train()
        spk_rec, _ = forward_pass(net, num_steps, data)
        
        # initialize the loss & sum over time
        loss_val = loss_fn(spk_rec, targets)
        # Gradient calculation + weight update
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        # Store loss history for future plotting
        loss_hist.append(loss_val.item())

        # Test set
        if counter % 50 == 0:
          with torch.no_grad():
              net.eval()

              # Test set forward pass
              test_acc = batch_accuracy(test_loader, net, num_steps)
              print(f"Iteration {counter}, Test Acc: {test_acc * 100:.2f}%\n")
              test_acc_hist.append(test_acc.item())

        counter += 1
