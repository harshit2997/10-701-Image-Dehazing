import math

import Models
import random
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

# Setup global parameters to adjust for training our model
iterations = 2
batch_size = 2
learning_rate = 0.0006
exponent = 5  # channel exponent to control network size
save_loss = False  # boolean indicating whether we save losses per epoch
model_path = ''  # file path to a pre-trained model (if it exists)
is_gpu = False


def setup_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def print_model(model):
    print("Printing out our current model information:")
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("Initialized Model with " + str(params) + " trainable params")
    print("The model architecture looks like this:")
    print(model)


def setup_datasets():
    # TODO: Currently using a dummy dataset, need to modify to use our dataset
    first_rand = torch.randn(5, 3, 572, 572)
    second_rand = torch.randn(5, 3, 572, 572)
    training_data = TensorDataset(first_rand, second_rand)
    t_loader = DataLoader(training_data, batch_size, True, drop_last=True)

    validation_data = TensorDataset(first_rand, second_rand)
    v_loader = DataLoader(validation_data, batch_size, False, drop_last=True)
    return t_loader, v_loader


# Setup our model and then print out associated relevant information
unet = Models.UNet(enc_chs=(3, 64, 128, 256), dec_chs=(256, 128, 64), retain_dim=True)
unet.apply(Models.weights_init)
# TODO: Update the loss function to be good for our problem (L1 Loss maybe)
loss_function = nn.MSELoss()
print_model(unet)
if is_gpu:
    unet.cuda()
    loss_function.cuda()

optimizer = optim.Adam(unet.parameters(), lr=learning_rate)

# Setup the datasets to use for testing + validation
train_loader, validation_loader = setup_datasets()
print("Using " + str(len(train_loader)) + " Training Batches")
print("Using " + str(len(validation_loader)) + " Validation Batches")

epochs = math.ceil(iterations/len(train_loader))

if len(model_path) > 0:
    unet.load_state_dict(torch.load(model_path))
    print("Loaded the model stored in " + model_path)

# TODO: Need to update this to work with the dimensions of our problem
if is_gpu:
    inputs = torch.autograd.Variable(torch.FloatTensor(batch_size, 1, 28, 28))
    targets = torch.autograd.Variable(torch.FloatTensor(batch_size, 10))
else:
    inputs, targets = None, None

# TODO: Update these to display loss however we want to display it :)
for epoch in range(epochs):
    for i, train_data in enumerate(train_loader, 0):
        inputs_cpu, targets_cpu = train_data
        if is_gpu:
            inputs_cpu = inputs_cpu.float().cuda()
            targets_cpu = targets_cpu.float().cuda()
            inputs.data.resize_as(inputs_cpu).copy(inputs_cpu)
            targets.data.resize_as(targets_cpu).copy(targets_cpu)
        else:
            inputs = inputs_cpu
            targets = targets_cpu

        unet.zero_grad()
        outputs = unet(inputs)
        loss = loss_function(outputs, targets)
        loss.backward()
        loss_val = 0
        optimizer.step()
        print(f'Epoch: {epoch}, Batch: {i}, Loss: {loss.item()}')

    unet.eval()
    for i, valid_data in enumerate(validation_loader, 0):
        inputs_cpu, targets_cpu = valid_data
        if is_gpu:
            inputs_cpu = inputs_cpu.float().cuda()
            targets_cpu = targets_cpu.float().cuda()
            inputs.data.resize_as(inputs_cpu).copy(inputs_cpu)
            targets.data.resize_as(targets_cpu).copy(targets_cpu)
        else:
            inputs = inputs_cpu
            targets = targets_cpu

        outputs = unet(inputs)
        loss = loss_function(outputs, targets)
        print(f'Batch: {i}, Loss: {loss.item()}')