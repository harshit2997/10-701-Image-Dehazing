import math

import Models, ModelsV2
import random
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from dataset import DehazeDataset, DehazeDatasetVal
import copy

# Setup global parameters to adjust for training our model
epochs = 100
batch_size = 64
learning_rate = 0.00006
exponent = 3  # channel exponent to control network size
save_loss = False  # boolean indicating whether we save losses per epoch
model_path = ''
is_gpu = True


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
    training_data = DehazeDataset("./ResideITS/hazy", "./ResideITS/clear", 5, 0.2)
    t_loader = DataLoader(training_data, batch_size, True, drop_last=True)

    validation_data = DehazeDatasetVal(training_data)
    v_loader = DataLoader(validation_data, batch_size, False, drop_last=True)
    return t_loader, v_loader


# Setup our model and then print out associated relevant information
unet = ModelsV2.UNet()
unet.apply(ModelsV2.weights_init)
loss_function = nn.MSELoss()
#print_model(unet)
if is_gpu:
    unet.cuda()
    loss_function.cuda()

optimizer = optim.Adam(unet.parameters(), lr=learning_rate)

# Setup the datasets to use for testing + validation
train_loader, validation_loader = setup_datasets()
print("Using " + str(len(train_loader)) + " Training Batches")
print("Using " + str(len(validation_loader)) + " Validation Batches")

if len(model_path) > 0:
    unet.load_state_dict(torch.load(model_path))
    print("Loaded the model stored in " + model_path)

if is_gpu:
    inputs = torch.autograd.Variable(torch.FloatTensor(batch_size, 3, 256, 256))
    targets = torch.autograd.Variable(torch.FloatTensor(batch_size, 3, 256, 256))
else:
    inputs, targets = None, None

train_losses = []
val_losses = []

for epoch in range(epochs):
    train_loss_accum = 0.0
    for i, train_data in enumerate(train_loader, 0):
        inputs_cpu, targets_cpu = train_data
        if is_gpu:
            inputs_cpu = inputs_cpu.float().cuda()
            targets_cpu = targets_cpu.float().cuda()
            inputs.data = torch.clone(inputs_cpu)
            targets.data = torch.clone(targets_cpu)
        else:
            inputs = inputs_cpu
            targets = targets_cpu

        unet.zero_grad()
        outputs = unet(inputs)
        loss = loss_function(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss_accum+=loss.item()
        #print(f'Epoch: {epoch}, Batch: {i}, Loss: {loss.item()}')

    torch.save(copy.deepcopy(unet.state_dict()),"Models/UNet_256sq_64_6e-5_64_ep"+str(epoch)+".pt")
    print ("Train loss at epoch "+str(epoch)+" : "+str(train_loss_accum/float(len(train_loader))))
    train_losses.append(train_loss_accum/float(len(train_loader)))

    val_loss_accum = 0.0
    unet.eval()
    for i, valid_data in enumerate(validation_loader, 0):
        inputs_cpu, targets_cpu = valid_data
        if is_gpu:
            inputs_cpu = inputs_cpu.float().cuda()
            targets_cpu = targets_cpu.float().cuda()
            inputs.data = torch.clone(inputs_cpu)
            targets.data = torch.clone(targets_cpu)
        else:
            inputs = inputs_cpu
            targets = targets_cpu

        outputs = unet(inputs)
        loss = loss_function(outputs, targets)
        val_loss_accum+=loss.item()
        #print(f'Batch: {i}, Loss: {loss.item()}')
    
    print ("Validation loss at epoch "+str(epoch)+" : "+str(val_loss_accum/float(len(validation_loader))))
    val_losses.append(val_loss_accum/float(len(validation_loader)))

print ("Training losses:")
print (train_losses)
print ("Validation losses:")
print (val_losses)