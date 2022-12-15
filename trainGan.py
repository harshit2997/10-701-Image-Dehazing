import ModelsV2
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
learning_rateG = 0.0006
learning_rateD = 0.000006
exponent = 5  # channel exponent to control network size of generator
starting_channels = 32 # first layer channels of discriminator
save_loss = False  # boolean indicating whether we save losses per epoch
model_pathG = '' # path to a pre-trained generator model (if it exists)
model_pathD = '' # path to a pre-trained discriminator model (if it exists)
is_gpu = True
log_every_batch = False
gen_model_name_base_to_save = 'P2P_gen_256sq_64_6e-4_256_ep'
discr_model_name_base_to_save = 'P2P_discr_256sq_64_6e-5_32_ep'

def setup_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

setup_random_seed(0)

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

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# Setup our model and then print out associated relevant information
unet = ModelsV2.UNet(channelExponent=exponent)
unet.apply(weights_init)
netD = ModelsV2.NetD(3,3,ch=starting_channels)
netD.apply(weights_init)
l1_loss_function = nn.L1Loss()
bce_loss_function = nn.BCELoss()

print_model(unet)
print_model(netD)

if is_gpu:
    unet.cuda()
    netD.cuda()
    l1_loss_function.cuda()
    bce_loss_function.cuda()

optimizerG = optim.Adam(unet.parameters(), lr=learning_rateG)
optimizerD = optim.Adam(netD.parameters(), lr=learning_rateD)

# Setup the datasets to use for testing + validation
train_loader, validation_loader = setup_datasets()
print("Using " + str(len(train_loader)) + " Training Batches")
print("Using " + str(len(validation_loader)) + " Validation Batches")

if len(model_pathG) > 0 and len(model_pathD) > 0:
    unet.load_state_dict(torch.load(model_pathG))
    print("Loaded the UNet model stored in " + model_pathG)
    netD.load_state_dict(torch.load(model_pathD))
    print("Loaded the discriminator model stored in " + model_pathD)    

if is_gpu:
    inputs = torch.autograd.Variable(torch.FloatTensor(batch_size, 3, 256, 256))
    targets = torch.autograd.Variable(torch.FloatTensor(batch_size, 3, 256, 256))
else:
    inputs, targets = None, None

train_lossesG = []
train_lossesD = []
val_losses = []

for epoch in range(epochs):
    loss_accumG = 0.0
    loss_accumD = 0.0
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

        netD.zero_grad()
        outputs_realD = netD(inputs, targets).squeeze()
        real_lossD = bce_loss_function(outputs_realD, torch.autograd.Variable(torch.ones(outputs_realD.size()).cuda()))
        outputsG = unet(inputs)
        outputs_fakeD = netD(inputs, outputsG).squeeze()
        fake_lossD = bce_loss_function(outputs_fakeD, torch.autograd.Variable(torch.zeros(outputs_fakeD.size()).cuda()))

        lossD = (real_lossD + fake_lossD) * 0.5
        lossD.backward()
        optimizerD.step()
        loss_accumD+=lossD.item()

        unet.zero_grad()
        outputsG = unet(inputs)
        outputs_fakeD = netD(inputs, outputsG).squeeze()
        l1lossG = l1_loss_function(outputsG, targets)
        lossG = l1lossG + bce_loss_function(outputs_fakeD, torch.autograd.Variable(torch.ones(outputs_fakeD.size()).cuda()))
        lossG.backward()
        optimizerG.step()

        loss_accumG+=lossG.item()

        if log_every_batch:
          print(f'Epoch: {epoch}, Batch: {i}, discriminator loss: {lossD.item()}')
          print(f'Epoch: {epoch}, Batch: {i}, L1 generator loss: {l1lossG.item()}')
          print(f'Epoch: {epoch}, Batch: {i}, total generator loss: {lossG.item()}')

    torch.save(copy.deepcopy(unet.state_dict()), "Models/" + gen_model_name_base_to_save + str(epoch) + ".pt")
    torch.save(copy.deepcopy(netD.state_dict()),"Models/" + discr_model_name_base_to_save + str(epoch) + ".pt")
    
    print ("Train generator loss at epoch "+str(epoch)+" : "+str(loss_accumG/float(len(train_loader))))
    train_lossesG.append(loss_accumG/float(len(train_loader)))
    print ("Train discriminator loss at epoch "+str(epoch)+" : "+str(loss_accumD/float(len(train_loader))))
    train_lossesD.append(loss_accumD/float(len(train_loader)))

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
        loss = l1_loss_function(outputs, targets)
        val_loss_accum+=loss.item()
        if log_every_batch:
            print(f'Batch: {i}, Loss: {loss.item()}')
    
    print ("Validation loss at epoch "+str(epoch)+" : "+str(val_loss_accum/float(len(validation_loader))))
    val_losses.append(val_loss_accum/float(len(validation_loader)))

print ("Training losses generator:")
print (train_lossesG)
print ("Training losses discriminator:")
print (train_lossesD)
print ("Validation losses:")
print (val_losses)