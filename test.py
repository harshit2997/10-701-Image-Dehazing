import ModelsV2
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from dataset import DehazeDataset, DehazeDatasetVal
from PIL import Image
import torchvision.transforms as T
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import time


# Setup global parameters to adjust for testing our model
batch_size = 1
exponent = 5  # channel exponent to control network size
model_path = ''  # file path to a pre-trained model (if it exists)
is_gpu = False
output_folder_name = ''

# Setup our model and then print out associated relevant information
unet = ModelsV2.UNet(channelExponent=exponent)

if len(model_path) > 0:
    if is_gpu:
        unet.load_state_dict(torch.load(model_path))
    else:
        unet.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    print("Loaded the model stored in " + model_path)
else:
    print ("No model to test on")

if is_gpu:
    unet.cuda()

# Setup the datasets to use for testing
test_dataset = DehazeDataset("./ResideSOTS/indoor/hazy", "./ResideSOTS/indoor/gt", 5, 0.)
test_loader = DataLoader(test_dataset, batch_size, False)
print("Using " + str(len(test_loader)) + " Testing Batches")

# Transform to resize the output to the original image size
transform = T.Resize((460,620))

if is_gpu:
    inputs = torch.autograd.Variable(torch.FloatTensor(batch_size, 3, 256, 256))
    targets = torch.autograd.Variable(torch.FloatTensor(batch_size, 3, 256, 256))
else:
    inputs, targets = None, None

unet.eval()

psnr_accum = 0.0
ssim_accum = 0.0
time_accum = 0.0

for i, test_data in enumerate(test_loader, 0):
    inputs_cpu, targets_cpu = test_data
    if is_gpu:
        inputs_cpu = inputs_cpu.float().cuda()
        targets_cpu = targets_cpu.float().cuda()
        inputs.data = torch.clone(inputs_cpu)
        targets.data = torch.clone(targets_cpu)
    else:
        inputs = inputs_cpu
        targets = targets_cpu

    start_time = time.time()
    outputs = unet(inputs)
    end_time = time.time()

    outputs_cpu = outputs.data.cpu().numpy()[0]
    targets_cpu = targets_cpu.cpu().numpy()[0]

    outputs_cpu = (outputs_cpu.transpose([1, 2, 0])*255.0).clip(0, 255).astype(np.uint8)
    im_output = transform(Image.fromarray(outputs_cpu))
    im_output.save("./ResideSOTS/" + output_folder_name + "/" + test_dataset.fileNames[i])

    target_np = np.asarray(Image.open("./ResideSOTS/indoor/gt/"+test_dataset.fileNames[i]))
    output_np = np.asarray(im_output)
   
    psnr = peak_signal_noise_ratio(target_np, output_np)
    ssim = structural_similarity(target_np, output_np, channel_axis=2)

    time_accum += float(end_time-start_time)
    psnr_accum += psnr
    ssim_accum += ssim

psnr_accum/=float(len(test_loader))
ssim_accum/=float(len(test_loader))
time_accum/=float(len(test_loader))

print ("PSNR: "+str(psnr_accum))
print("SSIM: "+str(ssim_accum))
print("Average evaluation time (in seconds): "+str(time_accum))