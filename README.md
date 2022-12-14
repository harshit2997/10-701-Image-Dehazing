# 10-701-Image-Dehazing
This repository contains all of the code that we used to train both the GAN and the U-Net for the problem of Image Dehazing

## Usage
First you need to download the ResideITS dataset. This dataset contains all of the images that were used for training
After this run the following commands depending on what you want to train. Make sure to adjust file paths so they work with the file system being used by your training method.
```
bash
# Setup the training data
!python '/SOTSIndoorClearMultiply.py'
# Run the training for our simple U-Net
!python '/train.py'
# Run the training for the GAN
!python 'trainGAN.py'
```
A trained model is saved after each epoch and you can modify where it is saved by modifying the file paths in train.py and trainGAN.py.

## Hyperparameters
For the best U-Net, set exponent=5 (256-dim bottleneck) and learnig rate = 6e-5.
For the GAN, set exponent=5 (256-dim bottleneck), starting_channels=32 (first layer channels of discriminator), learning rate of generator = 6e-4, learning rate of discriminator = 6e-6.

Set batch size=64 and epochs=100.

## Contributors
This code was created for a CMU 10-701 project by Karthik Natarajan and Harshit Mehrota. All code that was derived from papers is cited when used
