# Deep_final
This repository contain material for the final project in deep learning.

The usage of the different files/folders are:

GTApack: Is a package containing functions that is reused.
 - __init__ : Used to make this directory a python package.
 - GTA_antihot: Function used to turn an onehot encoded image into an image with
                3 channels.
 - GTA_hotloader: Dataloader function for normal input image and onehot
                  encoded targets.
 - GTA_imageplot: A function for plotting input/target(/output) pictures.
 - GTA_loader:  Dataloader function for normal input and target images.
 - GTA_prop_to_hot: Turns the output of the network (in probability format) into
                    onehot encoding.
 - GTA_tester: Function for finding the accuarcy of test or validation data.
 - GTA_Unet: The implementation of the Unet.

 Old functions or files: Retired files that is no longer in use.
  - GTA_Unetpadding: Experimentation with using padding on the network.



trainer: This file contain the 7 different runs of the network.

plotting-results: This file plots the input, target and prediction from a
                  network.
