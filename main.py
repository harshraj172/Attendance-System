# -*- coding: utf-8 -*-
"""
Building an Attendance Notification System which compares a person's selfie image with his reference image to return a 'Match' or 'No Match'.
"""

import numpy as np
import os
import cv2
import imutils
import regex as re
import matplotlib.pyplot as plt
import pandas as pd
from mtcnn.mtcnn import MTCNN
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2 # opencv
from mtcnn.mtcnn import MTCNN #  Multi Cascade Convulational Network
from matplotlib import pyplot as plt
from PIL import Image
from google.colab.patches import cv2_imshow
import torch
import torchvision.transforms as T
# call the defined libraries
from extract import extract_faces
from autoencoder import ConvEncoder
from autoencoder import ConvDecoder
from step import train_step
from step import val_step
from embedding import create_embedding

"""
Prepare Data
"""
transform = T.Compose([T.ToTensor()]) # Normalize the pixels and convert to tensor.

data = [] # list to store tensors
PATH = '/content/trainset'

# looping for every image 
for subdir1 in os.listdir(PATH):
  for subdir2 in os.listdir(f"{PATH}/{subdir1}"):
    for subdir3 in os.listdir(f"{PATH}/{subdir1}/{subdir2}"):
      faces = extract_faces(f"{PATH}/{subdir1}/{subdir2}/{subdir3}") # list of face in every image
      for face in faces: 
        tensor_image = transform(face) # transforming image into a tensor
        data.append(tensor_image)

"""
Training Script
"""

full_dataset = data # Create dataset.

# Split data to train and test
train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [4002, 1334]) 

# Create the train dataloader
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
 
# Create the validation dataloader
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32)

# Create the full dataloader
full_loader = torch.utils.data.DataLoader(full_dataset, batch_size=32)

loss_fn = nn.MSELoss() # We use Mean squared loss which computes difference between two images.

encoder = ConvEncoder() # Our encoder model
decoder = ConvDecoder() # Our decoder model

device = "cuda"  # GPU device

# Shift models to GPU
encoder.to(device)
decoder.to(device)

# Both the enocder and decoder parameters
autoencoder_params = list(encoder.parameters()) + list(decoder.parameters())
optimizer = optim.Adam(autoencoder_params, lr=1e-3) # Adam Optimizer

# Time to Train !!!
EPOCHS = 10
# Usual Training Loop
for epoch in tqdm(range(EPOCHS)):
        train_loss = train_step(encoder, decoder, train_loader, loss_fn, optimizer, device=device)
        
        print(f"Epochs = {epoch}, Training Loss : {train_loss}")
        
        val_loss = val_step(encoder, decoder, val_loader, loss_fn, device=device)
        
        print(f"Epochs = {epoch}, Validation Loss : {val_loss}")

        # Simple Best Model saving
        if val_loss < max_loss:
            print("Validation Loss decreased, saving new best model")
            torch.save(encoder.state_dict(), "encoder_model.pt")
            torch.save(decoder.state_dict(), "decoder_model.pt")

# Save the feature representations.
EMBEDDING_SHAPE = (1, 256, 16, 16) # This we know from our encoder

# We need feature representations for complete dataset not just train and validation.
# Hence we use full loader here.
embedding = create_embedding(encoder, full_loader, EMBEDDING_SHAPE, device)

# Convert embedding to numpy and save them
numpy_embedding = embedding.cpu().detach().numpy()
num_images = numpy_embedding.shape[0]

# Save the embeddings for complete dataset, not just train
flattened_embedding = numpy_embedding.reshape((num_images, -1))
np.save("data_embedding.npy", flattened_embedding)

"""
Generate Output
"""

from scipy.spatial import distance

PATH = {'path to test set'}

for subdir1 in os.listdir(PATH):
  for subdir2 in os.listdir(f"{PATH}/{subdir1}"):
    ref, selfie = [], []
    for subdir3 in os.listdir(f"{PATH}/{subdir1}/{subdir2}"):
      
      # Pattern for an image file contains reference image
      pattern = [f"{subdir2}_script.jpg", 
                 f"{subdir2}_script_2.jpg",
                 f"{subdir2}_script_3.jpg",
                 f"{subdir2}_script_4.jpg"]
      
      faces = extract_faces(f"{PATH}/{subdir1}/{subdir2}/{subdir3}")
      
      # Check if the image file contains reference image
      disp = False
      for pat in pattern:
        if subdir3 == pat:
          disp = True
      
      # for reference image
      if disp == True: 
        # convert each face in the train set into embedding
        for face in faces:
          ref.append(face)
      # for selfie image
      else: 
        for face in faces:
          selfie.append(face)
    
    # Average of the similarity score of each selfie with every reference image from a person's directory  
    sum = 0 
    threshold = 0.70 # Determine whether the selfie matches the reference images
    for i in range(len(selfie)):
      for j in range(len(ref)):
        sum += similarity_score(selfie[i], ref[j])
      if sum/j > threshold:
        print('Matched')
      else:
        print("Not Matched")
