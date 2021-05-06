# Autoencoder: Facial recognition for Attendance System
Attendance system using face recognition is a procedure of recognizing people by using face biostatistics based on the high definition monitoring and other computer technologies.
Here I have used concepts of Autoencoders to produce feature representation of images to compare the reference image of a person present in the dataset with the selfie taken to identify his presence.

# Used scripts

1. extract.py: extract faces from each image
2. autoencoder.py: architecture for Convulational Autoencoder
3. embedding.py: creates embedding of image with the use of Convulational Autencoder
4. step.py: includes training and validation step for producing image embeddings
5. similarity.py: compares the similarity of embeddings of reference anf selfie images 
6. main.py: executes the above functions to output a 'Match' or 'No Match'

# Run
1. In main.py edit the PATH variable with the test set directory  
2. Install required python packages from requirements.txt
3. Execute each python script in separate IDLE shells  
