# -*- coding: utf-8 -*-

# extract a single face from a given photograph
def extract_faces(filename, required_size=(150, 150)):
    # load image from file
    image = Image.open(filename)
    # convert to RGB
    image = image.convert('RGB')
    # convert to array
    pixels = np.asarray(image)
    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    results = detector.detect_faces(pixels)
    faces = []
    # extract the bounding box for every face 
    for result in results:
      x1, y1, width, height = result['box']
      x1, y1 = abs(x1), abs(y1)
      x2, y2 = x1 + width, y1 + height
      # face pixels
      face = pixels[y1:y2, x1:x2]
      # get image from pixels
      image = Image.fromarray(face)  
      image = image.resize(required_size)
      faces.append(image)

    return faces
