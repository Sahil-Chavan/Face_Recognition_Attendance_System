import sys
import os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.insightface.deploy import face_model
from imutils import paths
import numpy as np
import pickle
import cv2
import os

def genFaceEmbedings():
    print('---->>> Creating data embeddings <<<---')
    embedding_model = face_model.FaceModel('112,112', "src/insightface/models/model-y1-test2/model,0", 1.24, 0)
    # Initialize our lists of extracted facial embeddings and corresponding people names
    knownEmbeddings = []
    knownNames = []
    # Initialize the total number of faces processed
    total = 0
    # Loop over the imagePaths
    imagePaths = list(paths.list_images('datasets\\train'))
    for (i, imagePath) in enumerate(imagePaths):
        # extract the person name from the image path
        print("---->>> processing image {}/{} <<<---".format(i + 1, len(imagePaths)))
        name = imagePath.split(os.path.sep)[-2]
            
        # load the image
        image = cv2.imread(imagePath)
        # convert face to RGB color
        nimg = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        nimg = np.transpose(nimg, (2, 0, 1))
        # Get the face embedding vector
        face_embedding = embedding_model.get_feature(nimg)
        # add the name of the person + corresponding face
        # embedding to their respective list
        knownNames.append(name)
        knownEmbeddings.append(face_embedding)
        total += 1
    data = {"embeddings": knownEmbeddings, "names": knownNames}
    with open('src/embeddings/embeddings.pickle', 'wb') as handle:
        pickle.dump(data, handle)


# genFaceEmbedings()