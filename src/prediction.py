import sys,os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pickle
import cv2
from mtcnn import MTCNN

from tensorflow.keras.models import load_model
from src.insightface.deploy import face_model
from src.insightface.src.common import face_preprocess


def findCosineDistance(vector1, vector2):
        vec1 = vector1.flatten()
        vec2 = vector2.flatten()

        a = np.dot(vec1.T, vec2)
        b = np.dot(vec1.T, vec1)
        c = np.dot(vec2.T, vec2)
        return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

def CosineSimilarity(test_vec, source_vecs):
    cos_dist = 0
    for source_vec in source_vecs:
        cos_dist += findCosineDistance(test_vec, source_vec)
    return cos_dist / len(source_vecs)


def makePredictions():

    # Variables Initialization
    image_size = '112,112'
    model = "src/insightface/models/model-y1-test2/model,0"
    threshold = 1.24
    det = 0
        
    # # Initialize detector
    detector = MTCNN()
        
    # Initialize faces embedding model
    embedding_model = face_model.FaceModel(image_size, model, threshold, det)
      
    embeddings = "src/embeddings/embeddings.pickle"
    le = "src/embeddings/labels.pickle"
        
    # Load embeddings and labels
    data = pickle.loads(open(embeddings, "rb").read())
    embeddings = np.array(data['embeddings'])
    le = pickle.loads(open(le, "rb").read())
    labels = le.fit_transform(data['names'])
       
    # Load the classifier model
    model = load_model('src/embeddings/my_model.h5')
        
    cosine_threshold = 0.8
    proba_threshold = 0.85
    comparing_num = 5
    texts = []
    frames = 0

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        frames += 1
        rgb_input = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        bboxes =  detector.detect_faces(frame)

        if len(bboxes) != 0:
            for bboxe in bboxes:
                bbox = bboxe['box']
                bbox = np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
                landmarks = bboxe['keypoints']
                landmarks = np.array([landmarks["left_eye"][0], landmarks["right_eye"][0], landmarks["nose"][0],
                                    landmarks["mouth_left"][0], landmarks["mouth_right"][0],
                                    landmarks["left_eye"][1], landmarks["right_eye"][1], landmarks["nose"][1],
                                    landmarks["mouth_left"][1], landmarks["mouth_right"][1]])
                landmarks = landmarks.reshape((2, 5)).T
                nimg = face_preprocess.preprocess(frame, bbox, landmarks, image_size='112,112')
                nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
                nimg = np.transpose(nimg, (2, 0, 1))
                embedding = embedding_model.get_feature(nimg).reshape(1, -1)
                
                text = "Unknown"
                
                # Predict class
                preds =  model.predict(embedding)
                preds = preds.flatten()
                # Get the highest accuracy embedded vector
                j = np.argmax(preds)
                proba = preds[j]
                # Compare this vector to source class vectors to verify it is actual belong to this class
                match_class_idx = (labels == j)
                match_class_idx = np.where(match_class_idx)[0]
                selected_idx = np.random.choice(match_class_idx, comparing_num)
                compare_embeddings = embeddings[selected_idx]
                # Calculate cosine similarity
                cos_similarity =  CosineSimilarity(embedding, compare_embeddings)
                if cos_similarity < cosine_threshold and proba > proba_threshold:
                    name =  le.classes_[j]
                    text = "{}".format(name)
                    print("Recognized: {} <{:.2f}>".format(name, proba * 100))
                 
            y = bbox[1] - 10 if bbox[1] - 10 > 10 else bbox[1] + 10
            cv2.putText(frame, text, (bbox[0], y), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (179, 0, 149), 4)

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


# makePredictions()