import os,sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import numpy as np
from sklearn.model_selection import KFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
# from src.com_in_ineuron_ai_training.softmax import SoftMax

from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow import keras

def trainModel():
    # Load Embeddings
    data = pickle.loads(open('src/embeddings/embeddings.pickle',"rb").read())
    embeddings = np.array(data["embeddings"])
    # Encode the labels
    le = LabelEncoder()
    labels_le = le.fit_transform(data["names"])
    num_classes = len(np.unique(labels_le))
    labels_ler = labels_le.reshape(-1, 1)
    
    ct = ColumnTransformer([("labels", OneHotEncoder(), [0])], remainder = 'passthrough')
    labels = ct.fit_transform(labels_ler)
    # print(labels)

    # one_hot_encoder = OneHotEncoder(categorical_features = [0])
    # labels = one_hot_encoder.fit_transform(labels).toarray()

    # Initialize Softmax training model arguments
    BATCH_SIZE = 8
    EPOCHS = 5
    input_shape = embeddings.shape[1]
    # Softmax Model
    model = Sequential()
    model.add(Dense(1024, activation='relu', input_shape=(input_shape,)))
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss=keras.losses.categorical_crossentropy,optimizer=optimizer,metrics=['accuracy'])
    model.build()

    # Create KFold
    cv = KFold(n_splits = 5, random_state = 42, shuffle=True)

    # Train
    for train_idx, valid_idx in cv.split(embeddings):
        X_train, X_val, y_train, y_val = embeddings[train_idx], embeddings[valid_idx], labels[train_idx], labels[valid_idx]
        his = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1, validation_data=(X_val, y_val))

    model.save("src/embeddings/my_model.h5")
    with open('src/embeddings/labels.pickle', 'wb') as handle:
        pickle.dump(le, handle)
            
# trainModel()