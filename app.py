import sys
import os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import numpy as np
import tkinter as tk
from tkinter import *
from mtcnn.mtcnn import MTCNN
from imutils.video import FPS

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications import MobileNetV2,InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img,img_to_array

from src.insightface.src.common import face_preprocess
from src.genFaceEmbedings import genFaceEmbedings
from src.train import trainModel
from src.prediction import makePredictions


class faceRecogApp:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Face Recognition and Tracking")
        self.window.geometry("498x550")
        self.window.resizable(0, 0)

        header = tk.Label(self.window, text="Face Recognition System Phase-1", width=29, height=2, fg="#23006e", bg="#6ec9f5",
                          font=('times', 22, 'bold', 'underline'))
        header.place(x=0, y=0)
      
        label_col = "#e0f5ff"
        label_1 =Label(self.window,text="Name", width=15,bg=label_col,font=("bold",15))
        label_1.place(x=80,y=105)
        self.entry_1=Entry(self.window)
        self.entry_1.place(x=240,y=100,width=180,height=40)
        # self.entry_1.insert(END, 'sahil') #defalut value
        label_2 =Label(self.window,text="Dataset Size", width=15,bg=label_col,font=("bold",15))
        label_2.place(x=80,y=185)
        self.entry_2=Entry(self.window)
        self.entry_2.place(x=240,y=180,width=180,height=40)
        # self.entry_2.insert(END, '35') #defalut value

        # label_3 =Label(self.window,text="Nick Name", width=15,bg=label_col,font=("bold",15))
        # label_3.place(x=80,y=265)
        # self.entry_3=Entry(self.window)
        # self.entry_3.place(x=240,y=260,width=180,height=40)
        # self.entry_3.insert(END, 'ssc')  #defalut value
        self.but_0 = Button(self.window, text='Clear Data'      ,command=self.del_data    , width=20,fg="#23006e", bg="#6ec9f5")
        self.but_0.place(x=30,y=260,width=210,height=40)
        self.but_1 = Button(self.window, text='Capture Images'  ,command=self.captureImages    , width=20,fg="#23006e", bg="#6ec9f5")
        self.but_1.place(x=30,y=320,width=210,height=40)
        self.but_1 = Button(self.window, text='Train Model'     ,command=self.trainModel_2      , width=20,fg="#23006e", bg="#6ec9f5")
        self.but_1.place(x=260,y=260,width=210,height=40)
        self.but_1 = Button(self.window, text='Make Prediction' ,command=self.makePrediction_2  , width=20,fg="#23006e", bg="#6ec9f5")
        self.but_1.place(x=260,y=320,width=210,height=40)
        self.but_1 = Button(self.window, text='Exit'            ,command=self.exit          , width=20,fg="#23006e", bg="#d99191")
        self.but_1.place(x=145,y=380,width=210,height=40)

        label_4 = Label(self.window, text="Notification : ", width=13,height=2,bg="#ffc2c8",font=('bold', 15))
        label_4.place(x=20,y=460)

        self.message = tk.Label(self.window, text="Waiting for your command .... ", bg="#ffc2c8", fg="black", width=28, height=2, activebackground="#bbc7d4",font=('times', 15))
        self.message.place(x=160, y=461)

        Label(self.window, text="Made By :- Sahil S Chavan").place(x=350,y=525)
        self.window.mainloop()

    def get_data(self):
        self.name = self.entry_1.get() #https://stackoverflow.com/questions/14824163/how-to-get-the-input-from-the-tkinter-text-widget
        try:
            self.count = min(int(self.entry_2.get()),75)
        except:
            self.entry_2.delete(0, 'end')
            self.message.configure(text="Input an valid data")
            return False
        return True

    def del_data(self):
        self.entry_1.delete(0, 'end')
        self.entry_2.delete(0, 'end')
        self.message.configure(text="")

    def exit(self):
        pass
    def captureImages(self):
        fdetector = MTCNN()
        if not self.get_data(): return
        skip_frame = 1
        fps = FPS().start()

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open camera")
            exit()
        itern = 1
        cnt = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            # Checking and making the directory
            directory = 'datasets/train/'+self.name
            if not os.path.exists(directory):
                os.makedirs(directory)

            # face detection using MTCNN
            faces = fdetector.detect_faces(frame)
            # face in faces ->>> {'box': [268, 126, 162, 211], 'confidence': 0.982231855392456, 'keypoints': {'left_eye': (310, 206), 
            # 'right_eye': (387, 194), 'nose': (349, 228), 'mouth_left': (320, 287), 'mouth_right': (388, 278)}}
            for face in faces[:1]:
                # if face['confidence']>97:
                face = faces[0]
                x,y,w,h = face['box']
                # Phase 1
                # offest = 25
                # x,y,w,h = max(0,x-offest), max(0,y-offest),w+offest+15,h+offest+20
                # cv2.rectangle(frame, (x,y), (x+w,y+h), (255, 0, 0), 2)
                # Phase 2
                tempf = frame.copy()
                cv2.rectangle(frame,(x,y), (x+w,y+h), (255, 0, 0), 2)
                if itern%skip_frame==0:
                    # Phase 2
                    landmarks = face['keypoints']
                    landmarks = np.array([landmarks["left_eye"][0], landmarks["right_eye"][0], landmarks["nose"][0],
                                          landmarks["mouth_left"][0], landmarks["mouth_right"][0],
                                          landmarks["left_eye"][1], landmarks["right_eye"][1], landmarks["nose"][1],
                                          landmarks["mouth_left"][1], landmarks["mouth_right"][1]])
                    landmarks = landmarks.reshape((2, 5)).T
                    bbox = np.array([x,y,x+w,y+h])
                    crop_img = face_preprocess.preprocess(tempf,bbox, landmarks, image_size='112,112')
                    # Phase 1
                    # crop_img = frame[y:y+h, x:x+w]

                    path = directory+'/'+self.name+'_{}.jpg'.format(cnt)
                    print('--->>> Images saved at',path,crop_img.shape,' <<<----')
                    cv2.imwrite(path,crop_img)
                    cnt+=1
                itern+=1
          
            # Showing and saving the images
            cv2.imshow('image', frame)
            # if cnt%skip_frame==0:
            #     cv2.imwrite(directory+'/'+self.name+'_{}.jpg'.format(cnt),frame)
            fps.update()
            if cv2.waitKey(1) == ord('q') or cnt ==self.count:
                break
            
        fps.stop()
        print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
        
        cap.release()
        cv2.destroyAllWindows()
        notifctn = "Successfully captured {} images".format(self.count)
        self.message.configure(text=notifctn)


    def trainModel(self):
        train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

        train_generator = train_datagen.flow_from_directory(
        'datasets/train',
        target_size=(224,224),
        batch_size=32,
        class_mode='categorical')
        self.classes = train_generator.class_indices

        valid_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

        valid_generator = valid_datagen.flow_from_directory(
        'datasets/test',
        target_size=(224,224),
        batch_size=32,
        class_mode='categorical')

        base_model = InceptionV3(include_top=False,weights='imagenet',input_shape=(224,224,3),pooling='max')
        x = base_model.output
        # x = layers.Dense(512, activation='relu')(x)
        x = layers.Dense(128, activation='relu')(x)
        # x = layers.Flatten()(x)
        predictions = layers.Dense(len(self.classes), activation='softmax')(x)

        model = Model(inputs=base_model.input, outputs=predictions)

        for layer in base_model.layers:
            layer.trainable = False

        model.compile(optimizer='adam', loss='categorical_crossentropy')
        model.fit(train_generator,epochs=10,validation_data=valid_generator)
        self.prediction_model = model
        notifctn = "Model training is successful.No you can go for prediction."
        self.message.configure(text=notifctn)


    def makePrediction(self):
        notifctn = "Prediction Ongoing"
        self.message.configure(text=notifctn)
        fdetector = MTCNN()

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open camera")
            exit()

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            
            faces = fdetector.detect_faces(frame)
            for face in faces[:1]:
                face = faces[0]
                offest = 25
                x,y,w,h = face['box']
                x,y,w,h = max(0,x-offest), max(0,y-offest),w+offest+15,h+offest+20
                cv2.rectangle(frame, (x,y), (x+w,y+h), (255, 0, 0), 2)
                crop_img = frame[y:y+h, x:x+w]
                input_img = cv2.resize(crop_img,(224,224))
                input_image = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
                input_array = input_image/255
                input_array = np.expand_dims(input_image,axis=0)
                prediction = self.prediction_model.predict(input_array)
                tag = 'Unclassified'
                if max(prediction.tolist()[0]) > 0.5:
                    tag = list(self.classes.keys())[prediction.argmax()]
                    print('Prediction: ',tag,'-->',prediction)
                tagx = x+int(w/2)
                tagy = y
                cv2.putText(frame,tag,(tagx,tagy),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) == ord('q'):
                break
        self.message.configure(text='')
        cap.release()
        cv2.destroyAllWindows()
        
    def trainModel_2(self):
        # self.message.configure(text='Generating Embeddings')
        genFaceEmbedings()
        # self.message.configure(text='Training the model')
        trainModel()
        self.message.configure(text='Model training is successful.')

    def makePrediction_2(self):
        # self.message.configure(text='Prediction Ongoing')
        makePredictions()
        self.message.configure(text='Prediction Done') 


if __name__ == '__main__':
    appp = faceRecogApp()



