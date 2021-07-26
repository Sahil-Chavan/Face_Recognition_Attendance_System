# Face_Recognition_Attendance_System

  This is an Attendance system baesd on Face recognition tech. The application provides an interface to first register you face into the system with registration id and your name.
Then an option is present to train the model with the new data (including the old data). The training of the model happens real-time and then an option to start prediction is present. On selecting an prediciton window opens which recognizes different people based on the dataset created.
  
I completed this project in two phases :
### Phase 1 :  
I searched for an face detection system, I first tried the ever famous Haar cascades using opencv, but found out that it barely works on edge cases like when face is tilted, etc. So after some internet scrapping I came across MTCNN. Which was soo much better that haar cascades, due CNN’s ability to learn features from an input image, as opposed to the hardcoded kernels like haar cascades. But it came at the cost of speed.

  Using MTCNN detector, I was able to detect faces from images and also realtime througn my local machine's camera. Then I cropped those detected boxes saved it as training data for different persons and did an Transfer Learning training on Resnet 50, through applications in Keras.
My model was able to recognize people but with very less accuracy, it misclassified most of the times.
Due to failure of this model it tried finding the cause, and came to know that Face Recognition is an very different task all together than image classification. So I rectifed all the problems in my project's next phase.

### Phase 2 :
Here I have overcome the problem of Face Recognition by removing the image classifier and replacing it with the following:
I did my research and found out about Facenet, Arcface, Circle loss etc. Since being an beginner in deep learining I didn't went into depth of these papers but after comparing some factors and ease of implementation I went with Deepsight's Insightface implementation of Arcface (https://github.com/deepinsight/insightface). 
  
  Using it I was able to obtain 128 facial embeddings for each face after preprocessing, and feed these embeddings as an input to an small neural network, with output layer having softmax activation function to classify between different people. Train it with 0.2 validation split and 10 epochs. 

  During prediction, I obtained the similar facial embeddings for input stream frames, and passed it to the trained NN get the closest cluster of images and then applied Cosine distance as an similarity measure, to obtain an similarity score of the input image with the rest of the cluster. Kept an threshold for classification confidence from NN and for cosine distance and finally present the results.

<b>Dataset :</b> Is created in real-time on the app.

<b>Model :</b> MTCNN, Insightface (for obtaining embeddings), 3 layered sequential Neural Network (trained real-time on dataset).

<b>End point: </b>  One can make use of the project through the PC application I have created using Tkinter.

<b>End Result :</b> An system which takes real-time footage of an person, extracts face data, trains the model in real-time and is capable of facially recognizing different people with great accuracy in real-time. 

<b>Future Plans :</b> While doing the real-time prediction irrespective of previous prediction my app makes predictions for each frame, hence consuming an lot of computation power. So I am going to introduce object tracking in next phase of the project.

<b>Libraries Used :</b> Tensorflow 2.X, Keras, Sklearn, Numpy, MTCNN, Insightface, OpenCV, Imutils, Tkinter, etc and Mxnet and Pytorch as dependencies.

<b>Shout Out to all these references :</b>
- https://paperswithcode.com/task/face-recognition
- https://learnopencv.com/face-recognition-with-arcface
- https://github.com/deepinsight/insightface
- https://medium.com/@ageitgey/machine-learning-is-fun-part-4-modern-face-recognition-with-deep-learning-c3cffc121d78
- https://www.youtube.com/watch?v=uwJltCOrpEI
