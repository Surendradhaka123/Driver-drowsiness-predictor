# Driver Drowsiness Detection using Transfer Learning with EfficientNetB5

![Drowsiness Detection](https://example.com/path/to/drowsiness-detection-image.png)

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- 
## Introduction

Driver drowsiness detection using transfer learning with EfficientNetB5 is a deep learning-based application that aims to detect the drowsiness level of drivers from real-time video streams. The model utilizes transfer learning with the EfficientNetB5 architecture, pre-trained on a large image dataset, to achieve high accuracy in real-world scenarios.

This README provides an overview of the project, instructions for setting up the environment, guidelines for model training, and information on how to contribute.

## Dataset

The dataset used for training and evaluation can be found at [https://www.kaggle.com/datasets/prasadvpatil/mrl-dataset]

This dataset have two directories one contains 2000 photos with open eyes and another contains 2000 photos with closed eyes.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/Driver-Drowsiness-Detection.git
cd Driver-Drowsiness-Detection
```

2. Set up a virtual environment (optional but recommended):

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

To use the trained drowsiness detection model on real-time video streams, follow these steps:

1. Ensure you have installed the required dependencies as mentioned in the [Installation](#installation) section.

2. Load the pre-trained EfficientNetB5 model:

```python

# Load the pre-trained EfficientNetB5 model
from tensorflow.keras.models import load_model
model =  load_model('Driver_drowsiness_efficientnet.h5)

```

3. Use the model for drowsiness detection on video frames captured from a camera:

```python
import cv2

# Assuming 'camera' is the camera object capturing video frames
multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades

            #https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

            #https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
            eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            mixer.init()
            sound= mixer.Sound(r'mixkit-digital-clock-digital-alarm-buzzer-992.wav')
            cap = cv2.VideoCapture(0)
            Score = 0
            openScore = 0
            while 1:

                ret, img = cap.read()
                height,width = img.shape[0:2]
                frame = img
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor= 1.3, minNeighbors=2)

                for (x,y,w,h) in faces:
                    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                    roi_gray = gray[y:y+h, x:x+w]
                    roi_color = img[y:y+h, x:x+w]
                    eye= img[y:y+h,x:x+w]
                    eye= cv2.resize(eye, (256 ,256))
                    im = tf.constant(eye, dtype = tf.float32)
                    img_array = tf.expand_dims(im, axis = 0)
                    prediction = model.predict(img_array)
                    print(np.argmax(prediction[0]))

                    # if eyes are closed
                    if np.argmax(prediction[0])<0.50:
                        cv2.putText(frame,'closed',(10,height-20),fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,fontScale=1,color=(255,255,255),
                                   thickness=1,lineType=cv2.LINE_AA)
                        cv2.putText(frame,'Score'+str(Score),(100,height-20),fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,fontScale=1,color=(255,255,255),
                                   thickness=1,lineType=cv2.LINE_AA)
                        Score=Score+1
                        if(Score>25):
                            try:
                                sound.play()

                            except:
                                pass

                    # if eyes are open
                    elif np.argmax(prediction[0])>0.60:
                        cv2.putText(frame,'open',(10,height-20),fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,fontScale=1,color=(255,255,255),
                                   thickness=1,lineType=cv2.LINE_AA)      
                        cv2.putText(frame,'Score'+str(Score),(100,height-20),fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,fontScale=1,color=(255,255,255),
                                   thickness=1,lineType=cv2.LINE_AA)
                        Score = Score-1
                        openScore = openScore +1
                        if (Score<0 or openScore >8):
                            Score=0


                cv2.imshow('frame',img)

                if cv2.waitKey(33) & 0xFF==ord('c'):
                    break
            cap.release()
            cv2.destroyAllWindows()
```

## Model Training
- For training the model on your own dataset you can follow the `Driver_drowsiness_detection.ipynb` file.

## Evaluation

It is essential to evaluate the trained model's performance on a separate test dataset to assess its accuracy and generalization.

Follow the `Driver_drowsiness_detection.ipynb` file for evaluation.

---

We hope this README helps you understand the Driver Drowsiness Detection using Transfer Learning with EfficientNetB5 project. If you have any questions or need further assistance, please don't hesitate to reach out.

Thank you for using our project!
