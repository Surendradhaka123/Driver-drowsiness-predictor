import streamlit as st
import cv2
import time
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from pygame import mixer



from datetime import datetime
model = load_model('Drowsiness_model_efficient.h5')

html_temp= """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:centre;">Drowsiness Detection App </h2>
    </div>
    """
st.markdown(html_temp,unsafe_allow_html=True)

st.markdown(
   
    """
     This app is developed for drowsiness detection. This app will raise an alarm if the person is drowsy.
"""
)
Warning="By selecting the check box you are agree to use our app.\nDon't worry!! We will not save your any data."
check=st.checkbox("I agree",help=Warning)
if(check):
    st.write('Great!')
    btn=st.button("Start")
    st.write('Press (c) for ending the stream')
    if btn:
        
            #multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades

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
            
            st.text("Thanks for using")
if st.button("About"):
        st.text("Created by Surendra Kumar")
## footer
from htbuilder import HtmlElement, div, ul, li, br, hr, a, p, img, styles, classes, fonts
from htbuilder.units import percent, px
from htbuilder.funcs import rgba, rgb


def image(src_as_string, **style):
    return img(src=src_as_string, style=styles(**style))


def link(link, text, **style):
    return a(_href=link, _target="_blank", style=styles(**style))(text)


def layout(*args):
    style = """
    <style>
      # MainMenu {visibility: hidden;}
      footer {visibility: hidden;}
     .stApp { bottom: 105px; }
    </style>
    """

    style_div = styles(
        position="fixed",
        left=0,
        bottom=0,
        margin=px(0, 0, 0, 0),
        width=percent(100),
        color="black",
        text_align="center",
        height="auto",
        opacity=1
    )

    style_hr = styles(
        display="block",
        margin=px(8, 8, "auto", "auto"),
        border_style="solid",
        border_width=px(0.5)
    )

    body = p()
    foot = div(
        style=style_div
    )(
        hr(
            style=style_hr
        ),
        body
    )
    st.markdown(style,unsafe_allow_html=True)

    for arg in args:
        if isinstance(arg, str):
            body(arg)

        elif isinstance(arg, HtmlElement):
            body(arg)

    st.markdown(str(foot), unsafe_allow_html=True)


def footer():
    myargs = [
        "©️ surendraKumar",
        br(),
        link("https://www.linkedin.com/in/surendra-kumar-51802022b", image('https://icons.getbootstrap.com/assets/icons/linkedin.svg') ),
        br(),
        link("https://www.instagram.com/im_surendra_dhaka/",image('https://icons.getbootstrap.com/assets/icons/instagram.svg')),
    ]
    layout(*myargs)

if __name__ == "__main__":
    footer()