import cv2
import tensorflow as tf
import numpy as np


model_path = "./model/best_model.keras"
tf_model = tf.keras.models.load_model(model_path)


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


emotions = {0:'Neutral', 1:'Happiness', 2:'Surprise', 3:'Sadness', 4:'Anger'}

# Prepare the webcam
feed = cv2.VideoCapture(0)

def preprocess_face(image):
    image = cv2.resize(image, (48, 48))
    image = np.array(image).reshape(1, 48, 48, 1)
    return image / 255.0

while True:
    ret, frame = feed.read()
    if not ret:
        break  

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]
        preprocessed_face = preprocess_face(face_img)

       
        prediction = tf_model.predict(preprocessed_face)
        emotion_label = emotions[np.argmax(prediction)]

        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    
    cv2.imshow('Real-time Facial Emotion Recognition', frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Exit with ESC key
        break


feed.release()
cv2.destroyAllWindows()
