import cv2 as cv
import tensorflow as tf
import numpy as np

#loading the pre-trained keras model
tf_model = tf.keras.models.load_model("/model/best_model.keras")
tf_model.summary()

# haar cascade used in facial detection
haar = cv.data.haarcascades
face_cascade = cv.CascadeClassifier(haar)

# function used to extract feature from an image
def feat_extract(image):
    feature = np.array(image)
    feature = feature.reshape(1,48,48,1)
    return feature / 255.0

# dictionary for emotions we are looking to capture 'neutral', 'happiness', 'surprise', 'sadness', 'anger'
emotions = {0:'Neutral', 1: "Happiness", 3: "Surprise", 4: "Sadness", 5: "Anger"}

# get real time video from webcam
feed = cv.VideoCapture(0)

while True:
    i, im = feed.read()
    gray_conv = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    detect = face_cascade.detectMultiScale(im, 1.3, 5)
    try:
        for(p,q,r,s) in detect:
            image = gray_conv[q:q + s, p:p + r]
            # Draw a rectangle around the detected face
            cv.rectangle(im, (p, q), (p + r, q + s), (255, 0, 0), 2)

            # Resize the face image to the required input size (48x48)
            image = cv.resize(image, (48, 48))

            # Extract features from the resized face image
            img = feat_extract(image)

            # Make a prediction using the trained model
            pred = tf_model.predict(img)

            # Get the predicted label for emotion
            prediction_label = emotions[pred.argmax()]

            # Display the predicted emotion label near the detected face
            cv.putText(im, f'Emotion: {prediction_label}', (p - 10, q - 10),
                        cv.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255))

        # Display the frame with annotations in real-time
        cv.imshow("Real-time Facial Emotion Recognition", im)

        # Break the loop if the 'Esc' key is pressed
        if cv.waitKey(1) == 27:
            break

    except cv.error:
        pass