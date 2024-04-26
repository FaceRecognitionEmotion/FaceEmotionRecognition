import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd
from tensorflow.keras.utils import to_categorical


import sys
import os

# Update the path if necessary
sys.path.append('C:/Users/jakea/OneDrive/Desktop/FaceEmotionRecognition')

# Now import your function
from train import load_and_preprocess_data

model = tf.keras.models.load_model('./model/best_model.keras')
base_folder = "./data"

# Load your test dataset
# Make sure test_dataset is not limited to a single batch; it should cover all test data.
test_image_folder = os.path.join(base_folder, 'FER2013Test')
test_csv_path = os.path.join(test_image_folder, 'label.csv')
test_dataset = load_and_preprocess_data(test_csv_path, test_image_folder, image_size=(48, 48), num_classes=8)

# Initialize lists to store the true labels and predictions
all_labels = []
all_predictions = []

# Iterate over the entire test dataset
for images, labels in test_dataset:
    preds = model.predict(images)
    predicted_classes = np.argmax(preds, axis=1)
    true_classes = np.argmax(labels.numpy(), axis=1)
    
    all_labels.extend(true_classes)
    all_predictions.extend(predicted_classes)

# Compute the confusion matrix
cm = confusion_matrix(all_labels, all_predictions)

# Function to plot confusion matrix
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], '.2f' if normalize else 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Plotting the confusion matrix
class_names = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt']
plot_confusion_matrix(cm, classes=class_names)

plt.show()
