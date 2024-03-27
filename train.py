import tensorflow as tf
import os
import pandas as pd
from tensorflow.keras.utils import to_categorical
import argparse

def load_and_preprocess_data(csv_path, image_size=(64, 64), num_classes=8):
    df = pd.read_csv(csv_path)
    image_paths = df['image_path'].values
    labels = df['label'].values

    def _parse_function(filename, label):
        image_string = tf.io.read_file(filename)
        image_decoded = tf.image.decode_image(image_string, channels=1)
        image_resized = tf.image.resize(image_decoded, image_size)
        label_one_hot = to_categorical(label, num_classes=num_classes)
        return image_resized, label_one_hot

    filenames_tensor = tf.constant(image_paths)
    labels_tensor = tf.constant(labels)
    
    dataset = tf.data.Dataset.from_tensor_slices((filenames_tensor, labels_tensor))
    dataset = dataset.map(_parse_function)
    
    return dataset

# Assume the CSV file has columns 'image_path' and 'label' for image file paths and labels respectively

def build_and_compile_model(num_classes):
    model = build_vgg13_model(num_classes)  # Use the build_vgg13_model function from the previous reply
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',  # Using 'categorical_crossentropy' for multi-class classification
                  metrics=['accuracy'])
    return model

# Custom training step with Majority Voting
@tf.function
def train_step(images, labels, model, loss_fn, optimizer, train_loss, train_accuracy):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_fn(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss.update_state(loss)
    train_accuracy.update_state(labels, predictions)

# Custom validation step
@tf.function
def val_step(images, labels, model, loss_fn, val_loss, val_accuracy):
    predictions = model(images, training=False)
    v_loss = loss_fn(labels, predictions)

    val_loss.update_state(v_loss)
    val_accuracy.update_state(labels, predictions)

# Assume we have a function to display a summary of the data
def display_data_summary(train_dataset, val_dataset, test_dataset):
    pass

def train_and_evaluate(base_folder, training_mode='majority', num_classes=8, max_epochs=100):
    # Load and preprocess the data
    train_dataset = load_and_preprocess_data(os.path.join(base_folder, 'FER2013Train', 'label.csv'))
    val_dataset = load_and_preprocess_data(os.path.join(base_folder, 'FER2013Valid', 'label.csv'))
    test_dataset = load_and_preprocess_data(os.path.join(base_folder, 'FER2013Test', 'label.csv'))

    # Display data summary (implement this function based on your needs)
    display_data_summary(train_dataset, val_dataset, test_dataset)

    # Build and compile the model
    model = build_and_compile_model(num_classes)

    # Prepare for training
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam()

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
    val_loss = tf.keras.metrics.Mean(name='val_loss')
    val_accuracy = tf.keras.metrics.CategoricalAccuracy(name='val_accuracy')

    # Start training loop
    for epoch in range(max_epochs):
        # Training loop
        for images, labels in train_dataset:
            train_step(images, labels, model, loss_fn, optimizer, train_loss, train_accuracy)

        # Validation loop
        for images, labels in val_dataset:
            val_step(images, labels, model, loss_fn, val_loss, val_accuracy)

        # Logging and potentially early stopping based on validation accuracy
        # ...

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--base_folder', type=str, required=True,
                        help='Base folder containing the training, validation, and testing data.')
    parser.add_argument('--training_mode', type=str, default='majority',
                        help="Specify the training mode: majority")
    parser.add_argument('--epochs', type=int, default=100,
                        help="Maximum number of training epochs.")
    parser.add_argument('--num_classes', type=int, default=8,
                        help="Number of emotion classes.")

    args = parser.parse_args()
    train_and_evaluate(args.base_folder, args.training_mode, args.num_classes, args.epochs)

