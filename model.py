import tensorflow


# Load the pre-trained VGG16 model without the top layer
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the layers of the base model
for layer in base_model.layers:
    layer.trainable = False

# Add new top layers for our classification task
x = Flatten()(base_model.output)
x = Dense(1024, activation='relu')(x)
predictions = Dense(7, activation='softmax')(x)  # Assuming 7 emotions in FER+

# Create the full model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Model summary
model.summary()
