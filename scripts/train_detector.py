import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import os

# 1. Path Setup
base_path = os.path.normpath("C:/Projects/data/processed_faces")

# 2. Data Loading (Batch size 4 works better for small forensic samples)
datagen = ImageDataGenerator(rescale=1./255)

train_data = datagen.flow_from_directory(
    base_path,
    target_size=(224, 224),
    batch_size=4,
    class_mode='binary'
)

# 3. Build the MobileNetV2 Model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x) # Sigmoid gives a 0 to 1 "probability"

model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the base so we don't ruin the pre-trained 'expert' knowledge
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 4. Train the Model
print("\n--- Starting Forensic AI Training ---")
model.fit(train_data, epochs=10) # 10 passes through the evidence

# 5. Save the Brain
os.makedirs('../models', exist_ok=True)
model.save('../models/deepfake_detector.h5')
print("\nSUCCESS: Model saved to ../models/deepfake_detector.h5")