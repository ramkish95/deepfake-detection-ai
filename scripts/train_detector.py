import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import os

# 1. PATH SETUP
# We use absolute paths to avoid any "File Not Found" errors on Windows
base_path = "C:/Projects/deepfake-detection-ai/data/processed_faces"

# 2. DATA PREPARATION (With Augmentation for better Forensic accuracy)
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    base_path,
    target_size=(224, 224),
    batch_size=16,
    class_mode='binary',
    subset='training'
)

val_data = datagen.flow_from_directory(
    base_path,
    target_size=(224, 224),
    batch_size=16,
    class_mode='binary',
    subset='validation'
)

# 3. BUILD THE MODEL (MobileNetV2)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Freeze base layers to keep pre-trained weights
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 4. TRAINING
print("\n--- Starting Training: Deepfake Detector ---")
model.fit(train_data, validation_data=val_data, epochs=6)

# 5. SAVE THE MODEL WITH ABSOLUTE PATH
# This ensures the website (app.py) can find the file easily
final_model_dir = "C:/Projects/deepfake-detection-ai/models"
final_model_path = os.path.join(final_model_dir, "deepfake_detector.h5")

# Create the folder if it doesn't exist
if not os.path.exists(final_model_dir):
    os.makedirs(final_model_dir)

# Save the model
model.save(final_model_path)
print(f"\n[SUCCESS] Model saved at: {final_model_path}")

# Verify file existence in the terminal
if os.path.exists(final_model_path):
    print("Verification: File confirmed in directory.")
else:
    print("Verification FAILED: File still not found.")