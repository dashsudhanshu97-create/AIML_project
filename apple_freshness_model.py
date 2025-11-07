import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob

# ============================
# STEP 1: Folder Paths
# ============================
train_dir = 'fruit_freshness_dataset/train'

# ============================
# STEP 2: Data Preprocessing (only train folder)
# ============================
img_size = (128, 128)
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,  # 80% train, 20% validation
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='training'
)

val_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'
)

# ============================
# STEP 3: Build CNN Model
# ============================
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# ============================
# STEP 4: Train Model
# ============================
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10
)

# ============================
# STEP 5: Visualize Training
# ============================
plt.figure(figsize=(10,5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Model Accuracy')
plt.show()

plt.figure(figsize=(10,5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Model Loss')
plt.show()

# ============================
# STEP 6: Save Model
# ============================
model.save('apple_freshness_model.h5')
print("💾 Model saved as 'apple_freshness_model.h5'")

# ============================
# STEP 7: Predict Any Image
# ============================
def predict_image(img_path):
    model = load_model('apple_freshness_model.h5')  # load saved model
    img = image.load_img(img_path, target_size=(128,128))
    img_array = image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) / 255.0
    prediction = model.predict(img_array)

    if prediction[0][0] > 0.5:
        print(f"🍎 Prediction: Rotten ({prediction[0][0]:.2f})")
    else:
        print(f"🍏 Prediction: Fresh ({prediction[0][0]:.2f})")

# Example manual test:
# predict_image('fruit_freshness_dataset/test_images/apple1.jpg')

# ============================
# STEP 8: Test Multiple Images (optional)
# ============================
def test_multiple_images(folder_path):
    for img_path in glob.glob(os.path.join(folder_path, '*.jpg')):
        print(f"\n🖼️ Testing: {img_path}")
        predict_image(img_path)

# Example usage:
# test_multiple_images('fruit_freshness_dataset/test_images')

# ============================
# STEP 9: Capture Image from Webcam
# ============================
def capture_image(save_path='captured_apple.jpg'):
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Could not open camera.")
        return

    print("📸 Press SPACE to capture | ESC to exit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame.")
            break

        cv2.imshow("Apple Freshness Detector", frame)
        key = cv2.waitKey(1)

        if key % 256 == 27:  # ESC
            print("🚪 Exiting capture...")
            break
        elif key % 256 == 32:  # SPACE
            cv2.imwrite(save_path, frame)
            print(f"✅ Image saved as {save_path}")
            break

    cap.release()
    cv2.destroyAllWindows()

# Example usage:
capture_image('captured_apple.jpg')
predict_image('captured_apple.jpg')
