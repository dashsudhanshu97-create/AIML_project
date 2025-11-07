# ============================
# 🍏 Apple Freshness Detection - Testing Script
# ============================

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
import os
import glob

# ============================
# STEP 1: Load the Trained Model
# ============================
model_path = 'apple_freshness_model.h5'

if not os.path.exists(model_path):
    print("❌ Model file not found! Train it first by running apple_freshness_model.py.")
    exit()

model = load_model(model_path)
print("✅ Model loaded successfully!")

# ============================
# STEP 2: Predict a Single Image
# ============================
def predict_image(img_path):
    """Predict whether an image is of a fresh or rotten apple."""
    img = image.load_img(img_path, target_size=(128,128))
    img_array = image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) / 255.0

    prediction = model.predict(img_array)
    if prediction[0][0] > 0.5:
        print(f"🍎 Prediction: Rotten ({prediction[0][0]:.2f})")
    else:
        print(f"🍏 Prediction: Fresh ({prediction[0][0]:.2f})")

# Example usage:
# predict_image('fruit_freshness_dataset/test_images/apple1.jpg')

# ============================
# STEP 3: Test Multiple Images from Folder
# ============================
def test_multiple_images(folder_path):
    """Loop through all images in a folder and predict each."""
    for img_path in glob.glob(os.path.join(folder_path, '*.jpg')):
        print(f"\n🖼️ Testing: {img_path}")
        predict_image(img_path)

# Example usage:
# test_multiple_images('fruit_freshness_dataset/test_images')

# ============================
# STEP 4: Capture Image via Webcam
# ============================
def capture_image(save_path='captured_apple.jpg'):
    """Capture an image using your webcam."""
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

        cv2.imshow("Apple Freshness Detector - Capture Mode", frame)
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

# ============================
# STEP 5: Capture + Predict
# ============================
def capture_and_predict():
    """Capture an apple image and immediately predict freshness."""
    save_path = 'captured_apple.jpg'
    capture_image(save_path)
    predict_image(save_path)

# ============================
# STEP 6: Menu System for Easy Testing
# ============================
def main():
    print("\n🍏 Apple Freshness Detection - Test Mode 🍎")
    print("1️⃣  Predict a single image file")
    print("2️⃣  Predict all images in a folder")
    print("3️⃣  Capture via webcam")
    print("4️⃣  Capture + Predict directly")
    choice = input("\nSelect an option (1/2/3/4): ").strip()

    if choice == '1':
        path = input("Enter image path: ").strip()
        predict_image(path)
    elif choice == '2':
        folder = input("Enter folder path: ").strip()
        test_multiple_images(folder)
    elif choice == '3':
        capture_image()
    elif choice == '4':
        capture_and_predict()
    else:
        print("❌ Invalid option. Exiting...")

if __name__ == "__main__":
    main()
