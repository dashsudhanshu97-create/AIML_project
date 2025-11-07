🍏 Apple Freshness Detection (AI/ML Project)

A CNN-based model that classifies apples as Fresh 🍏 or Rotten 🍎 using TensorFlow and OpenCV.

⚙️ Setup & Run
🐧 Ubuntu

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 apple_freshness_model.py   # Train once
python3 test_model.py              # Test images / webcam

🪟 Windows
cd C:\path\to\fruit_freshness_project
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python apple_freshness_model.py    # Train once
python test_model.py               # Test images / webcam

🧾 requirements.txt
tensorflow
matplotlib
numpy
opencv-python
scipy

🧪 Testing Options

Run python3 test_model.py (or python test_model.py on Windows) and choose:

1️⃣  Predict single image
2️⃣  Predict all in folder
3️⃣  Capture via webcam
4️⃣  Capture + Predict directly
