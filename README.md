```markdown
# 🍏🌽 Plant Disease Classifier using CNN

This project uses a **Convolutional Neural Network (CNN)** built with **TensorFlow & Keras** to classify plant leaf diseases — specifically **Apple Disease** and **Corn Disease** — from image datasets.  

It includes complete code for:
- Dataset upload and preprocessing  
- Model training and validation  
- Model saving in both `.h5` and `.keras` formats  
- Prediction on new images  

---

## 🚀 Features

- ✅ Train on your own dataset (inside a ZIP file)
- ✅ Handles multiple plant diseases automatically
- ✅ Generates a deep learning model (`.h5` or `.keras`)
- ✅ Predicts disease type from uploaded leaf image
- ✅ Works fully in **Google Colab**

---

## 🧩 Project Structure

```

```
dick.zip
│
└── dick/
├── APPLE_DISEASE/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
│
└── CORN_DISEASE/
├── image1.jpg
├── image2.jpg
└── ...

```

Each subfolder represents a **disease class** and contains respective leaf images.

---

## ⚙️ Setup & Training (Google Colab)

1. **Open a new Colab notebook**
2. **Upload your dataset ZIP** (for example `dick.zip`)
3. Paste and run the following code:

```python
import os, zipfile
from google.colab import files
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Upload dataset
uploaded = files.upload()
zip_name = list(uploaded.keys())[0]

# Extract ZIP
extract_dir = "/content/dataset"
os.makedirs(extract_dir, exist_ok=True)
with zipfile.ZipFile(zip_name, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

base_dir = os.path.join(extract_dir, "dick")

# Image generators
train_datagen = ImageDataGenerator(rescale=1/255, validation_split=0.2,
                                   rotation_range=20, zoom_range=0.2, horizontal_flip=True)

train_gen = train_datagen.flow_from_directory(base_dir, target_size=(128,128),
                                              batch_size=32, class_mode='categorical', subset='training')
val_gen = train_datagen.flow_from_directory(base_dir, target_size=(128,128),
                                            batch_size=32, class_mode='categorical', subset='validation')

# Build CNN model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(2, activation='softmax')
])

# Compile
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
history = model.fit(train_gen, validation_data=val_gen, epochs=10)

# Save safely
model.save("/content/disease_model.keras", include_optimizer=False)
model.save("/content/disease_model.h5", include_optimizer=False)
print("✅ Model saved successfully!")
````

---

## 🔍 Predict Disease from Uploaded Image

After training, use the following snippet to upload any leaf image and predict:

```python
from google.colab import files
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

model_path = "/content/disease_model.h5"  # or .keras
model = tf.keras.models.load_model(model_path)
print("✅ Model loaded successfully!")

uploaded = files.upload()
img_path = list(uploaded.keys())[0]

img = image.load_img(img_path, target_size=(128,128))
img_array = np.expand_dims(image.img_to_array(img) / 255.0, axis=0)

pred = model.predict(img_array)
predicted_class = np.argmax(pred, axis=1)[0]

class_labels = ['APPLE_DISEASE', 'CORN_DISEASE']

plt.imshow(image.load_img(img_path))
plt.axis("off")
plt.title(f"Predicted: {class_labels[predicted_class]}")
plt.show()

print(f"🌱 Predicted disease: {class_labels[predicted_class]}")
print(f"Confidence: {pred[0][predicted_class]*100:.2f}%")
```

---

## 🧠 Future Enhancements

* Add more plant disease categories.
* Integrate Grad-CAM for visualizing affected leaf regions.
* Build a web interface using Flask or Streamlit for real-time prediction.
* Connect to a smartphone app for on-field disease detection.

---

## 🧪 Requirements

| Library      | Version (Recommended)   |
| ------------ | ----------------------- |
| TensorFlow   | 2.13+                   |
| NumPy        | 1.24+                   |
| Matplotlib   | 3.7+                    |
| Keras        | bundled with TensorFlow |
| Google Colab | Latest                  |

Install missing dependencies with:

```bash
pip install tensorflow numpy matplotlib
```

---

## 📂 Model Files

* `disease_model.h5` → legacy format (for backward compatibility)
* `disease_model.keras` → modern, stable format

Both contain the trained CNN for disease classification.

---

## 📜 License

This project is open-source and available under the **MIT License**.
You’re free to use, modify, and distribute with attribution.

---

## 👨‍💻 Author

**Devangshu Pandey**
*Aspiring Full-Stack Developer & ML Enthusiast*
📧 [Reach out via GitHub Issues or Discussions](https://github.com/pandeydevangshu12)

---

### 🌾 “Empowering farmers through AI-driven plant health detection.” 🌱

```

---

Would you like me to tailor this README further — for example, add **badges (e.g., TensorFlow, Python, License)** and a **preview section with example predictions** to make it look more professional on GitHub?
```
