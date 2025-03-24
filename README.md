## **PARKINSON’S DISEASE PREDICTION**

#### 🏥 **Early Detection of Parkinson’s Disease Using Deep Learning**  

This repository contains a **Gradio-based web application** that predicts **Parkinson’s Disease** using **deep learning (CNN model)**. The model processes **handwritten spiral drawings** to classify individuals as **Healthy (Class 0) or Parkinson’s (Class 1)**.

---

## 🚀 **Features**  
✅ **Uses a CNN-based deep learning model**  
✅ **Classifies images as either Healthy or Parkinson’s Disease**  
✅ **Simple Gradio web interface for easy image upload & prediction**  
✅ **Displays confidence percentage of the prediction**  
✅ **Runs on Google Colab with a shareable public link**  

---

## 🖥 **Installation & Setup**  

1️⃣ **Clone the Repository**  
```bash
git clone https://github.com/Tanisi1234/Parkinsons-Disease-Prediction.git
cd Parkinsons-Disease-Prediction
```

2️⃣ **Install Dependencies**  
```bash
pip install gradio tensorflow numpy pillow
```

3️⃣ **Run the Gradio App**  
```bash
python app.py
```

---

## 🏃 **Running in Google Colab**  

1️⃣ Upload your **trained model** (`your_model.h5`) to Colab  
2️⃣ Install Gradio using:  
   ```python
   !pip install gradio
   ```
3️⃣ Run the **Python script** and click the **Gradio link** to use the app  

---

## 🔬 **Model & Dataset**  
- **Model**: Convolutional Neural Network (CNN)  
- **Dataset**: Handwritten Spiral Drawings (used as input for classification)  

---

## 📜 **Code: Gradio App for Parkinson’s Disease Prediction**  

```python
import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model("your_model.h5")  # Replace with your actual model path

# Define class labels
class_labels = {0: "Healthy", 1: "Parkinson’s Disease"}

# Function to make predictions
def predict_disease(image):
    image = image.resize((224, 224))  # Resize for the model
    img_array = np.array(image) / 255.0  # Normalize
    img_array = img_array.reshape(1, 224, 224, 3)  # Reshape for model input
    
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)  # Get highest probability class
    confidence = np.max(prediction) * 100  # Confidence percentage
    
    return f"Prediction: {class_labels[class_index]} (Confidence: {confidence:.2f}%)"

# Create the Gradio UI
interface = gr.Interface(
    fn=predict_disease,
    inputs=gr.Image(type="pil"),
    outputs=gr.Textbox(),
    title="Parkinson’s Disease Detection",
    description="Upload a handwritten spiral drawing to check if it's Healthy or Parkinson’s Disease.",
)

# Launch the app
interface.launch(share=True)
```

---

## 📊 **Results**  

| Sample Input | Prediction | Confidence |
|-------------|------------|------------|
| Healthy Drawing | Healthy | 71.85% |
| Parkinson’s Drawing | Parkinson’s Disease | 97.60% |

---

## 📚 **Future Work**  
🔹 Expand dataset for **higher generalization**  
🔹 Implement **multi-class classification** for different **stages of Parkinson’s**  
🔹 Develop a **real-time mobile app** for faster detection  
🔹 Enhance model with **Hybrid Deep Learning techniques**  

---

## ✨ **Contributors**  
👩‍💻 **Tanisi Jha** - *AICTE Internship on AI, TechSaksham (Microsoft & SAP)*  

📌 **GitHub Repository**: [Tanisi1234/Parkinsons-Disease-Prediction](https://github.com/Tanisi1234/Parkinsons-Disease-Prediction)  

---

## 📝 **License**  
🔹 This project is **open-source** and free to use for research & development purposes.  

---

### 🚀 **Get Started Now & Detect Parkinson’s with AI!** 🏥💡  

---

This README follows the **# Parkinsons-Disease-Prediction** format and is ready to **upload to GitHub**! Let me know if you need modifications! 🚀
