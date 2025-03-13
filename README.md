Here is an updated **`README.md`** that matches your project structure and includes the required installation instructions:

---

# 🔥 AI Fire Detection - Machine Learning Demo

This project demonstrates how to use **MobileNetV2** for **fire detection** using **supervised learning** in **PyTorch**. Additionally, it compares **machine learning classification** with **LLM-based image analysis** using **LLaMA 3.2 Vision** via Ollama.

---

## 🚀 Getting Started

### 1️⃣ Install Dependencies

Ensure you have **Python 3.10+** installed, then install the required packages:

```bash
pip install -r requirements.txt
```

---

## 🖥️ CUDA Support

To enable **GPU acceleration**, install **CUDA & cuDNN**.

### 2️⃣ Check CUDA Version

To see if **CUDA is installed**, run:

```bash
nvcc --version
```

Your **CUDA version must be ≥ 11.8** to use modern PyTorch versions.

### 3️⃣ Install CUDA & cuDNN

#### **Windows:**
1. Download **CUDA Toolkit** from [NVIDIA](https://developer.nvidia.com/cuda-downloads).
2. Install **cuDNN** by following [NVIDIA cuDNN Guide](https://developer.nvidia.com/cudnn).
3. Verify installation by running:

```python
import torch
print(torch.cuda.is_available())  # Should return: True
print(torch.cuda.get_device_name(0))  # Should return GPU model
```

#### **Linux:**
Run:

```bash
sudo apt install nvidia-cuda-toolkit
```

Then restart your machine.

---

## 📂 Project Structure

```
📂 dataset/                    # Image dataset
    ├── fire_images/           # Fire images
    ├── non_fire_images/       # Non-fire images
TrainModel.ipynb               # Training MobileNetV2 model
ReTrainModel.ipynb             # Fine-tuning with misclassified images
TestModel.ipynb                # Evaluating on full dataset
Test-ReTrainedModel.ipynb      # Testing improved model
fire_classifier.pth            # Trained model
fire_classifier_retrained.pth  # Fine-tuned model
requirements.txt               # Python dependencies
README.md                      # This file
```

---

## 🔥 Running the Demo

### 4️⃣ Train the Fire Detection Model

To **train the model**, open `TrainModel.ipynb` and run all cells.

- Uses **MobileNetV2** as a lightweight **CNN model**.
- Trains on **fire vs. non-fire images** using a **small dataset**.
- Saves the trained model as `fire_classifier.pth`.

### 5️⃣ Fine-Tune with Misclassified Images

Run `ReTrainModel.ipynb` to **improve the model** by re-training with **misclassified images**.

### 6️⃣ Evaluate the Model

Run `TestModel.ipynb` to test **full dataset performance** and **generate confusion matrix**.

If you fine-tuned the model, use `Test-ReTrainedModel.ipynb` instead.

---

## 🧠 Using a Local LLM (LLaMA 3.2 Vision)

Instead of training a model, you can **use a local LLM** to describe images.

### 7️⃣ Install Ollama & LLaMA 3.2 Vision

Ollama is required to **run LLaMA locally**.

- Install Ollama:  
  [https://ollama.com/download](https://ollama.com/download)

- Download LLaMA 3.2 Vision:

  ```bash
  ollama pull llama3.2-vision
  ```

⚠️ **Minimum GPU Requirement:**  
LLaMA 3.2 Vision requires **at least 8GB VRAM** for inference.

### 8️⃣ Run Image Analysis with LLaMA 3.2 Vision

```python
import ollama

response = ollama.chat(
    model='llama3.2-vision',
    messages=[{
        'role': 'user',
        'content': 'What is in this image?',
        'images': ['test_image.jpg']
    }]
)

print(response['message']['content'])
```

---

## 🔄 **Comparison: Machine Learning vs. LLM for Fire Detection**

| Feature            | MobileNetV2 (Trained Model) | LLaMA 3.2 Vision (LLM) |
|--------------------|---------------------------|------------------------|
| **Training Required?** | ✅ Yes, trained on fire images | ❌ No, pre-trained on general data |
| **Purpose** | 🔥 Fire classification | 🖼️ Image description |
| **Accuracy** | ✅ High for classification | ⚠️ Can hallucinate (make up info) |
| **Local Execution?** | ✅ Works offline | ⚠️ Requires >8GB VRAM |
| **Customization?** | ✅ Can fine-tune | ❌ Limited flexibility |

---

## 📜 License

This project is released under **MIT License**.

---

### ✅ Next Steps:
- Let me know if you need **CUDA installation scripts** (`.bat` or `.sh`).
- The README now matches your **exact folder structure**.
- You can now **test both MobileNetV2 and LLaMA 3.2 Vision** and compare results. 🚀