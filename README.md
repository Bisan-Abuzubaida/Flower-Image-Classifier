# Flower Image Classifier 🌸

A deep learning image classifier built with **Python and PyTorch**, capable of identifying different flower species. This project is the **final project of Udacity’s AI Programming with Python Nanodegree**, demonstrating skills in neural networks, computer vision, and Python programming.  

---

## 🚀 Project Overview

This project allows users to:

- Train a neural network to classify flower images using PyTorch.
- Predict the class of new flower images with a trained model.
- Map class indices to flower names using a JSON file (`cat_to_name.json`).

The workflow includes both **training** and **inference**, with modular Python scripts and an interactive Jupyter notebook for experimentation.

---

## 📂 File Structure

<pre>
  flower-image-classifier/
├── assets/ # Example images and visual assets
├── cat_to_name.json # Class index to flower name mapping
├── train.py # Script to train the model
├── predict.py # Script to predict a flower image
├── Image Classifier Project.ipynb # Notebook with full workflow
├── README.md # Project documentation
├── LICENSE # License file
└── CODEOWNERS # Optional project metadata
</pre>

---

## 🛠️ Technologies & Skills

- **Python**  
- **PyTorch** (Deep Learning, Neural Networks)  
- **NumPy, Pandas, Matplotlib**  
- **Image Classification & Computer Vision**  
- **Jupyter Notebook & Python Scripting**  
- **JSON data handling**  

---

## 📈 How It Works

1. **Training**
   - Load and preprocess images.
   - Define a neural network architecture.
   - Train on flower image dataset.
   - Save the trained model for predictions.

2. **Prediction**
   - Load the trained model.
   - Predict the class of a new flower image.
   - Map prediction index to the flower name using `cat_to_name.json`.

3. **Jupyter Notebook**
   - Step-by-step workflow.
   - Visualizations of training metrics and predictions.

---
🚀 Achievements

 - Built an image classifier for 102 flower species using PyTorch and transfer learning with VGG19.

 - Implemented data preprocessing & augmentation (random rotation, cropping, normalization) to improve robustness.

 - Designed and trained a custom feed-forward classifier with dropout and ReLU activations.

 - Optimized training using Adam optimizer and early validation checks.

 - Achieved ~72% accuracy on test data for unseen flower images.

 - Created a command-line interface (train.py and predict.py) to train new models and make predictions on any image input.
   
---

## 📸 Demo

<img width="593" height="632" alt="image" src="https://github.com/user-attachments/assets/65baaf59-2e16-4e58-805b-df33b8eecb5d" />

<img width="593" height="632" alt="image" src="https://github.com/user-attachments/assets/397dc720-22e1-4521-9ca7-611c57a59991" />

*The model correctly predicts flower types from images.*

---

## 💡 Project Highlights

- Final project of **Udacity AI Programming with Python Nanodegree**.  
- Demonstrates **deep learning, Python programming, and image classification skills**.  
- Includes **training scripts, prediction scripts, and notebook workflow** for reproducibility.  

---

## 📂 Usage

Clone the repo:

```bash
git clone https://github.com/Bisan-Abuzubaida/Flower-Image-Classifier.git
cd flower-image-classifier
```
Train the model:
```
python train.py
```

Predict a new image:
```
python predict.py --image path_to_image --checkpoint checkpoint.pth
```

Or explore the workflow interactively in the Jupyter notebook:
```
jupyter notebook "Image Classifier Project.ipynb"
```
## 📄 License

This project is licensed under the [Udacity License](https://github.com/Bisan-Abuzubaida/Flower-Image-Classifier/blob/main/LICENSE)

---

## 👩‍💻 Author

Bisan Abu Zubaida

AI Programming with Python Nanodegree – Udacity

Passionate about Deep Learning, Python, and AI projects

Portfolio: [GitHub](https://github.com/Bisan-Abuzubaida)

