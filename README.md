#  Brain Tumor Classification using CNN

##  Project Overview
This project aims to classify brain tumor MRI images into four categories using Convolutional Neural Networks (CNN):

- Glioma  
- Meningioma  
- Pituitary tumor  
- No tumor  

The model is trained using the Adam optimizer with Cross-Entropy Loss for multi-class classification. It is further improved by adding an additional convolutional layer, along with Batch Normalization and Dropout to enhance generalization. The model is trained for up to 60 epochs with a batch size of 64, and early stopping is applied based on validation performance.

---

##  Dataset
- Source: Brain Tumor MRI Dataset ([Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset/data))  
- Original training set: 5,600 images (1,400 per class)  
- After splitting:
  - Training set: 4,480 images (1,120 per class)  
  - Validation set: 1,120 images (280 per class)  
- Test set: 1,600 images (400 per class)  


---

##  Data Preprocessing

- Resize images to **224 × 224**
- Convert to grayscale (1 channel)
- Normalize: mean = 0.5, std = 0.5
- Data augmentation (training only):
  - Random rotation ±10 degree


---

##  Baseline Model
A baseline CNN with 3 convolutional layers (1→16→32→64), each followed by ReLU and max pooling. The extracted features (64 × 28 × 28) were then flatten and passed through two fully connected layers with a ReLU activation for classification.

Compared to the baseline, the improved model introduces an additional convolutional layer, Batch Normalization, and Dropout, resulting in better generalization and performance.

---

##  Model Architecture (Improved CNN)

###  Feature Extraction
- Conv(1 → 16) + BatchNorm + ReLU + MaxPool  
- Conv(16 → 32) + BatchNorm + ReLU + MaxPool  
- Conv(32 → 64) + BatchNorm + ReLU + MaxPool  
- Conv(64 → 128) + BatchNorm + ReLU + MaxPool  

###  Classifier
- Flatten  
- Linear (128 × 14 × 14 → 64)  
- ReLU  
- Dropout (p = 0.3)  
- Linear (64 → 4)  

---

##  Training Setup
- Loss: CrossEntropyLoss  
- Optimizer: Adam  
- Batch size: 64  
- Learning rate: 0.0007
- Epochs: up to 60  
- Early stopping based on validation loss
    - patience = 8   


---

##  Output
- The model outputs logits for 4 classes  
- CrossEntropyLoss is applied, which implicitly includes a Softmax operation  
- 4 class probabilities are obtained 

---

##  Evaluation Metrics
- Accuracy    
- Precision  (per class)  
- Recall  (per class)  
- F1-score (per class)  
- Confusion Matrix  

---

##  Results Summary
- The model achieves strong performance on the validation and test sets, with a validation accuracy of 0.9554, and a corresponding training accuracy of 0.9712   
- No significant overfitting is observed, as the training and validation curves remain close throughout training.  
- Batch Normalization and Dropout effectively improve generalization and prevent the model from memorizing the training data.  
- The model achieves a macro F1-score of 0.89, indicating balanced performance across all classes.  

---

##  Visualization
- Training vs Validation Accuracy curves  
- Training vs Validation Loss curves
- Per Class Precision, Recall, F1-score histogram.   
- Confusion Matrix heatmap  

---

##  How to Run
- Open the notebook in Google Colab  
- Enable GPU acceleration (Runtime → Change runtime type → GPU)  
- Mount the dataset from Google Drive before training  
- Run all cells in `improved_model.ipynb` to train and evaluate the model