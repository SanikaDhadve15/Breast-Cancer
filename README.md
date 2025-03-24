Breast Cancer Prediction and Classification

Project Overview
Breast cancer is one of the most prevalent and fatal diseases among women worldwide. Early detection significantly improves survival rates. This project, CancerNet, leverages deep learning to classify breast cancer histology images as benign or malignant using Convolutional Neural Networks (CNNs). The model is trained on the **IDC dataset** and achieves high accuracy, demonstrating potential for clinical use.

Features
- Deep Learning Model: CNN-based classifier trained on histopathology images.
- Data Processing & Augmentation: Includes rescaling, normalization, and augmentation techniques.
- Performance Evaluation: Uses accuracy, precision, recall, AUC-ROC, and confusion matrix for analysis.
- Web-based Deployment: Implemented with **Streamlit** for real-time image classification.

Dataset
The project utilizes the Invasive Ductal Carcinoma (IDC) dataset, consisting of 277,524 image patches (50x50 pixels).
- 198,738 benign (non-cancerous) images
- 78,786 malignant (cancerous) images

 Methodology
1. Data Preprocessing
   - Images normalized and resized to 50x50 pixels
   - Data augmentation applied (rotation, zoom, shifting, flipping)
2. Model Training
   - CNN with multiple convolutional layers
   - Dropout and batch normalization to prevent overfitting
   - Class weighting to handle data imbalance
3. Evaluation Metrics
   - Confusion matrix, Accuracy, Precision, Recall, AUC-ROC curve
4. Deployment
   - **Streamlit-based Web App** for real-time image classification

Installation
  Prerequisites
- Python 3.x
- TensorFlow/Keras
- Streamlit
- OpenCV
- NumPy, Pandas, Matplotlib

Author
Sanika Dhadve

References
1. H. Mechria et al., "Breast Cancer Detection using Deep CNN," ICAART 2019.
2. M. Tiwari et al., "Breast Cancer Prediction using ML and DL Techniques," IEEE 2020.
3. P. Kumar et al., "A Review on Breast Cancer Detection Using Deep Learning," IOP 2021.

