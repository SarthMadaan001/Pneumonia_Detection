🫁 Pneumonia Detection using EfficientNet-B4

A deep learning–based medical imaging project that detects Pneumonia vs Normal from Chest X-ray images using EfficientNet-B4 with proper preprocessing and fine-tuning.

✅ Built and tested in Google Colab
✅ Uses multiple Kaggle datasets (merged)
✅ Uses EfficientNet-correct preprocessing
✅ Includes training, validation, testing & inference

📌 Project Overview

Pneumonia is a serious lung infection that can be detected from chest X-ray images.
This project uses transfer learning with EfficientNet-B4 to classify X-ray images into:

NORMAL

PNEUMONIA

The pipeline follows medical-AI best practices, avoiding common mistakes like preprocessing mismatch.

🧠 Model Used

EfficientNet-B4

Pretrained on ImageNet

Fine-tuned for chest X-ray classification

Binary classification with sigmoid output

📂 Datasets Used (Kaggle)

We used three Kaggle datasets, merged into a single unified dataset.
paultimothymooney/chest-xray-pneumonia
umitka/chest-x-ray-balanced
pcbreviglieri/pneumonia-xray-images

📁 Merged Dataset Structure (IMPORTANT)

After merging ALL datasets, the final structure is:
/content/pneumonia_dataset/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── val/
    ├── NORMAL/
    └── PNEUMONIA/
✔ All datasets are merged into train/
✔ A 20% validation split is created from training data
✔ Test/val folders from Kaggle are NOT used directly (avoids data leakage)

🏋️ Training Pipeline
Phase 1 – Feature Extraction

EfficientNet-B4 base frozen

Custom classification head trained

Phase 2 – Fine-Tuning

Top layers of EfficientNet unfrozen

Lower learning rate (1e-5)

Improves pneumonia sensitivity

Callbacks Used

EarlyStopping

ReduceLROnPlateau

📊 Expected Performance
Metric	Value
Accuracy	97%
AUC	> 0.97
Recall (Pneumonia)	High
False Positives	Reduced
False Negatives	Minimized

🧪 Testing on Custom Images

You can upload a chest X-ray from your PC and test it.

Decision logic:

< 0.4 → NORMAL

0.4 – 0.6 → UNCERTAIN

> 0.6 → PNEUMONIA

This reflects real medical decision-support systems.

▶️ How to Run (Colab)

Merge datasets into /content/pneumonia_dataset

Create validation split

Run training script

Save model to Google Drive

Test using uploaded X-ray images

🚀 Future Improvements

Grad-CAM explainability

ROC-based threshold tuning

Streamlit / Gradio web app

Multi-class pneumonia classification
