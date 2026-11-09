# Blood Cells Classification Using Machine Learning and Deep Learning

Machine learning project for the classification of blood cells images from eight classes using classical machine learning and CNN.

The project covers the full cycle of work on machine learning problems: data exploration, data preprocessing and modeling.

## Overview

The **BloodMNIST** dataset from the MedMNIST collection is used for classification of microscopic images of blood cells.

There are two different approaches which are implemented and compared:

- **Random Forest** - classical machine learning approach based on pixel features of image
- **Convolutional Neural Network (CNN)** - deep learning approach implemented using PyTorch.

### Final Test Results

| Metric     | Random Forest | CNN         |
|------------|----------:|----------:|
| Accuracy   |   82.80%  | **92.60%**|
| Macro Precision |   83.40%  | **91.90%**|
| Macro Recall    |   78.40%  | **91.30%**|
| Macro F1-score  |   0.799   | **0.915** |
| Weighted F1-score |   0.824  | **0.926** |

### CNN performance by class

| Blood Cell Class | Precision | Recall | F1-score |
|------------------|----------:|-------:|---------:|
| Basophil         |   90.3%  |   84.0%  |   0.870  |
| Eosinophil       |   97.9%  |   98.2%  |   0.981  |
| Erythroblast     |   98.9%  |   88.7%  |   0.936  |
| Immature granulocytes |   83.2% |   85.3%  |   0.842  |
| Lymphocyte       |   87.9%  |   89.7%  |   0.888  |
| Monocyte         |   82.2%  |   87.7%  |   0.848  |
| Neutrophil       |   95.4%  |   96.4%  |   0.959  |
| Platelet         |   99.8%  |  100.0%  |   0.999  |

## Dataset

**BloodMNIST - MedMNIST v2**

- 17,092 original RGB images
- Image size: 28 × 28 pixels
- 8 class targets
- Train / validation / test original split: 11,959 / 1,712 / 3,421 images

Eight classes are:

1. Basophil
2. Eosinophil
3. Erythroblast
4. Immature granulocytes
5. Lymphocyte
6. Monocyte
7. Neutrophil
8. Platelet

As a result of data cleaning, the following dataset has been created:

| Split | Original images | Cleaned images |
|-------|---------------:|--------------:|
| Training |          11,959 |           11,943 |
| Validation |            1,712 |             1,711 |
| Test |               3,421 |              3,420 |
| **Total** | **17,092** | **17,074** |

Test split was separated from training and model selection processes.

### Random Forest Baseline

Baseline configuration:

```python
RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
```

CNN is implemented using PyTorch.

**Training Parameters:**

| Parameter | Value |
|-----------|------:|
| Framework | PyTorch |
| Loss function | CrossEntropyLoss |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Batch size | 128 |
| Max. Epochs in training loop | 15 |
| Checkpoint selection | Lowest validation loss |
| Hardware Acceleration | NVIDIA CUDA GPU |

The model with the lowest validation loss is selected for final evaluation.

Evaluation includes:

- Accuracy
- Precision
- Recall
- Macro F1-score
- Weighted F1-score
- Classification reports
- Confusion matrices
- Learning curves for training and validation stages

Macro-average metrics are especially important due to unequal number of samples per class.

## Project Structure

```text
blood-cell-classification/
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_cnn_training.ipynb
│   └── 05_final_evaluation.ipynb
│
├── src/
│   └── bloodcell/
│       ├── __init__.py
│       └── model.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── random_forest_baseline.joblib
│   └── bloodcell_cnn_best.pth
│
├── reports/
│   ├── final_model_comparison.csv
│   ├── final_confusion_matrices.png
│   ├── random_forest_confusion_matrix.csv
│   └── cnn_confusion_matrix.csv
│
├── tests/
├── .gitignore
├── requirements.txt
└── README.md
```

Execute the notebooks in numerical order:

1. `01_data_exploration.ipynb`
2. `02_data_preprocessing.ipynb`
3. `03_model_training.ipynb`
4. `04_cnn_training.ipynb`
5. `05_final_evaluation.ipynb`

## References

1. Acevedo, A. et al. (2020). *A dataset of microscopic peripheral blood cell images for development of automatic recognition systems.* Data in Brief, 30, 105474. [https://doi.org/10.1016/j.dib.2020.105474](https://doi.org/10.1016/j.dib.2020.105474)

2. Yang, J. et al. (2023). *MedMNIST v2 - A large-scale lightweight benchmark for 2D and 3D biomedical image classification.* Scientific Data, 10, 41. [https://doi.org/10.1038/s41597-022-01721-8](https://doi.org/10.1038/s41597-022-01721-8)

3. [MedMNIST official repository](https://github.com/MedMNIST/MedMNIST)

The BloodMNIST dataset is distributed under the Creative Commons Attribution 4.0 International (CC BY 4.0) license. Please refer to the original dataset and publication for attribution details.
