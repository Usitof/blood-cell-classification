# Blood Cells Classification Using Machine Learning and Deep Learning

Machine learning project for the classification of blood cells images from eight classes using classical machine learning and CNN.

The project covers the full cycle of work on machine learning problems: data exploration, data preprocessing and modeling.

## Overview

The **BloodMNIST** dataset from the MedMNIST collection is used for classification of microscopic images of blood cells.

There are two different approaches which are implemented and compared:

- **Random Forest** - classical machine learning approach based on pixel features of image
- **Convolutional Neural Network (CNN)** - deep learning approach implemented using PyTorch.

The aim of the project is to investigate whether the CNN can learn spatial features of images better than a classical machine learning algorithm.

### Final Test Results

| Metric     | Random Forest | CNN         |
|------------|----------:|----------:|
| Accuracy   |   82.80%  | **92.60%**|
| Macro Precision |   83.40%  | **91.90%**|
| Macro Recall    |   78.40%  | **91.30%**|
| Macro F1-score  |   0.799   | **0.915** |
| Weighted F1-score |   0.824  | **0.926** |

The CNN outperforms the Random Forest in all metrics mentioned above with improvement of test accuracy by 9.8 percentage points.

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

The largest improvements over the Random Forest were reached for the classes Basophil and Monocyte, which are the most difficult ones to classify using the classical approach.

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

### Data Cleaning

In addition to initial data exploration, an additional data cleaning was performed.

- Exactly duplicated images were found based on SHA-256 hash values.
- Duplicated images within one split were removed.
- Duplicates between training, validation and test splits were detected in order to prevent data leakage.
- There was an exactly duplicated image pair with the conflicting labels, so it was eliminated.
- In case when there were duplicated images between the splits, preference was given to the test split, followed by validation and then training split.

As a result of data cleaning, the following dataset has been created:

| Split | Original images | Cleaned images |
|-------|---------------:|--------------:|
| Training |          11,959 |           11,943 |
| Validation |            1,712 |             1,711 |
| Test |               3,421 |              3,420 |
| **Total** | **17,092** | **17,074** |

Test split was separated from training and model selection processes.

## Methodology

### 1. Exploratory Data Analysis

First of all, initial data exploration includes:

- Image size and data types
- Classes distribution and classes imbalance
- Visualization of example images
- Distribution of pixel intensities in RGB channels
- Channel-wise mean and std of pixels
- Exactly duplicated images detection
- Detection of duplicated images and inconsistent labels between splits

### 2. Data Preprocessing

Preprocessing steps include:

- Conversion of pixel values from range 0-255 to 0-1
- RGB channel-wise normalization using statistics collected from training set only
- Applying the same normalization parameters for validation and test splits
- Flattening images for classical machine learning
- Transformation to channel-first tensors (NCHW) for PyTorch

For both methods, two kinds of inputs are prepared:

- Random Forest: 2,352 flattened pixels per image
- CNN: 3-channel image tensors with size 3 × 28 × 28

### 3. Random Forest Baseline

For the classical machine learning baseline, Random Forest classifier from scikit-learn library is used.

Baseline configuration:

```python
RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
```

Additional research included:

- Class weight balancing (`class_weight="balanced"`).
- Number of candidate features for each tree (`max_features`).
- Tree regularization (`min_samples_leaf`).

The original baseline demonstrated the best performance on validation set among other Random Forest configurations.

### 4. Convolutional Neural Network

CNN is implemented using PyTorch.

**Network Architecture:**

```text
Input: 3 × 28 × 28
       |
Conv2D (3 → 32, kernel size 3)
       |
ReLU
       |
MaxPooling (2 × 2)
       |
Conv2D (32 → 64, kernel size 3)
       |
ReLU
       |
MaxPooling (2 × 2)
       |
Flatten (64 × 7 × 7)
       |
Fully Connected (3136 → 128)
       |
ReLU
       |
Dropout (0.3)
       |
Fully Connected (128 → 8)
       |
Output: 8 class logits
```

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

### 5. Model Evaluation

For both models, test split is used independently.

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

Large dataset files and generated model artifacts are kept out of version control and can be regenerated by running the corresponding notebooks.

## Installation and Reproducibility

The project was developed and tested using Python 3.12 and PyTorch with CUDA support.

### 1. Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd blood-cell-classification
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate the environment before installing dependencies.

### 3. Install dependencies

Core libraries:

```bash
pip install numpy pandas matplotlib scikit-learn medmnist joblib pillow jupyter ipykernel
```

For NVIDIA GPU acceleration, install the corresponding CUDA-enabled PyTorch build. The development environment used:

```bash
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130
```

The notebooks can also be executed on a CPU, although training will generally take longer.

### 4. Run the notebooks

Execute the notebooks in numerical order:

1. `01_data_exploration.ipynb`
2. `02_data_preprocessing.ipynb`
3. `03_model_training.ipynb`
4. `04_cnn_training.ipynb`
5. `05_final_evaluation.ipynb`

The data is downloaded through the MedMNIST package. Processed data, trained models and evaluation reports are saved locally.

## Key Findings

- Exact duplicate detection is an important part of preparing image datasets for reliable evaluation.
- Flattened pixel features provide a useful classical machine learning baseline.
- CNNs can learn spatial features directly from image data.
- The CNN achieved substantially better test performance than the Random Forest.
- Basophils and monocytes were particularly difficult for the Random Forest, but were recognized considerably better by the CNN.
- Evaluating models with macro-averaged metrics provides a more balanced picture than accuracy alone.

## Limitations

This is an educational machine learning project based on a standardized benchmark dataset.

The images are low-resolution (28 × 28 pixels) and represent individual blood cells from the BloodMNIST dataset. The results do not establish clinical validity or diagnostic suitability.

The reported test metrics apply only to this dataset and its evaluation protocol. Further validation on independent datasets and real-world clinical data would be required before any medical application could be considered.

## Future Work

Potential extensions include:

- Hyperparameter optimization and further CNN experiments
- FastAPI-based model serving
- Automated tests with pytest
- Docker-based deployment
- Model monitoring and versioning
- Unsupervised learning and clustering
- Experiments with embeddings and additional machine learning approaches

## References

1. Acevedo, A. et al. (2020). *A dataset of microscopic peripheral blood cell images for development of automatic recognition systems.* Data in Brief, 30, 105474. [https://doi.org/10.1016/j.dib.2020.105474](https://doi.org/10.1016/j.dib.2020.105474)

2. Yang, J. et al. (2023). *MedMNIST v2 - A large-scale lightweight benchmark for 2D and 3D biomedical image classification.* Scientific Data, 10, 41. [https://doi.org/10.1038/s41597-022-01721-8](https://doi.org/10.1038/s41597-022-01721-8)

3. [MedMNIST official repository](https://github.com/MedMNIST/MedMNIST)

The BloodMNIST dataset is distributed under the Creative Commons Attribution 4.0 International (CC BY 4.0) license. Please refer to the original dataset and publication for attribution details.