# Bone Fracture Detection Using Deep Learning

This project focuses on detecting bone fractures from X-ray images using deep learning. A pretrained ResNet18 model is used to classify X-ray images into two categories: Fractured and Not Fractured.

The project is developed using Python and PyTorch.

## Project Objective

The main objective is to build and evaluate a Convolutional Neural Network (CNN) that can identify bone fractures from X-ray images.

## Dataset

The project uses the Bone Fracture Multi-Region X-ray dataset available on Kaggle.

Dataset Link:
https://www.kaggle.com/datasets/bmadushanirodrigo/fracture-multi-region-x-ray-data

The dataset contains X-ray images from different body regions, divided into training, validation, and testing sets.

## Model and Training

- Model: Pretrained ResNet18
- Framework: PyTorch
- Image Size: 224 × 224
- Optimizer: Adam
- Loss Function: CrossEntropyLoss
- Learning Rate: 0.0001
- Batch Size: 16
- Maximum Epochs: 20
- Early Stopping Patience: 5

The code also supports ResNet34 and ResNet50.

## Methodology

1. Load and prepare the X-ray dataset.
2. Remove exact duplicate images.
3. Resize and normalize the images.
4. Train the pretrained ResNet model.
5. Validate the model during training.
6. Save the best-performing model.
7. Evaluate the model using the test dataset.

## Results

The model achieved the following performance:

| Metric | Result |
|---|---|
| Best Validation Accuracy | 98.66% |
| Test Accuracy | 90.06% |
| Precision | 82.47% |
| Recall | 99.04% |
| F1-Score | 90.00% |

The model was evaluated on 463 test images, including 209 fractured and 254 non-fractured images.

The high recall indicates that the model successfully identified most fractured X-ray images in the test dataset.

## Technologies Used

- Python
- PyTorch
- Torchvision
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Pillow

## How to Run

1. Clone or download the repository.

2. Install the required libraries:

```bash
pip install -r requirements.txt
```

3. Download the dataset from Kaggle and place the `Bone_Fracture_Binary_Classification` folder in the project directory.

4. Run the project:

```bash
python main.py
```

The model will train and evaluate automatically. Results will be saved in the `output` folder.

## Project Files

- `main.py` - Main execution file
- `model.py` - ResNet model architecture
- `dataset.py` - Image loading and preprocessing
- `data_preparation.py` - Dataset preparation
- `trainer.py` - Model training and validation
- `evaluate.py` - Model evaluation
- `args.py` - Training configurations
- `requirements.txt` - Required libraries

## Limitations

The model is trained on a public X-ray dataset, and its performance may vary on other datasets. Patient-level separation between dataset splits has not been verified.

This project is intended for educational and research purposes only, not for clinical diagnosis.

## Author

Md Sajid Husen

Bone Fracture Detection Using Deep Learning