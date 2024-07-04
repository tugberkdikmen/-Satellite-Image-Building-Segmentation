# Satellite Image Segmentation with U-Net

This project focuses on the segmentation of satellite images to identify urban and rural buildings using the U-Net architecture. The goal is to create a robust model that can accurately segment buildings from satellite imagery, enabling applications such as urban planning, disaster management, and environmental monitoring.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Models](#models)
- [Implementation](#implementation)
- [Results](#results)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [References](#references)
- [Links](#links)

## Introduction

Satellite image segmentation is a crucial task in remote sensing, which involves partitioning an image into different regions based on the objects present. This project utilizes the U-Net architecture, a convolutional neural network designed for biomedical image segmentation, adapted for satellite imagery.

## Dataset

The dataset used in this project includes preprocessed satellite images and their corresponding masks. The dataset is divided into urban and rural areas to ensure the model performs well in both settings.

- **Preprocessed Satellite Images and Masks**: [Kaggle Dataset](https://www.kaggle.com/datasets/tugberkdikmen/preprocessed-satellite-images-masks)

## Models

Several models were trained with different configurations and epochs. The best-performing models are available for download.

- **Trained Models**: [Kaggle Models](https://www.kaggle.com/models/tugberkdikmen/sat_img_seg_models/settings)

## Implementation

The implementation involves data preprocessing, model training, and evaluation. The key components include:

- **Data Augmentation**: To improve model generalization, various data augmentation techniques were applied.
- **Regularization**: L2 regularization and dropout layers were used to prevent overfitting.
- **Callbacks**: Early stopping and learning rate reduction were implemented to optimize training.

### Key Scripts

1. **Data Preprocessing**: Scripts to resize images and masks, and save them in a format suitable for model training.
2. **Model Training**: Training the U-Net model with data augmentation and regularization.
3. **Evaluation**: Scripts to evaluate the model's performance on test data and visualize the results.

### Jupyter Notebook

- **U-Net Implementation Notebook**: [Kaggle Notebook](https://www.kaggle.com/code/tugberkdikmen/u-net/notebook)

## Results

The results show that the U-Net model can effectively segment buildings from satellite images. Below are some key metrics:

- **Validation Loss**: ~0.23xx
- **Validation Accuracy**: ~0.72xx

### Example Predictions

![Example Prediction](path_to_example_image.png)

## Future Work

- **Enhance Model Architecture**: Experiment with different architectures and hyperparameters.
- **Increase Dataset Size**: Incorporate more diverse satellite images for training.
- **Deploy Model**: Develop a web application to allow users to upload satellite images and receive segmentation results.

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your changes.

## References

- **U-Net: Convolutional Networks for Biomedical Image Segmentation**: [Link to paper](https://arxiv.org/abs/1505.04597)
- **Satellite Image Footprints**: [Link to Microsoft Dataset](https://github.com/microsoft/GlobalMLBuildingFootprints)

## Links

- **Project Notebook**: [U-Net Notebook](https://www.kaggle.com/code/tugberkdikmen/u-net/notebook)
- **Dataset**: [Preprocessed Satellite Images and Masks](https://www.kaggle.com/datasets/tugberkdikmen/preprocessed-satellite-images-masks)
- **Models**: [Trained Models](https://www.kaggle.com/models/tugberkdikmen/sat_img_seg_models)
