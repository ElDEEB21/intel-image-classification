# Intel Image Classification

This project aims to classify images from the Intel Image Classification dataset using two renowned deep learning models: AlexNet and ResNet18. The notebook `alex-res-net-classification.ipynb` encompasses the entire workflow, including data preprocessing, model training, evaluation, and inference.

## Project Overview

1. **Dataset Acquisition**:
    - The dataset is sourced from Kaggle using the `kagglehub` library.
    - The dataset path is displayed for reference.

2. **Libraries and Dependencies**:
    - Key libraries such as `torch`, `torchvision`, `matplotlib`, and `os` are imported.

3. **Data Preprocessing**:
    - A series of transformations are defined using `torchvision.transforms` to preprocess the images.
    - Images undergo resizing, random flipping, rotation, tensor conversion, and normalization.

4. **Dataset Loading**:
    - Training and testing directories are specified.
    - The dataset is loaded using `ImageFolder` and split into training and validation sets (70% training, 30% validation).

5. **Data Loaders**:
    - `DataLoader` objects are created for training, validation, and testing datasets to enable batch processing.

6. **Image Display Function**:
    - A function `imshow` is defined to display an image from the dataset along with its corresponding label.

7. **Sample Images Display**:
    - A batch of images is retrieved from the training dataset, and six sample images are displayed.

8. **Model Preparation**:
    - Pretrained models (`AlexNet` and `ResNet18`) are loaded from `torchvision.models`.
    - The final classification layer is adjusted to match the 6 output classes for Intel Image Classification.
    - Models are transferred to the available device (CPU or GPU).

9. **Training Function**:
    - A comprehensive training function is defined, incorporating:
      - Training and validation phases.
      - Loss and accuracy tracking.
      - Learning rate scheduler.
      - Model saving based on validation accuracy.
      - Training history recording.

10. **Hyperparameters and Training**:
     - Hyperparameters such as loss function, optimizer, learning rate scheduler, and number of epochs are defined.
     - Both `AlexNet` and `ResNet18` models are trained, and their training history is recorded.

11. **Model Evaluation**:
     - Functions to evaluate the trained models on the test set are defined.
     - Metrics such as accuracy, F1-score, and confusion matrix are computed and displayed.

12. **Training History Visualization**:
     - Functions to plot the training and validation loss and accuracy curves are defined.
     - Training history for each model is visualized.

13. **Model Inference**:
     - The trained ResNet18 model is loaded and prepared for inference.
     - New images are classified using the trained model, and predictions are saved to a CSV file.

## Conclusion

This project showcases the complete workflow of image classification using deep learning models. It covers data preprocessing, model training, evaluation, and inference, providing a comprehensive guide for similar tasks.
