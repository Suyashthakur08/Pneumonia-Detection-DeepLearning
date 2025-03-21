# Pneumonia Detection using Deep Learning

## Overview
This project applies deep learning techniques to classify chest X-ray images into Normal and Pneumonia categories. The model is trained on the **Chest X-ray Pneumonia dataset** from Kaggle, using three different architectures: **VGG16, DenseNet121, and ResNet50**. All three models achieved high accuracy, with the best model reaching **96.23% accuracy**.

## Dataset
The dataset consists of chest X-ray images categorized into:
- **Normal**: Healthy lung X-rays.
- **Pneumonia**: X-rays with signs of pneumonia.

Dataset structure:
```
/kaggle/input/chest-xray-pneumonia/
    ├── chest_xray/
        ├── train/
            ├── NORMAL/
            ├── PNEUMONIA/
        ├── test/
            ├── NORMAL/
            ├── PNEUMONIA/
        ├── val/
            ├── NORMAL/
            ├── PNEUMONIA/
```

## Model Architectures & Performance
Three deep learning architectures were implemented:

1. **VGG16**  
   - Accuracy: **94.69%**  
   - Precision: **93.5%**  
   - Recall: **95.2%**  

2. **DenseNet121**  
   - Accuracy: **95.00%**  
   - Precision: **94.0%**  
   - Recall: **96.0%**  

3. **ResNet50**  
   - Accuracy: **96.23%**  
   - Precision: **95.5%**  
   - Recall: **96.8%**  

The models were trained using TensorFlow/Keras, with data augmentation techniques to enhance generalization.

## Requirements
Install the following dependencies before running the project:
```bash
pip install tensorflow keras numpy pandas matplotlib seaborn scikit-learn
```

## Usage

### Training the Models
To train a model, run:
```bash
python train.py --model vgg16 --epochs 20 --batch_size 32 --lr 0.0001
python train.py --model densenet121 --epochs 20 --batch_size 32 --lr 0.0001
python train.py --model resnet50 --epochs 20 --batch_size 32 --lr 0.0001
```

### Evaluating the Models
To evaluate a trained model:
```bash
python test.py --model_path vgg16_model.pth
python test.py --model_path densenet121_model.pth
python test.py --model_path resnet50_model.pth
```

### Predicting on New Images
```bash
python predict.py --image_path path/to/image.jpg --model_path resnet50_model.pth
```

## Future Improvements
- Experimenting with ensemble learning using multiple architectures.
- Fine-tuning hyperparameters for optimal results.
- Deployment as a web service for real-time predictions.

## Acknowledgments
- **Dataset:** [Chest X-ray Pneumonia Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Libraries:** TensorFlow, Keras, OpenCV, Albumentations, and scikit-learn.
