##Transfer Learning Experiment

1. Experiment Summary

Objective: Compare two transfer learning strategies on the Food-11 dataset:

Feature Extraction: Freeze the pre-trained base model and train only the new classifier head.
Fine-Tuning: Unfreeze the base model and train jointly (not executed in this notebook).
Dataset: Food-11 image dataset from Kaggle (trolukovich/food11-image-dataset), consisting of 11 classes with 9,866 training images and 3,430 validation images. Images were resized to 256×256, rescaled by 1/255, and augmented (zoom_range=0.2, horizontal_flip, vertical_flip).

Model Architecture:

Base: InceptionResNetV2 (weights="imagenet", include_top=False, input_shape=(256,256,3))

Head:

GlobalAveragePooling2D

Flatten

Dense(256, activation="relu") + Dropout(0.3)

Dense(128, activation="relu") + Dropout(0.3)

Dense(11, activation="softmax")

Training Setup:

Batch size: 32

Optimizer: SGD (learning_rate=0.01, momentum=0.9)

Loss: Categorical Crossentropy; Metrics: Accuracy

Callbacks: EarlyStopping(monitor='loss', patience=20), ModelCheckpoint(save_best_only=True)

Epochs: 1 (feature extraction only)

2. Observations

2.1 Feature Extraction vs. Fine-Tuning

Convergence Speed: Feature extraction converged in 1 epoch; fine-tuning was not performed.

Final Accuracy: Feature extraction achieved 80.02% test accuracy; no fine-tuning comparison available.

Training Stability: Training was stable over the single epoch, but more epochs are required to fully assess plateau behavior.

2.2 Generalization

Validation Gap: After 1 epoch, the gap between training and validation accuracy is minimal, indicating consistent behavior for this short run.

Domain Adaptation: Fine-tuning was not executed, so domain-specific improvements remain untested.



2.4 Overfitting

Loss Divergence: No divergence between training and validation losses within one epoch.

Regularization Effects: Dropout (rate=0.3) was used in the head; overfitting effects were not evident at epoch 1.
