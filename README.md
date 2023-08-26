# Brain-Tumor-Segmentation-using-BRATS-22-Challenge

## Brain tumor due to is one of the most dangerous, life-threatening diseases faced by several globally.

  Today a plethora of research is conducted to automate powerful biomedical instruments
and detection tools using deep learning and machine learning techniques. Latest innovations are
being worked on to produce new technologies that can detect brain tumors faster and more
inexpensively. Automatic tumor segmentation should come with decreased errors, speed-up,
efficiency, and accuracy in the detection of tissue types as they are all factors of importance.
Nevertheless, this still poses a challenge due to the difficulties in obtaining real MRI scans of
large numbers for better and more efficient image-classifying models. The automatic detection
and histological semantic segmentation of distinct brain tumor sub-regions are done by a simple
efficient [UNET architecture](https://towardsdatascience.com/unet-line-by-line-explanation-9b191c76baf5)  which was later encoded with the pre-trained model [MobileNetV2](https://www.mathworks.com/help/deeplearning/ref/mobilenetv2.html)
based on transfer learning. Here, the deep learning technique is to use feature mapping with a
pyramid network for the encoder to obtain more spatial relevant features. The performance of the
two models was trained using real brain MRI images from the RSNA BRATS 2021 dataset and the results of the models are compared with recall, precision, accuracy, dice similarity in sub-
region vice classification, and dice loss. The results from the hybrid model had dice coefficients of 0.867 , 0.679, and 0.7153 for edema, enhancing tumor, and core tumor respectively with a
good accuracy of 90.78% for 4-classes and dice loss of 0.0187.

> **Keywords: brain tumor segmentation, deep learning, U-Net architecture, Encoder, transfer
learning, MobilenetV2, skip-connections**

![TASK1_train_00003](https://github.com/pk2203/Brain-Tumor-Segmentation-using-BRATS-22-Challenge/assets/105013665/189efaf4-07f2-472a-ba84-baaceefeabd2)

Figure 1: Four main channels of MRI image named flair, t1, t1ce, and t2 along with ground truth mask

## Objectives and Contributions:
• Image preprocessing and noise removal to smoothen the non-uniformity of training MRI
scan set
• Design the semantic segmentation U-Net architecture with appropriate filters and
hyperparameters such that the trainable parameters stay within the limits of memory
allocation possible.
• Understand the workings of the initial base model and fine tune accordingly to improve
results.
• Implementing appropriate hyperparameters for data augmentation of input images.
• Create the final hybrid model with the first model as the main base by importing the pre-
trained MobileNetV2 model as the encoder.
• Deploy the model into a web application to get a user-friendly demonstration.

## BraTs_MobileUNet:
1. Convolutional layer: Using a 2D operation for feature extraction by taking the input
image and adding feature channels to create a convoluted map.
2. Max Pooling layer: This extracts the higher significant images and cuts down the size as
it moves to the layers ahead.
3. Dropout layer: Used to prevent over-fitting while training the model by imputing random
noise to the data.
4. Convolutional Transpose layer: Performs the basic convolutional operation but it Up-
Samples the layer while allowing to concatenate exterior featured maps to it.

| Model Name | Trainable Params | Non-Train Params | Size of Model | Total Params |
|------------|------------------|------------------|---------------|--------------|
| Simple U-Net| 1,980,564       | 992              |   0.162       |   1,981,556  |
| MobileUNet  | 12,545,940      | 6,544            |    0.561      |   12,545,940 |

> As a result, it can be noted that the U-Net structure with a pre-trained model as an encoder can
stand superior to several approaches hence making it an efficient model for the application of
brain tumor segmentation.
