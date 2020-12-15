# Face Mask Detector

<img src="https://github.com/rajtilakls2510/face_mask_detector/blob/master/Videos/Screenshot_2020-12-15-08-38-07-770_com.mxtech.videoplayer.ad.jpg">

### Aim
    
The goal of this project was to build an Android App which could **Detect People with Mask or No Mask**.

### Contents of this Repo

- Jupyter Notebook where the model was created and trained. (This is actually a google colab notebook which was downloaded and put here).

Static Notebook Link: https://github.com/rajtilakls2510/face_mask_detector/blob/master/Applications/Training_Notebook.ipynb

Google colab Link: https://colab.research.google.com/drive/1iyeUiE1kCfsI6PkK1LCFHUz76AcIqUcZ?usp=sharing

- Python Project which uses webcam to detect masks.

Link: https://github.com/rajtilakls2510/face_mask_detector/tree/master/Applications/face_detector_priliminary

- Android App 

Link to project(Not APK: SRC): https://github.com/rajtilakls2510/face_mask_detector/tree/master/Applications/FaceMaskDetector%20Android%20App


###  Libraries

- Tensorflow and Keras (For Neural Network)
- OpenCV (For Image Processing: Applying Bounding Box )
- Tensorflow lite (For deployment in Android)

### How the project was done?

- Built the Neural Network using Tensorflow and Keras
    - Architecture: MobileNetV2 + Own Classifier

- Converted the Keras model to TFLITE model using Tensorflow Lite (With Quantization)

- Used the Keras model in Python Script to detect faces in Webcam

- Used the Tflite model in Android


### Dataset:
https://www.kaggle.com/rajtlakls2510/face-mask-detection
