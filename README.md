# Readme

## 1. MaSGAN Model
•	The MaSGAN model adopts an encoder-decoder-encoder structure. The primary encoder learns the distribution of normal samples, while the secondary encoder reduces the dimension of generated samples for anomaly detection.

•	It incorporates channel and frequency attention mechanisms in the convolutional block to enhance feature extraction capabilities.

•	During training, only normal samples are used, enabling the model to reconstruct normal sounds accurately and identify abnormal samples based on reconstruction errors.

•	The specific code of the MaSGAN model is in the masgan.py file.

## 2. LLSTM-MaSGAN Model
•	The LLSTM-MaSGAN model extends the MaSGAN model by integrating an improved linear long short-term memory network (LLSTM).

•	The LLSTM unit is designed to capture and analyze the temporal features of sound data, which is crucial for handling the dynamic changes in machine sound characteristics over time.

•	The specific code of the LLSTM-MaSGAN model is in the llstmmasgan.py file.

## 3. System operation instructions：
•	PyTorch 0.4.0

## 4. Dataset
The code is designed to work with the MIMII Dataset [1]. The dataset contains normal and abnormal sound clips from five types of industrial equipment (Fan, Gearbox, Pump, Slider rail, and Valve). The abnormal sounds were collected by damaging the machines, and background noise was added to simulate realistic environments.

[1] Kawaguchi Y, Imoto K, Koizumi Y, et al. Description and discussion on DCASE 2021 challenge task 2: Unsupervised anomalous sound detection for machine condition monitoring under domain shifted conditions[J], 2021.

## 5.Other  improvable modules
The Vision Transformer that can be added to the model and the TSM module for processing spatial information are in the vit.py and TSM.py files respectively.


