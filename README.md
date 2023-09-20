# Detection of Atrial Fibrillation Using 1D Convolutional Neural Network
# Abstract
The automatic detection of atrial fibrillation (AF) is crucial for its association with the risk of embolic stroke. Most of the existing AF detection methods usually convert 1D time-series electrocardiogram (ECG) signal into 2D spectrogram to train a complex AF detection system, which results in heavy training computation and high implementation cost. This paper proposes an AF detection method based on an end-to-end 1D convolutional neural network (CNN) architecture to raise the detection accuracy and reduce network complexity. By investigating the impact of major components of a convolutional block on detection accuracy and using grid search to obtain optimal hyperparameters of the CNN, we develop a simple, yet effective 1D CNN. Since the dataset provided by PhysioNet Challenge 2017 contains ECG recordings with different lengths, we also propose a length normalization algorithm to generate equal-length records to meet the requirement of CNN. Experimental results and analysis indicate that our method of 1D CNN achieves an average F1 score of 78.2%, which has better detection accuracy with lower network complexity, as compared with the existing deep learning-based methods.

# System architecture
![](system%20architecture.png)

# Citation
```
@article{hsieh2020detection,
  title={Detection of atrial fibrillation using 1D convolutional neural network},
  author={Hsieh, Chaur-Heh and Li, Yan-Shuo and Hwang, Bor-Jiunn and Hsiao, Ching-Hua},
  journal={Sensors},
  volume={20},
  number={7},
  pages={2136},
  year={2020},
  publisher={MDPI}
}
```
