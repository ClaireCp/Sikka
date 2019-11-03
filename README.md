# Sikka
The aim of this project is to design and train an MLP for speech digits recognition (0 to 9). The dataset used is the Free Spoken Digit Dataset (FSDD) available at https://github.com/Jakobovski/free-spoken-digit-dataset.

## Data Processing
For this task, data processing for feature extraction is the first and most important part. We use the Mel-frequency cepstral coeefficients (MFCCs) of the recordings, a feature widely used in automatic speech recognition. Some nice ressources:

  https://towardsdatascience.com/ok-google-how-to-do-speech-recognition-f77b5d7cbe0b
  https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html?fbclid=IwAR3NiJE_ZLoZUtDiCFclWFfWL0aRY7t0yuCcJ2cgXxaFk5e27TiQR2-Z-5I
  http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/

We also use data augmentation to add noise, shift and stretch the signal for the training set.

## ANN and training
By simply plotting the fast Fourier transforms of the signals wrt to the frequencies, we see that the signals are already very distinguishables, and so only a small NN should be necessary. The ANN is made up of only 3 layers, including input and output layers. The hidden layer has only 5 units, with ReLu activation function. Since this is a multi-classification task, we train using Cross-Entropy Loss. We train using Adadelta, on 200 epochs.

## MLPs in supervised learning
MLPs are universal function approximators, ie they can approximate any continuous functions on compact subsets of R<sup>n</sup> (under mild assumptions on the activation function). This is the universal approximation theorem. As such, MLP are a generic model that can adapt and solve complex tasks. By leveraging new computational software and hardware, MLP and deep-MLP can be trained very efficiently for both regression and classification problems, using the appropriate loss and the backpropagation algorithm.

However, deep-learning can have very slow convergence compared to numerical methods designed for a specific task. In the context of speech recognition and signal processing for exemple, using the fast Fourier Transform (FFT) on the data allows to almost directly distinguish between the digits, and requires only o(nlog(n)). Learning the FFT using a ANN (with linear activation function) would be more computationaly intensive.
The best solution for solving this problem is to use the 1D-scattering transform, as proposed here:
  https://www.kymat.io/gallery_1d/plot_classif.html
  
The Border Pair Method is an iterative procedure of building the optimal (smallest) MLP and computing the value of the parameters of the MLP directly from the geometry of the training data, thus avoiding the use of backpropagation. Link:
https://www.researchgate.net/publication/249011199_Border_Pairs_Method-constructive_MLP_learning_classification_algorithm
