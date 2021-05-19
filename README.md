# DeepImageTranslator
DeepImageTranslator: a free, user-friendly tool for image translation using deep-learning and its applications in CT image analysis 

DeepImageTranslator is designed to be a user-friendly graphical interface tool that allows researchers with no programming experience to easily build, train, and evaluate CNNs for image translation. Compared to existing software programs, our tool also allows users to customize their CNN (e.g. number of layers/channels, use of deep-supervision, and input layer resolution), the type of model optimizer algorithm, the loss function, and the data augmentation schemes. We showed that using only a standard personal computer, it is possible to train neural networks for accurate semantic segmentation of CT images. 

Installation and use :

1-	Download the software package at : https://sourceforge.net/projects/deepimagetranslator/

2-	Unzip the package

3-	Run DeepImageTranslator.exe

Citation: 

Ye RZ et al. DeepImageTranslator: a free, user-friendly graphical interface for image translation using deep-learning and its applications in 3D CT image analysis. biorxiv. 2021 May 18. doi: https://doi.org/10.1101/2021.05.15.444315 


![Fig  1](https://user-images.githubusercontent.com/84249081/118856072-87f41600-b8a4-11eb-874a-8c6bf05c1612.PNG)


Different features of DeepImageTranslator. a, The main window for image viewing. b, Training hyperparameter selection window. c, Neural network model builder. d, Command prompt window for training monitoring. e, Image augmentation toolbox.


![Fig  2](https://user-images.githubusercontent.com/84249081/118856107-96dac880-b8a4-11eb-9362-9fc283f4b420.PNG)


The pipeline for using the DeepImageTranslator. a, The construction of a custom U-net-like convolutional neural network and model training. b, The use of the trained neural network to make predictions based on new input data. ch: number of convolution maps (channels) after the first 3x3 convolution; Conv.: convolution; hin: input image height; ht: target image height; ReLU: rectified linear activation function; win: input image width; wt: target image width.


![Fig6](https://user-images.githubusercontent.com/84249081/118856307-cdb0de80-b8a4-11eb-9a31-fc33acd67f77.png)



Assessment of out-of-sample model generalizability based on scans from a severely obese male subject and a very lean female subject. a, Model generalizability in the obese male subject. left panel: input images randomly selected for inclusion in the figure; middle panel: ground truth segmentation maps; right panel: model predictions based on images in the left panel. b, Model generalizability in the obese male subject.  c, Model generalizability in the lean female subject. 


![Fig  10](https://user-images.githubusercontent.com/84249081/118856158-a528e480-b8a4-11eb-83a8-359a84f857ef.PNG)


Assessment of model generalizability for increased noise level. a, Randomly chosen thoracic CT slices from the obese subject with an increase of approximately 60% in noise intensity, demonstrating the efficiency of our model for noise reduction. b, Enlargement at the right and left lungs of the images shown in panel a, demonstrating our model’s ability to recover fine details of the pulmonary vasculature. Left panel: noisy CT images; middle panel: original
