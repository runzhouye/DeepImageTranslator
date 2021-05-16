# DeepImageTranslator
DeepImageTranslator: a free, user-friendly tool for image translation using deep-learning and its applications in CT image analysis 

DeepImageTranslator is designed to be a user-friendly graphical interface tool that allows researchers with no programming experience to easily build, train, and evaluate CNNs for image translation. Compared to existing software programs, our tool also allows users to customize their CNN (e.g. number of layers/channels, use of deep-supervision, and input layer resolution), the type of model optimizer algorithm, the loss function, and the data augmentation schemes. We showed that using only a standard personal computer, it is possible to train neural networks for accurate semantic segmentation of CT images. 

Installation and use :

1-	Download the software package at : https://sourceforge.net/projects/deepimagetranslator/

2-	Unzip the package

3-	Run DeepImageTranslator.exe


![image](https://user-images.githubusercontent.com/84249081/118382658-8f29d400-b5c5-11eb-8f0b-1150b30e91a7.png)

Different features of DeepImageTranslator. a, The main window for image viewing. b, Training hyperparameter selection window. c, Neural network model builder. d, Command prompt window for training monitoring. e, Image augmentation toolbox.


![image](https://user-images.githubusercontent.com/84249081/118382666-9d77f000-b5c5-11eb-86f2-51d9e8464471.png)

The pipeline for using the DeepImageTranslator. a, The construction of a custom U-net-like convolutional neural network and model training. b, The use of the trained neural network to make predictions based on new input data. ch: number of convolution maps (channels) after the first 3x3 convolution; Conv.: convolution; hin: input image height; ht: target image height; ReLU: rectified linear activation function; win: input image width; wt: target image width.


![image](https://user-images.githubusercontent.com/84249081/118382615-6a356100-b5c5-11eb-8a01-307db071e6fb.png)

Assessment of out-of-sample model generalizability based on scans from a severely obese male subject and a very lean female subject. a, Model generalizability in the obese male subject. left panel: input images randomly selected for inclusion in the figure; middle panel: ground truth segmentation maps; right panel: model predictions based on images in the left panel. b, Model generalizability in the obese male subject.  c, Model generalizability in the lean female subject. 


![image](https://user-images.githubusercontent.com/84249081/118382628-76b9b980-b5c5-11eb-9289-1e6fb3e44e8b.png)

Assessment of model generalizability for increased noise level. a, Randomly chosen thoracic CT slices from the obese subject with an increase of approximately 60% in noise intensity, demonstrating the efficiency of our model for noise reduction. b, Enlargement at the right and left lungs of the images shown in panel a, demonstrating our model’s ability to recover fine details of the pulmonary vasculature. Left panel: noisy CT images; middle panel: original
