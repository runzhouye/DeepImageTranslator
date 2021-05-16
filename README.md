# DeepImageTranslator
DeepImageTranslator: a free, user-friendly graphical interface for image translation using deep-learning and its applications in 3D CT image analysis

DeepImageTranslator is designed to be a user-friendly graphical interface tool that allows researchers with no programming experience to easily build, train, and evaluate CNNs for image translation. Compared to existing software programs, our tool also allows users to customize their CNN (e.g. number of layers/channels, use of deep-supervision, and input layer resolution), the type of model optimizer algorithm, the loss function, and the data augmentation schemes. We showed that using only a standard personal computer, it is possible to train neural networks for accurate semantic segmentation of CT images. The standalone software is freely available for Windows 10 at: https://sourceforge.net/projects/deepimagetranslator/

![image](https://user-images.githubusercontent.com/84249081/118382615-6a356100-b5c5-11eb-8a01-307db071e6fb.png)

Assessment of out-of-sample model generalizability based on scans from a severely obese male subject and a very lean female subject. a, Model generalizability in the obese male subject. left panel: input images randomly selected for inclusion in the figure; middle panel: ground truth segmentation maps; right panel: model predictions based on images in the left panel. b, Model generalizability in the obese male subject.  c, Model generalizability in the lean female subject. 

![image](https://user-images.githubusercontent.com/84249081/118382628-76b9b980-b5c5-11eb-9289-1e6fb3e44e8b.png)

Assessment of model generalizability for increased noise level. a, Randomly chosen thoracic CT slices from the obese subject with an increase of approximately 60% in noise intensity, demonstrating the efficiency of our model for noise reduction. b, Enlargement at the right and left lungs of the images shown in panel a, demonstrating our model’s ability to recover fine details of the pulmonary vasculature. Left panel: noisy CT images; middle panel: original
