GitHub repository with codes and explanations for the final project of ECBM4040 (Columbia University Deep Learning Course)
Final project is on the following article: MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications (https://arxiv.org/abs/1704.04861)

* utils/data.py : to download and load standford dogs dataset
* utils/MobileNet.py : MobileNet architecture implemented using keras
* utils/models.py : other models similar to MobileNet using Keras
* StanfordDogs-MobileNet-1.ipynb : MobileNet architecture with training on Stanford dogs dataset (with a width multiplier of 1)
* CIFAR-10-0.4 : MobileNet architecture with training on CIFAR-10 dataset (with a width multiplier of 0.4)
* CIFAR-10-0.75 : MobileNet architecture with training on CIFAR-10 dataset (with a width multiplier of 0.75)
* Builtin_MobileNet.ipynb : MobileNet using the ImageNet pre-trained weights (training on Stanford dogs)
* requirement.txt : file with all libraries and versions

If you want to use GPU, you have to download tensorflow-gpu==2.0.0 instead of tensorflow==2.0.0.
