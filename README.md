# FireDetectionCV (SST - Datathon)
Computer Vision program for detecting fire in video and images. The program was developed by the AIturbo team as solution of a case from Smart System Technologies in ML-fest hackathon. 

## About the project
1) The fire detection training model was written in Python using Tensorflow & Keras;
2) `train_imgs` dataset was used as data for training the model;
3) `PandNdivider.py` script was used to divide images into 2 datasets: **negative** & **positive** by the labels in `train_labels.csv` file;
4) Generated model name is `fire_model.h5`;
5) `fire_keras.py` is the main program that processes all images in the directory `test_imgs` by our trained model and shows the results of testing.
6) `test_labels.csv` contains results of the processing of all images of `test_imgs` dataset in which the class `1` means the presence of fire, and `0` its absence;

## Testing & Results

**Final results of testing our program in the `test_labels.csv` file!**

[![Testing Results][test-imgs-results]][contributors-url]


## Acknowledgments

* [TensorFlow](https://www.tensorflow.org/)
* [Keras](https://keras.io/)
* [Google Collab](https://colab.research.google.com/)


[test-imgs-results]: https://sun9-west.userapi.com/sun9-6/s/v1/ig2/Xz950WkFJFfigR6y7fyG4IhhLkh1lv2HR_NP83PqLg2hFAfHmxa5g-wVI9wjkTVtL6g-1VWvLwrBlfWueWVdpQpS.jpg?size=639x210&quality=95&type=album
[contributors-url]: https://github.com/silvermete0r
