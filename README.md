

# Grad-Cam - Tensorflow Slim 

<img src="https://github.com/hiveml/tensorflow-grad-cam/blob/master/images/cat_heatmap.png">


### Features:

Modular with Tensorflow slim. Easy to drop in other [Slim Models](https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models)

Udated to work with Tensorflow 1.5

Includes various output options: heatmap, shading, blur

#### More examples and explanation [here](https://thehive.ai/blog/inside-a-neural-networks-mind)


### Dependencies

* Python 2.7 and pip
* Scikit image : `sudo apt-get install python-skimage`
* Tkinter: `sudo apt-get install python-tk`
* Tensorflow >= 1.5 : `pip install tensorflow==1.5.0rc0`
* Opencv - see https://docs.opencv.org/2.4/doc/tutorials/introduction/linux_install/linux_install.html


### Installation

Clone the repo:
```
git clone https://github.com/hiveml/tensorflow-grad-cam.git
cd tensorflow-grad-cam
```
Download the ResNet-50 weights:
```
./imagenet/get_checkpoint.sh
```
### Usage
```
./main.sh
```

### Changing the Class

By default this code shows the grad-cam results for the top class. You can 
change the `predicted_class` argument to function `grad_cam` to see where the network 
would look for other classes.

### How to load another resnet\_v2 model

First download the new model from here: [Slim Models](https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models)

Then modify the input arguments in main.sh:
```
python main.py --model_name=resnet_v2_101 --dataset_dir=./imagenet/ --checkpoint_path=./imagenet/resnet_v2_101.ckpt --input=./images/cat.jpg --eval_image_size=299

```
<img src="https://github.com/hiveml/tensorflow-grad-cam/blob/master/images/scarjo.png">
