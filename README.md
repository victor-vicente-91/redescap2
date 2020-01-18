# Capsule Network Keras (Tensorflow 2.0)

This project aims to update the existing Capsule Network Architecture of [bojone](https://github.com/bojone/Capsule) (which was designed after https://kexue.fm/archives/5112) to be working with Tensorflow 2.0 & improving usability with the generalized input like in my [approach](https://github.com/TheLastFrame/CapsNet-Keras/) to update [XifengGuo's Capsule Network](https://github.com/XifengGuo/CapsNet-Keras/) to Tensorflow 2.0.

## Rquirements
Tensorflow >= 2.0.0

## How to use 
```console
python capsulenet.py -d training_data -n flower_model 
```
### Parameters
--epochs

--batch_size'

-r, --routings = Number of iterations used in routing algorithm. Should be > 0

--capsule_dim = Dimmension of the Capsule Layer.

*--debug = Save weights by TensorBoard (currently not working, may being deleted)*

--save_dir

--tflite = Option to export the trained model in Tensorflow Lite.

-d, --directory = Directory where the training data is stored. Error if not assigned.

-n, --name = Name for the model with which it will be saved.

-vs, --validation_split = Fraction of images reserved for validation (strictly between 0 and 1).

***###################### NOT fully implemented yet ###########################***

--grayscale' = Changes Network from grayscale mode to RGB mode.

--rotation_range = Rotation range for data augmentation.

--horizontal_flip = Enables horizontal flip for data augmentation.

--width_shift_range = Widht shift range for data augmentation. Should be within -1.0 to +1.0.

--height_shift_range = Height shift range for data augmentation. Should be within -1.0 to +1.0.

--shear_range = Shear range for data augmentation.

--zoom_range = Zoom range for data augmentation.

***not even implemented (comming in future itteration)***

*--channel_shift_range = Channel shift range for data augmentation.*

*--brightness_range = Brightness range for data augmentation.*

## Performance
### Training
#### MINST Dataset

|Graphics Card|min time/epoch|max time/epoch|steps/epoch|total training time|
|----------------|--------------|--------------|-----------|----------------|
|Nvidia GTX 1060M|
|Tesla T4 (Colab)

### Interference


## ToDo's
- create "capsulenet.py" for generalized image input via ImageDataGenerator
- add possibility to save as .tflite file
- create a load_model tutorial
- add saving method for easier model loading
- add benchmarks

**old text**

动态路由算法来自：https://kexue.fm/archives/5112

该版本的动态路由略微不同于Hinton原版，在“单数字训练、双数字测试”的准确率上有95%左右。

其他：

1、相比之前实现的版本：https://github.com/XifengGuo/CapsNet-Keras ，我的版本是纯Keras实现的(原来是半Keras半tensorflow)；

2、通过K.local_conv1d函数替代了K.map_fn提升了好几倍的速度，这是因为K.map_fn并不会自动并行，要并行的话需要想办法整合到一个矩阵运算；

3、其次我通过K.conv1d实现了共享参数版的；

4、代码运行环境是Python2.7 + tensorflow 1.8 + keras 2.1.4

## 交流
QQ交流群：67729435，微信群请加机器人微信号spaces_ac_cn
