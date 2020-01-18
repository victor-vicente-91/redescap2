from capsulelayer import *
import tensorflow as tf
from tensorflow.keras import utils
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
import numpy as np
import os
import argparse

def load_data(args):
	datagen_kwargs = dict(rescale=1./255, validation_split=args.validation_split)
	
	datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    **datagen_kwargs)
	
	val_generator = datagen.flow_from_directory(
		args.directory,
		target_size=(224, 224),
		batch_size=args.batch_size,
        color_mode = 'grayscale',
		subset='validation')

	
	train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
		#rotation_range=40,
		#horizontal_flip=True,
		#width_shift_range=0.2, height_shift_range=0.2,
		#shear_range=0.2, zoom_range=0.2,      
		**datagen_kwargs) #channel_shift_range=0.2, #brightness_range=[-0.1, 0.1], 

	train_generator = train_datagen.flow_from_directory(
		args.directory, subset="training", shuffle=True,
		target_size=(224, 224),
        color_mode = 'grayscale',
		batch_size=args.batch_size)

	return train_generator, val_generator

def margin_loss(y_true, y_pred):
    """
    Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
    :param y_true: [None, n_classes]
    :param y_pred: [None, num_capsule]
    :return: a scalar loss value.
    """
    #from older version
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))
    return K.mean(K.sum(L, 1))

    #return y_true*K.relu(0.9-y_pred)**2 + 0.25*(1-y_true)*K.relu(y_pred-0.1)**2

    


##Prepare the training data
#batch_size = 128
#num_classes = 10
#img_rows, img_cols = 28, 28

#(x_train, y_train), (x_test, y_test) = mnist.load_data()
#x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
#x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

#x_train = x_train.astype('float32')
#x_test = x_test.astype('float32')
#x_train /= 255
#x_test /= 255
#y_train = utils.to_categorical(y_train, num_classes)
#y_test = utils.to_categorical(y_test, num_classes)


##准备自定义的测试样本
##对测试集重新排序并拼接到原来测试集，就构成了新的测试集，每张图片有两个不同数字
#idx = list(range(len(x_test)))
#np.random.shuffle(idx)
#X_test = np.concatenate([x_test, x_test[idx]], 1)
#Y_test = np.vstack([y_test.argmax(1), y_test[idx].argmax(1)]).T
#X_test = X_test[Y_test[:,0] != Y_test[:,1]] #确保两个数字不一样
#Y_test = Y_test[Y_test[:,0] != Y_test[:,1]]
#Y_test.sort(axis=1) #排一下序，因为只比较集合，不比较顺序


if __name__ == "__main__":

    # setting the hyper parameters
    parser = argparse.ArgumentParser(description="Capsule Network written in Pure Keras.")
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('-r', '--routings', default=3, type=int,
                        help="Number of iterations used in routing algorithm. should > 0")
    parser.add_argument('--capsule_dim', default=16, type=int, help="Dimmension of the Capsule Layer.")
    parser.add_argument('--debug', action='store_true',
                        help="Save weights by TensorBoard")
    parser.add_argument('--save_dir', default='./result')
    parser.add_argument('--tflite', default=False, help="Option to export the trained model in Tensorflow Lite.")
    parser.add_argument('-d', '--directory', default="./images", help="Directory where the training data is stored. Error if not assigned.")
    parser.add_argument('-n', '--name', default="trained_model", help="Name for the model with which it will be saved.")
    parser.add_argument('-vs', '--validation_split', default=0.2, type=float, help="Fraction of images reserved for validation (strictly between 0 and 1).")
    ###################### NOT fully implemented yet ###########################
    parser.add_argument('--grayscale', default=True, help="Changes Network from grayscale mode to RGB mode.")
    parser.add_argument('--rotation_range', default=0, type=int, help="Rotation range for data augmentation.")
    parser.add_argument('--horizontal_flip', default=False, help="Enables horizontal flip for data augmentation.")
    parser.add_argument('--width_shift_range', default=0.0, type=float, help="Widht shift range for data augmentation. Should be within -1.0 to +1.0.")
    parser.add_argument('--height_shift_range', default=0.0, type=float, help="Height shift range for data augmentation. Should be within -1.0 to +1.0.")
    parser.add_argument('--shear_range', default=0.0, type=float, help="Shear range for data augmentation.")
    parser.add_argument('--zoom_range', default=0.0, type=float, help="Zoom range for data augmentation.")
	#parser.add_argument('--channel_shift_range', default=0.0, type=float, help="Channel shift range for data augmentation.")
	#parser.add_argument('--brightness_range', default=0.0, type=float, help="Brightness range for data augmentation.")
    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # load data
    #IMAGE_SIZE = 224
    train_generator, val_generator = load_data(args)

	#save image labels to file
    print (train_generator.class_indices)
    classes = len(train_generator.class_indices.keys())
    labels = '\n'.join(sorted(train_generator.class_indices.keys()))
    label_file_name = args.save_dir + '\\' + args.name + '_labels.txt'
    with open(label_file_name, 'w') as f:
        f.write(labels)

    # define model
    input_image = Input(shape=(None,None,1 if(args.grayscale==True) else 3))
    cnn = Conv2D(64, (3, 3), activation='relu')(input_image)
    cnn = Conv2D(64, (3, 3), activation='relu')(cnn)
    cnn = AveragePooling2D((2,2))(cnn)
    cnn = Conv2D(128, (3, 3), activation='relu')(cnn)
    cnn = Conv2D(128, (3, 3), activation='relu')(cnn)
    cnn = Reshape((-1, 128))(cnn)
    capsule = Capsule(classes, args.capsule_dim, args.routings, True)(cnn) #num capsule (classes), dim capsule, routings
    output = Lambda(lambda x: K.sqrt(K.sum(K.square(x), 2)), output_shape=(10,))(capsule)

    model = Model(inputs=input_image, outputs=output)
    model.compile(loss=margin_loss,
                  optimizer='adam',
                  metrics=['accuracy'])

    model.summary()

    #train_x, train_y = val_generator

    model.fit_generator(train_generator,
            #batch_size=args.batch_size,
            epochs=args.epochs,
            verbose=1,
            validation_data=val_generator)

    tf.keras.models.save_model(model, args.save_dir + args.name+".h5")

    model.save(args.save_dir + args.name, save_format='tf')

    #if(args.tflite):
        #converter = tf.lite.TFLiteConverter.from_keras_model(model)
        #tflite_model = converter.convert()

        #with open('output.tflite', 'wb') as f:
        #  f.write(tflite_model)