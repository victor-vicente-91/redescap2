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
import time

def load_data(args):
    datagen_kwargs = dict(rescale=1./255, validation_split=args.validation_split)
    #datagen_kwargs = dict(rescale=1./255)
    
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(**datagen_kwargs)
    
    generator_args = dict()
    if(args.image_size!=0):
        generator_args["target_size"] = (args.image_size,args.image_size)
    if(args.grayscale):
        generator_args["color_mode"] = 'grayscale'
        

    val_generator = datagen.flow_from_directory(
        #args.directory_validation,
        args.directory,
        batch_size=args.batch_size,
        subset='validation',
        **generator_args)

    
    train_datagen_args = datagen_kwargs.copy()

    if (args.rotation_range!=0):
        train_datagen_args["rotation_range"]=args.rotation_range
    if(args.horizontal_flip):
        train_datagen_args["horizontal_flip"] = True
    if(args.width_shift_range!=0.0):
        train_datagen_args["width_shift_range"] = args.width_shift_range
    if(args.height_shift_range!=0.0):
        train_datagen_args["height_shift_range"] = args.height_shift_range
    if(args.shear_range!=0.0):
        train_datagen_args["shear_range"]=args.shear_range
    if(args.zoom_range!=0.0):
        train_datagen_args["zoom_range"]=args.zoom_range
    if(args.channel_shift_range!=0.0):
        train_datagen_args["channel_shift_range"]=args.channel_shift_range
    if(args.brightness_range!=0.0):
        train_datagen_args["brightness_range"] = [args.brightness_range*-1, args.brightness_range]


    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
		**datagen_kwargs)

    train_generator = train_datagen.flow_from_directory(
		args.directory, subset="training", shuffle=True,
		batch_size=args.batch_size,
        **generator_args)

    return train_generator, val_generator


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
    parser.add_argument('--save_dir', default='result')
    parser.add_argument('--tflite', default=False, help="Option to export the trained model in Tensorflow Lite.")
    parser.add_argument('-d', '--directory', default='images', help="Directory where the training data is stored. Error if not assigned.")
    parser.add_argument('-dv', '--directory_validation', default='images', help="Directory where the validation data is stored. Error if not assigned.")
    parser.add_argument('-n', '--name', default="trained_model", help="Name for the model with which it will be saved.")
    parser.add_argument('-vs', '--validation_split', default=0.2, type=float, help="Fraction of images reserved for validation (strictly between 0 and 1).")    
    parser.add_argument('--image_size', default=0, type=int, help="Size for images which should be used by model (image_size x image_size).")
    ###################### NOT fully implemented yet ###########################
    parser.add_argument('--grayscale', default=True, help="Changes Network from grayscale mode to RGB mode.")
    parser.add_argument('--rotation_range', default=0, type=int, help="Rotation range for data augmentation.")
    parser.add_argument('--horizontal_flip', default=False, help="Enables horizontal flip for data augmentation.")
    parser.add_argument('--width_shift_range', default=0.0, type=float, help="Widht shift range for data augmentation. Should be within -1.0 to +1.0.")
    parser.add_argument('--height_shift_range', default=0.0, type=float, help="Height shift range for data augmentation. Should be within -1.0 to +1.0.")
    parser.add_argument('--shear_range', default=0.0, type=float, help="Shear range for data augmentation.")
    parser.add_argument('--zoom_range', default=0.0, type=float, help="Zoom range for data augmentation.")
    parser.add_argument('--channel_shift_range', default=0.0, type=float, help="Channel shift range for data augmentation.")
    parser.add_argument('--brightness_range', default=0.0, type=float, help="Brightness range for data augmentation.")
    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if(args.tflite == True and args.image_size==0):
        print("Image Size must be set if exporting as .tflite!")
        sys.exit()

    # load data
    train_generator, val_generator = load_data(args)

	#save image labels to file
    print (train_generator.class_indices)
    classes = len(train_generator.class_indices.keys())
    labels = '\n'.join(sorted(train_generator.class_indices.keys()))
    label_file_name = args.save_dir + '\\' + args.name + '_labels.txt'
    with open(label_file_name, 'w') as f:
        f.write(labels)

    # define model
    input_image = Input(shape=(args.image_size if(args.image_size!=0) else None,
                               args.image_size if(args.image_size!=0) else None,
                                      1 if(args.grayscale==True) else 3))
    cnn = Conv2D(256, (3, 3), activation='relu')(input_image)
    cnn = Conv2D(256, (3, 3), activation='relu')(cnn)
    #cnn = AveragePooling2D((2,2))(cnn)
    #cnn = Conv2D(128, (3, 3), activation='relu')(cnn)
    #cnn = Conv2D(128, (3, 3), activation='relu')(cnn)
    cnn = Reshape((-1, 256))(cnn)
    capsule = Capsule(classes, args.capsule_dim, args.routings, True)(cnn) #num capsule (classes), dim capsule, routings
    output = Lambda(lambda x: K.sqrt(K.sum(K.square(x), 2)), output_shape=(classes,))(capsule)

    model = Model(inputs=input_image, outputs=output)
    model.compile(loss=[margin_loss,'mse'],
                  optimizer='adam',
                  metrics=['accuracy'])

    model.summary()

    #train_x, train_y = val_generator

    start = time.time()

    model.fit_generator(train_generator,
            #batch_size=args.batch_size,
            epochs=args.epochs,
            verbose=1,
            validation_data=val_generator)

    end = time.time()

    training_time = end - start
    print('Training time: {}'.format(time.strftime("%H:%M:%S", time.gmtime(training_time))))

    """
    Y_pred = model.predict(X_test) #用模型进行预测
    greater = np.sort(Y_pred, axis=1)[:,-2] > 0.5 #判断预测结果是否大于0.5
    Y_pred = Y_pred.argsort()[:,-2:] #取最高分数的两个类别
    Y_pred.sort(axis=1) #排序，因为只比较集合
    
    acc = 1.*(np.prod(Y_pred == Y_test, axis=1)).sum()/len(X_test)
    #print u'CNN+Capsule，不考虑置信度的准确率为：%s'%acc
    print(u'CNN+Capsule：%s'%acc)
    acc = 1.*(np.prod(Y_pred == Y_test, axis=1)*greater).sum()/len(X_test)
    #print u'CNN+Capsule，考虑置信度的准确率为：%s'%acc
    print(u'CNN+Capsule：%s'%acc)
    """

    tf.keras.models.save_model(model, args.save_dir + '\\' + args.name+".h5")

    model.save(args.save_dir + '\\'+ args.name, save_format='tf')


    run_model = tf.function(lambda x : model(x))
    concrete_func = run_model.get_concrete_function(
        tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype)
    )

    #tf.saved_model.save(concrete_func, args.save_dir + '\\'+ args.name+ "_con_func",
    #                    signatures=run_model.get_concrete_function(
    #                        tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype, name="Capsule Network")))


    if(args.tflite):
        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        #converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
        tflite_model = converter.convert()

        with open(args.save_dir + '\\'+ args.name + "_conc_func"+ '.tflite', 'wb') as f:
          f.write(tflite_model)

        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        #converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
        tflite_model = converter.convert()

        with open(args.save_dir + '\\'+ args.name + '.tflite', 'wb') as f:
          f.write(tflite_model)