import tensorflow as tf
import keras
from keras import layers as kl
import numpy as np
import cv2
import glob
import os
from tqdm import tqdm
from datetime import datetime
def raw_load_model():
    folders=glob.glob("./mymodel/*")
    folder=folders[-1]
    model=keras.models.load_model(folder)
    return model
class ConvModel(keras.Model):
    def __init__(self):
        self.file_path="./model/mymodel"
        super(ConvModel, self).__init__()
        self.conv_0=[kl.Conv2D(4,3,activation="relu",padding="same") for i in range(2)]
        self.pooling_1=kl.AveragePooling2D()
        self.conv_1=[kl.Conv2D(8,3,activation="relu",padding="same") for i in range(2)]
        self.pooling_2=kl.AveragePooling2D()
        self.conv_2=[kl.Conv2D(16,3,activation="relu",padding="same") for i in range(2)]
        self.pooling_3=kl.AveragePooling2D()
        self.conv_3=[kl.Conv2D(32,3,activation="relu",padding="same") for i in range(4)]
        self.pooling_4=kl.AveragePooling2D()
        self.flatten=kl.GlobalAveragePooling2D()
        self.dense_0=[kl.Dense(128,activation="relu"),kl.Dense(64,activation="softmax")]
    def call(self,inputs,training=False,debug=False):
        x=inputs
        for l in self.conv_0:
            x=l(x)
            if debug:
                print(x.shape)
        x=self.pooling_1(x)
        if debug:
            print(x.shape)
        for l in self.conv_1:
            x=l(x)
            if debug:
                print(x.shape)
        x=self.pooling_2(x)
        if debug:
            print(x.shape)
        for l in self.conv_2:
            x=l(x)
            if debug:
                print(x.shape)
        x=self.pooling_3(x)
        if debug:
            print(x.shape)
        x=self.flatten(x)
        for l in self.dense_0:
            x=l(x)
        return x

# 残差ブロック(Bottleneckアーキテクチャ)
class Res_Block(tf.keras.Model):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        bneck_channels = out_channels // 4

        self.bn1 = kl.BatchNormalization()
        self.av1 = kl.Activation(tf.nn.relu)
        self.conv1 = kl.Conv2D(bneck_channels, kernel_size=1,
                                strides=1, padding='valid', use_bias=False)

        self.bn2 = kl.BatchNormalization()
        self.av2 = kl.Activation(tf.nn.relu)
        self.conv2 = kl.Conv2D(bneck_channels, kernel_size=3,
                                strides=1, padding='same', use_bias=False)

        self.bn3 = kl.BatchNormalization()
        self.av3 = kl.Activation(tf.nn.relu)
        self.conv3 = kl.Conv2D(out_channels, kernel_size=1,
                                strides=1, padding='valid', use_bias=False)

        self.shortcut = self._scblock(in_channels, out_channels)
        self.add = kl.Add()

    # Shortcut Connection
    def _scblock(self, in_channels, out_channels):
        if in_channels != out_channels:
            self.bn_sc1 = kl.BatchNormalization()
            self.conv_sc1 = kl.Conv2D(out_channels, kernel_size=1,
                                        strides=1, padding='same', use_bias=False)
            return self.conv_sc1
        else:
            return lambda x: x

    def call(self, x,training):
        out1 = self.conv1(self.av1(self.bn1(x,training)))
        out2 = self.conv2(self.av2(self.bn2(out1,training)))
        out3 = self.conv3(self.av3(self.bn3(out2,training)))
        shortcut = self.shortcut(x)
        out4 = self.add([out3, shortcut])
        return out4


# ResNet(Pre Activation)
class ResNet(tf.keras.Model):
    def __init__(self, input_shape, output_dim):
        super().__init__()
        self._kl = [
            kl.BatchNormalization(),
            kl.Activation(tf.nn.relu),
            kl.Conv2D(16, kernel_size=7, strides=2, padding="same", use_bias=False, input_shape=input_shape),
            kl.MaxPool2D(pool_size=3, strides=2, padding="same"),
            Res_Block(16, 32),
            [
                Res_Block(32, 32) for _ in range(1)
            ],
            kl.Conv2D(64, kernel_size=1, strides=2),
            [
                Res_Block(64, 64) for _ in range(2)
            ],
            kl.Conv2D(64, kernel_size=1, strides=2, use_bias=False),
            [
                Res_Block(64, 64) for _ in range(2)
            ],
            kl.Conv2D(128, kernel_size=1, strides=2, use_bias=False),
            [
                Res_Block(128, 128) for _ in range(2)
            ],
            kl.GlobalAveragePooling2D(),
            kl.Dense(128, activation="relu"),
            kl.Dense(output_dim, activation="softmax")
        ]
    def call(self, x, training=True):
        for layer in self._kl:
            if isinstance(layer, list):
                for _layer in layer:
                    x = _layer(x,training)
            else:
                if type(layer)==kl.BatchNormalization:
                    x = layer(x,training)
                else:
                    x = layer(x)
        return x

def fix_data(data,isDataset=False):
    data=cv2.resize(data,dsize=(224,224),interpolation=cv2.INTER_LINEAR)
    data=cv2.cvtColor(data,cv2.COLOR_BGRA2GRAY)
    data=np.array(data,dtype="float32")
    data/=255
    return data

class train_module():
    def __init__(self,model,loss_object,optimizer):
        self.model=model
        self.loss_object=loss_object
        self.optimizer=optimizer
    @tf.function
    def train_step(self,images, labels):
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = self.model(images, training=True)
            loss = self.loss_object(labels, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        try:
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        except KeyboardInterrupt:
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            raise KeyboardInterrupt()
        return loss

    @tf.function
    def test_step(self,images, labels):
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = self.model(images, training=False)
        t_loss = self.loss_object(labels, predictions)
        return t_loss
    
    def start_train(self,train_ds,test_ds,EPOCH=10):
        for e in range(EPOCH):
            print("EPOCH:",e)
            for images,labels in tqdm(train_ds):
                loss=self.train_step(images,labels)
            print("train loss:",np.mean(loss),end="   ")
            loss_array=[]
            for images,labels in test_ds:
                loss=self.test_step(images,labels)
                loss_array.append(loss)
            print("test loss:",np.mean(loss_array))
    def save_model(self,model_path="./model/"):
        self.model.save_weights(model_path+str(datetime.now())+".h5")
            

                