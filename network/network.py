import tensorflow as tf
import keras
from keras import layers as kl
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from sklearn.linear_model import LinearRegression
import tqdm as tq
from datetime import datetime
import sys
import random

moduleList = sys.modules
ENV_COLAB = False
if 'google.colab' in moduleList:
    print("Is google_colab")
    ENV_COLAB = True
    tqdm = tq.tqdm_notebook
else:
    tqdm = tq.tqdm
    print("Not google_colab")


def raw_load_model(file_name=None, get_filename=False):
    model = miniResNet((8, 8, 2), 64)
    model(np.empty((1, 8, 8, 2)))
    try:
        model_files = glob.glob("./network/model/*")
        model_file = max(model_files, key=os.path.getctime)
        if type(file_name) is str:
            model_file = file_name
        model.load_weights(model_file)
    except FileNotFoundError:
        raise FileNotFoundError("failed to get Model file(s) \n",
                                f"Expected File:{model_file}"
                                f"targeted Files:{model_files}")
    if get_filename:
        return model, model_file
    return model


class ConvModel(keras.Model):
    def __init__(self, inp_ch, out_ch):
        self.file_path = "./model/mymodel"
        super(ConvModel, self).__init__()
        self.conv_0 = [
            kl.Conv2D(4, 3, activation="relu", padding="same") for i in range(2)]
        self.pooling_1 = kl.AveragePooling2D()
        self.conv_1 = [
            kl.Conv2D(8, 3, activation="relu", padding="same") for i in range(2)]
        self.pooling_2 = kl.AveragePooling2D()
        self.conv_2 = [
            kl.Conv2D(16, 3, activation="relu", padding="same") for i in range(2)]
        self.pooling_3 = kl.AveragePooling2D()
        self.conv_3 = [
            kl.Conv2D(32, 3, activation="relu", padding="same") for i in range(4)]
        self.pooling_4 = kl.AveragePooling2D()
        self.flatten = kl.GlobalAveragePooling2D()
        self.dense_0 = [kl.Dense(128, activation="relu"), kl.Dense(
            out_ch, activation="softmax")]

    def call(self, inputs, training=False, debug=False):
        x = inputs
        for l in self.conv_0:
            x = l(x)
            if debug:
                print(x.shape)
        x = self.pooling_1(x)
        if debug:
            print(x.shape)
        for l in self.conv_1:
            x = l(x)
            if debug:
                print(x.shape)
        x = self.pooling_2(x)
        if debug:
            print(x.shape)
        for l in self.conv_2:
            x = l(x)
            if debug:
                print(x.shape)
        x = self.pooling_3(x)
        if debug:
            print(x.shape)
        x = self.flatten(x)
        for l in self.dense_0:
            x = l(x)
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

    def call(self, x, training):
        out1 = self.conv1(self.av1(self.bn1(x, training)))
        out2 = self.conv2(self.av2(self.bn2(out1, training)))
        out3 = self.conv3(self.av3(self.bn3(out2, training)))
        shortcut = self.shortcut(x)
        out4 = self.add([out3, shortcut])
        return out4


# ResNet(Pre Activation)
class ResNet(tf.keras.Model):
    def __init__(self, input_shape, output_dim):
        super().__init__()
        self._kl = [
            # kl.BatchNormalization(),
            kl.Activation(tf.nn.relu),
            kl.Conv2D(16, kernel_size=7, strides=2, padding="same",
                      use_bias=False, input_shape=input_shape),
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
                Res_Block(64, 64) for _ in range(3)
            ],
            kl.Conv2D(128, kernel_size=1, strides=2, use_bias=False),
            [
                Res_Block(128, 128) for _ in range(3)
            ],
            kl.GlobalAveragePooling2D(),
            kl.Dense(256, activation="relu"),
            kl.Dense(output_dim, activation="softmax")
        ]

    def call(self, x, training=True, isDebug=False):
        for layer in self._kl:
            if isinstance(layer, list):
                for _layer in layer:
                    x = _layer(x, training)
                    if isDebug:
                        print(_layer.name, x.shape, np.min(np.array(x)), np.max(
                            np.array(x)), np.mean(np.array(x)), np.std(np.array(x)))
            else:
                if type(layer) == kl.BatchNormalization:
                    x = layer(x, training)
                    if isDebug:
                        print(layer.name, x.shape, np.min(np.array(x)), np.max(
                            np.array(x)), np.mean(np.array(x)), np.std(np.array(x)))
                else:
                    x = layer(x)
                    if isDebug:
                        print(layer.name, x.shape, np.min(np.array(x)), np.max(
                            np.array(x)), np.mean(np.array(x)), np.std(np.array(x)))
        return x

class value_resNet(tf.keras.Model):
    def __init__(self, input_shape, output_dim):
        self.init_input_shape = input_shape
        super().__init__()
        self._kl = [
            kl.BatchNormalization(),
            kl.Activation(tf.nn.relu),
            kl.Conv2D(32, kernel_size=3, strides=(1, 1),
                      padding="same", activation="relu"),
            [Res_Block(32, 32) for _ in range(2)],
            kl.GlobalAveragePooling2D(),
            kl.Dense(64, activation="relu"),
            kl.Dense(output_dim, activation="softmax")
        ]
    def call(self, x, training=True, isDebug=False):
        try:
            assert (self.init_input_shape == x.shape[1:])
        except AssertionError:
            raise AssertionError(f"Seems like input shape differs from init one.\n",
                                 f"init shape:{self.init_input_shape}",
                                 f"input shape:{x.shape[1:]}")
        for layer in self._kl:
            if isinstance(layer, list):
                for _layer in layer:
                    x = _layer(x, training)
                    if isDebug:
                        print(_layer.name, x.shape, np.min(np.array(x)), np.max(
                            np.array(x)), np.mean(np.array(x)), np.std(np.array(x)))
            else:
                if type(layer) == kl.BatchNormalization:
                    x = layer(x, training)
                    if isDebug:
                        print(layer.name, x.shape, np.min(np.array(x)), np.max(
                            np.array(x)), np.mean(np.array(x)), np.std(np.array(x)))
                else:
                    x = layer(x)
                    if isDebug:
                        print(layer.name, x.shape, np.min(np.array(x)), np.max(
                            np.array(x)), np.mean(np.array(x)), np.std(np.array(x)))
        return x
class miniResNet(tf.keras.Model):
    def __init__(self, input_shape, output_dim):
        self.init_input_shape = input_shape
        super().__init__()
        self._kl = [
            kl.BatchNormalization(),
            kl.Activation(tf.nn.relu),
            kl.Conv2D(256, kernel_size=3, strides=(1, 1),
                      padding="same", activation="relu"),
            [Res_Block(256, 256) for _ in range(19)],
            kl.GlobalAveragePooling2D(),
            kl.Dense(512, activation="relu"),
            kl.Dense(output_dim, activation="softmax")
        ]

    def call(self, x, training=True, isDebug=False):
        try:
            assert (self.init_input_shape == x.shape[1:])
        except AssertionError:
            raise AssertionError(f"Seems like input shape differs from init one.\n",
                                 f"init shape:{self.init_input_shape}",
                                 f"input shape:{x.shape[1:]}")
        for layer in self._kl:
            if isinstance(layer, list):
                for _layer in layer:
                    x = _layer(x, training)
                    if isDebug:
                        print(_layer.name, x.shape, np.min(np.array(x)), np.max(
                            np.array(x)), np.mean(np.array(x)), np.std(np.array(x)))
            else:
                if type(layer) == kl.BatchNormalization:
                    x = layer(x, training)
                    if isDebug:
                        print(layer.name, x.shape, np.min(np.array(x)), np.max(
                            np.array(x)), np.mean(np.array(x)), np.std(np.array(x)))
                else:
                    x = layer(x)
                    if isDebug:
                        print(layer.name, x.shape, np.min(np.array(x)), np.max(
                            np.array(x)), np.mean(np.array(x)), np.std(np.array(x)))
        return x


class train_module():
    def __init__(self, model, loss_object, optimizer, input_shape=(8, 8, 2),
                 tester=None,test_round=5,test_against="Random",test_game_count=500):
        if isinstance(model, str):
            self.model = ResNet(input_shape)
            self.model.load_weights(model)
        else:
            self.model = model
        self.loss_object = loss_object
        self.optimizer = optimizer
        self.last_train = None
        self.train_count = 0
        self.train_loss = []
        self.test_loss = [[], []]
        self.test_loss_unc = []
        self.plt_file_name = "loss_graph.png"
        
        # Testing params
        self.tester=tester
        self.test_round=test_round
        self.test_against=test_against
        self.game_count=test_game_count

    @tf.function
    def train_step(self, images, labels):
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = self.model(images, training=True)
            loss = self.loss_object(labels, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        try:
            self.optimizer.apply_gradients(
                zip(gradients, self.model.trainable_variables))
        except KeyboardInterrupt:
            self.optimizer.apply_gradients(
                zip(gradients, self.model.trainable_variables))
            raise KeyboardInterrupt()
        return loss

    @tf.function
    def test_step(self, images, labels):
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = self.model(images, training=False)
        t_loss = self.loss_object(labels, predictions)
        return t_loss

    def start_train(self, train_ds, test_ds, EPOCH=10, skip_rate=0.001):
        print("last train:", self.last_train)
        skip_rate_cap = 1//skip_rate
        best_test_loss = 3.9  # Probably the baseline
        best_model = None
        for e in range(EPOCH):
            self.last_train = datetime.now()
            print("EPOCH:", e)
            loss_array = []
            skip_count = 0
            with tqdm(train_ds) as t:
                for images, labels in t:
                    if not random.randint(1, skip_rate_cap) == 1:
                        loss = self.train_step(images, labels)
                        self.train_count += 1
                        self.train_loss.append(np.mean(np.array(loss)))
                        loss_array.append(np.mean(np.array(loss)))
                        t.set_description(
                            f"loss:{np.round(float(np.mean(loss_array)),decimals=5)}")
                    else:
                        skip_count += 1
                    t.update()
            print("skip_count:", skip_count, end="    ")
            print("train loss:", float(np.mean(loss_array)), end="")
            loss_array = []
            for images, labels in test_ds:
                loss = self.test_step(images, labels)
                loss_array.append(np.mean(loss))

            self.test_loss_unc.append(
                np.std(np.array(loss_array))/np.sqrt(len(loss_array)))
            self.test_loss[1].append(np.mean(np.array(loss_array)))
            self.test_loss[0].append(self.train_count)
            test_inc = [get_linear_inc(self.test_loss[0][-int(len(self.test_loss[0])*(0.5**i)):],
                                       self.test_loss[1][-int(len(self.test_loss[0])*(0.5**i)):])
                        if int(len(self.test_loss[0])*(0.5**i)) > 1 else 0 for i in range(3)]
            print("\t test loss:", np.mean(np.array(loss_array)),
                  "±", self.test_loss_unc[-1])
            for i in range(3):
                print("\t", end="")
                if test_inc[i] > 0:
                    print('\033[31m', end="")
                print(f" {100*(0.5**i)}%:{test_inc[i]}", end="")
                if test_inc[i] > 0:
                    print('\033[0m', end="")
                print()
            self.save_fig()
            self.save_model()
            if (self.test_loss[1][-1] < best_test_loss):
                best_test_loss=self.test_loss[1][-1]
                best_model=tf.keras.models.clone_model(self.model)
                #self.save_best_model()
            if self.tester is not None:
                if e+1%self.test_round==0:
                    if self.test(best_model):
                        self.save_best_model(arg_model=best_model)
                        print("Created ideal model,Exiting...")
                        break
            self.save_train()
    def test(self,model):
        using_model=[model,None]
        if self.test_against == "Model":
            baseline_model=raw_load_model("baseline.h5")
            using_model[1]=baseline_model
        players=["Model",self.test_against]
        test_module=self.tester(players=players,model=using_model,game_count=self.game_count,DoShuffle=True)
        result=test_module.loop_game()
        print("Test play result:",result)
        if result[0]/(result[0]+result[1])>=0.55:
            return True
        else:
            return False
    def save_fig(self):
        train_loss = get_moving_ave(self.train_loss)
        plt.plot([i for i in range(len(self.train_loss))],
                 train_loss, label="train")
        plt.plot(self.test_loss[0], self.test_loss[1], label="test", marker="o")
        plt.grid()
        plt.legend()
        plt.savefig(self.plt_file_name)
        plt.close()

    def save_model(self, model_path="./model/"):
        model_file_name = str(datetime.now())+".h5"
        self.model.save_weights(model_path+model_file_name)
        if ENV_COLAB:
            self.model.save_weights("./drive/MyDrive/model/"+model_file_name)
        print("model saved as ", model_file_name)

    def save_best_model(self, model_path="./model/",arg_model = None):
        model_file_name = "best_model.h5"
        if arg_model is None:
            model=self.model
        else:
            model=arg_model
        model.save_weights(model_path+model_file_name)
        if ENV_COLAB:
            model.save_weights("./drive/MyDrive/model/"+model_file_name)
        print("Best model has been updated")

    def save_train(self, path="./model/"):
        self.model.save(path+"latest_model", overwrite=True, save_format="tf")
        if ENV_COLAB:
            self.model.save("./drive/MyDrive/model/latest_model")


def get_linear_inc(x, y):
    lr = LinearRegression()
    lr.fit(np.array(x).reshape(-1, 1), np.array(y).reshape(-1, 1))
    return lr.coef_[0][0]


def get_moving_ave(data, b=100, result=None):
    length = len(data)
    temp = []
    if result is None:
        result = []
    init_len = len(result)
    for d in data[init_len:]:
        i = len(result)
        temp.append(d)
        if i >= b:
            temp.pop(0)
        result.append(np.mean(temp))
    return result


if __name__ == "__main__":
    model=miniResNet((8,8,2),64)
    print(model(np.empty((1,8,8,2))))