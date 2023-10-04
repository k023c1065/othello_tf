from network import *
import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
batch_size=32


def main():
    print("loading...",end="")
    with open("dataset/data.dat","rb") as f:
        dataset=pickle.load(f)
    print("Done")
    x,y=dataset[0],dataset[1]
    y=y.reshape(y.shape[0],64)
    y=tf.image.per_image_standardization(y)
    x_train,x_test=train_test_split(x,random_state=0)
    y_train,y_test=train_test_split(np.array(y).reshape(y.shape[0],64),random_state=0)
    test_ds = tf.data.Dataset.from_tensor_slices(
                (x_test,y_test)
            ).batch(64)
    train_ds = tf.data.Dataset.from_tensor_slices(
            (x_train, y_train)).shuffle(10000).batch(batch_size)
    model=ResNet((2,8,8),64)
    optimizer=tf.optimizers.Adam()
    loss_object=tf.keras.losses.categorical_crossentropy
    t_module=train_module(model,loss_object,optimizer)
    t_module.start_train(train_ds,test_ds)
    t_module.save_model()
    
    
    
if __name__=="__main__":
    main()   