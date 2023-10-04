from network import *
import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split


def main(EPOCH=10,batch_size=32):
    print("loading...",end="")
    with open("dataset/data.dat","rb") as f:
        dataset=pickle.load(f)
    print("Done")
    x,y=dataset[0],dataset[1]
    y=tf.keras.layers.Rescaling(1.0/y.max())(y)
    x=np.array(x,dtype="uint8")
    x_train,x_test,y_train,y_test=train_test_split(x,np.array(y).reshape(y.shape[0],64),test_size=0.2,random_state=0)
    #=train_test_split(,random_state=0)
    test_ds = tf.data.Dataset.from_tensor_slices(
                (x_test,y_test)
            ).batch(64)
    train_ds = tf.data.Dataset.from_tensor_slices(
            (x_train, y_train)).shuffle(10000).batch(batch_size)
    model=ResNet((2,8,8),64)
    optimizer=tf.optimizers.Adam()
    loss_object=tf.keras.losses.categorical_crossentropy
    t_module=train_module(model,loss_object,optimizer)
    t_module.start_train(train_ds,test_ds,EPOCH=EPOCH)
    t_module.save_model()
    
    return t_module
    
if __name__=="__main__":
    main()   