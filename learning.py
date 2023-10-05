from network import *
import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd

def main(EPOCH=10,batch_size=32):
    print("loading...",end="")
    with open("dataset/data.dat","rb") as f:
        dataset=pickle.load(f)
    print("Done")
    x,y=dataset[0],dataset[1]
    # y=y-y.mean()
    # y=tf.keras.layers.Rescaling(1.0/y.max())(y)
    x=np.array(x,dtype="uint8")
    print("---Describe of Dataset---")
    print(pd.DataFrame(pd.Series(x.ravel()).describe()).transpose())
    print(pd.DataFrame(pd.Series(np.array(y,dtype="float32").reshape(y.shape[0],64).ravel()).describe()).transpose())
    print("-------Describe End------")
    x_train,x_test,y_train,y_test=train_test_split(x,np.array(y,dtype="float32").reshape(y.shape[0],64),test_size=0.25,random_state=0)
    #=train_test_split(,random_state=0)
    test_ds = tf.data.Dataset.from_tensor_slices(
                (x_test,y_test)
            ).batch(64)
    train_ds = tf.data.Dataset.from_tensor_slices(
            (x_train, y_train)).shuffle(10000).batch(batch_size)
    model=ResNet((64,64,2),64)
    optimizer=tf.optimizers.Adam()
    loss_object=tf.keras.losses.mean_squared_error
    t_module=train_module(model,loss_object,optimizer)
    t_module.start_train(train_ds,test_ds,EPOCH=EPOCH)
    t_module.save_model()
    
    return t_module
    
if __name__=="__main__":
    main()   