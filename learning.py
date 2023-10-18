from network import *
import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd
from glob import glob
import math
model=None
optimizer=None
loss_object=None
def main(EPOCH=10,batch_size=16,input_shape=(8,8,2),t_module=None):
    global model,optimizer,loss_object
    print("loading...",end="")
    dataset_files=glob("./dataset/*.dat")
    dataset=None
    for file in dataset_files:
        try:
            with open(file,"rb") as f:
                data=pickle.load(f)
            if dataset is None:
                dataset=data
            else:
                dataset[0]=np.concatenate([dataset[0],data[0]])
                dataset[1]=np.concatenate([dataset[1],data[1]])
        except pickle.PickleError:
            print(f"Failed to pickle file:{file}. Skipping")
    print("Done")
    x,y=dataset[0],dataset[1]
    # y=y-y.mean()
    # y=tf.keras.layers.Rescaling(1.0/y.max())(y)
    x=np.array(x,dtype="float32")
    y=np.array(y,dtype="float32").reshape(y.shape[0],64)
    print("---Describe of Dataset---")
    print(pd.DataFrame(pd.Series(x[:min(len(x),30000)].ravel()).describe()).transpose())
    print(pd.DataFrame(pd.Series(np.array(y[:min(len(y),30000)],dtype="float32").reshape(min(y.shape[0],30000),64).ravel()).describe()).transpose())
    print("-------Describe End------")
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=random.randint(0,2**30))
    #=train_test_split(,random_state=0)
    print(x_train.shape)
    print(y_train.shape)
    test_batch_size=max(128,min(int(2**(int(math.log2(len(x_test)))-2)),2048))
    test_ds = tf.data.Dataset.from_tensor_slices(
                (x_test,y_test)
            ).batch(test_batch_size)
    train_ds = tf.data.Dataset.from_tensor_slices(
            (x_train, y_train)).shuffle(100000,reshuffle_each_iteration=True).batch(batch_size)
    model=miniResNet(input_shape,64)
    #model=ConvModel(input_shape,64)
    print(np.zeros(input_shape)[np.newaxis].shape)
    model.build(np.zeros(input_shape)[np.newaxis].shape)
    model.summary()
    if t_module is None:
        optimizer=tf.keras.optimizers.Adam()
        loss_object=tf.keras.losses.categorical_crossentropy
        t_module=train_module(model,loss_object,optimizer)
    try:
        t_module.start_train(train_ds,test_ds,EPOCH=EPOCH)
    except KeyboardInterrupt:
        #t_module.save_model()
        return t_module
    t_module.save_model()
    return t_module
    # for e in range(EPOCH):
    #     print("EPOCH:",e)
    #     flg=True
    #     for images,labels in tqdm(train_ds):
    #         if flg:
    #             flg=False
    #             #print(images.dtype,labels.dtype)
    #         loss=train_step(images,labels,loss_object)
    #     print("train loss:",np.mean(loss),end="     ")
    #     loss_array=[]
    #     flg=True
    #     for images,labels in test_ds:
    #         if flg:
    #             flg=False
    #             #print(images.shape,labels.shape)
    #         loss=test_step(model,images,labels)
    #         loss_array.append(np.mean(loss))
    #     print("test loss:",np.mean(np.array(loss_array)))
    # del optimizer,loss_object
    return [model,[x,y]]
    
if __name__=="__main__":
    main()   
    
    
@tf.function
def train_step(images, labels,loss_object):
    with tf.GradientTape() as tape:
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    try:
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    except KeyboardInterrupt:
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        raise KeyboardInterrupt()
    return loss


@tf.function
def test_step(model,images, labels):
    # training=False is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(images, training=False)
    t_loss = loss_object(labels, predictions)
    return t_loss