from dataset import dataset2tensor, loadDataset
from network import *
import tensorflow as tf
from sklearn.model_selection import train_test_split

model=None
optimizer=None
loss_object=None
def main(EPOCH=10,batch_size=16,input_shape=(8,8,2),t_module=None):
    global model,optimizer,loss_object
    dataset=loadDataset()
    train_ds,test_ds=dataset2tensor(dataset,batch_size)
    if t_module is None:
        model=miniResNet(input_shape,64)
        print(np.zeros(input_shape)[np.newaxis].shape)
        model.build(np.zeros(input_shape)[np.newaxis].shape)
        model.summary()
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
    
if __name__=="__main__":
    main()   
    
    