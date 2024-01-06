from dataset import dataset2tensor, load_train_test_data, split_array
from network import *
import tensorflow as tf


def main(EPOCH=10, batch_size=16, input_shape=(8, 8, 2), t_module=None):
    train,test = load_train_test_data()
    
    # Spliting datasets to avoid memory overload
    max_dataset_size=512*512
    print("train len:",len(train[0]))
    if len(train[0])>max_dataset_size:
        num=len(train[0])//max_dataset_size+1
        train=[split_array(train[0],num)[0],split_array(train[1],num)[0]]
        test=[split_array(test[0],num)[0],split_array(test[1],num)[0]]

    train_ds = dataset2tensor(train,batch_size,True)
    del train
    test_ds = dataset2tensor(test, 1024,False)
    del test

    if t_module is None:
        model = miniResNet(input_shape, 64)
        print(np.empty(input_shape)[np.newaxis].shape)
        model.build(np.empty(input_shape)[np.newaxis].shape)
        mfs = glob.glob("./model/*.h5")
        if len(mfs) > 0:
            load_model_flg = input("Seems like we got some model file(s) in a model folder." +
                                   "Do you prefer to load the model file?(Y/n):\n")
            if load_model_flg.lower() == "y":
                print("loading:", mfs[0])
                model.load_weights(mfs[0])
        model.summary()
        optimizer = tf.keras.optimizers.Adam()
        loss_object = tf.keras.losses.categorical_crossentropy
        t_module = train_module(model, loss_object, optimizer)
    try:
        t_module.start_train(train_ds, test_ds, EPOCH=EPOCH)
    except KeyboardInterrupt:
        t_module.save_train()
        return t_module
    t_module.save_model()
    return t_module


if __name__ == "__main__":
    main()
