from concurrent.futures import thread
import datetime
import time
import numpy as np
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from glob import glob
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import random,pickle,math,os,threading,tqdm
from gdrive_lib import *

print(__file__,os.path.dirname(__file__))
if not os.path.dirname(__file__) == "":
    os.chdir(os.path.dirname(__file__))
os.makedirs("./dataset/test/",exist_ok=True)
os.makedirs("./dataset/train/",exist_ok=True)
def move2board(move,turn):
    a=np.zeros((8,8))
    a[move[0]][move[1]]=1
    return a
def split_datasets(buffer_size=2**12,file_num=None):
    dataset=loadDataset()
    print("split_datasets dataset len:",len(dataset[0]),len(dataset[1]))
    x_train,x_test,y_train,y_test=train_test_split(dataset[0],dataset[1],test_size=0.1,random_state=random.randint(0,2048))
    if file_num is None:
        x_train=np.array_split(x_train,1+len(x_train)//buffer_size)
        y_train=np.array_split(y_train,1+len(y_train)//buffer_size)
    else:
        x_train=np.array_split(x_train,file_num)
        y_train=np.array_split(y_train,file_num)
    for i in range(len(x_train)):
        d=str(datetime.datetime.now()).replace(" ","_").replace(":","-")
        with open(f"./dataset/train/train_{d}.dat","wb") as f:
            pickle.dump([x_train[i],y_train[i]],f)
    if file_num is None:
        x_test=np.array_split(x_test,1+len(x_test)//buffer_size)
        y_test=np.array_split(y_test,1+len(y_test)//buffer_size)
    else:
        x_test=np.array_split(x_test,file_num)
        y_test=np.array_split(y_test,file_num)
    for i in range(len(x_test)):
        d=str(datetime.datetime.now()).replace(" ","_").replace(":","-")
        with open(f"./dataset/test/test_{d}.dat","wb") as f:
            pickle.dump([x_test[i],y_test[i]],f)
    temp_files=glob("dataset/*.dat")
    for f in temp_files:
        os.remove(f)
    return x_train,y_train,x_test,y_test
def get_dataset_num():
    print("loading...",end="")
    dataset_files=glob("./dataset/*.dat")
    dataset=None
    if len(dataset_files)<1:
        raise FileNotFoundError("Files that matched the pattern seems to be none.\n",
                                "Please check dataset folder.")
    num=0
    for file in tqdm(dataset_files):
        try:
            with open(file,"rb") as f:
                data=pickle.load(f)
            num+=len(data[0])
            del data
        except pickle.PickleError:
            print(f"Failed to pickle file:{file}. Skipping")
    if dataset is None:
        raise FileNotFoundError("It seems all dataset file(s) have been corruppted\n",
                                "Please Check a dataset folder for more info.",
                                f"dataset_files:{dataset_files}")
    print("Done")
    return num
def load_train_test_data():
    
    train_files=glob("./dataset/train/*.dat")
    if len(train_files)<1:
        raise FileNotFoundError("Files that matched the pattern seems to be none.\n",
                                "Please check dataset folder.")
    train_dataset=None
    for file in tqdm.tqdm_notebook(train_files):
        try:
            with open(file,"rb") as f:
                data=pickle.load(f)
            if train_dataset is None:
                train_dataset=data
            else:
                train_dataset[0] = np.concatenate([train_dataset[0],data[0]])
                train_dataset[1] = np.concatenate([train_dataset[1],data[1]])
        except pickle.PickleError:
            print(f"Failed to pickle file:{file}. Skipping")
        except EOFError:
            print(f"Failed to pickle file:{file}. Skipping")    
    
    test_files=glob("./dataset/test/*.dat")
    if len(test_files)<1:
        raise FileNotFoundError("Files that matched the pattern seems to be none.\n",
                                "Please check dataset folder.")
    test_dataset=None
    for file in tqdm.tqdm_notebook(test_files):
        try:
            with open(file,"rb") as f:
                data=pickle.load(f)
            if test_dataset is None:
                test_dataset=data
            else:
                test_dataset[0] = np.concatenate([test_dataset[0],data[0]])
                test_dataset[1] = np.concatenate([test_dataset[1],data[1]])
        except pickle.PickleError:
            print(f"Failed to pickle file:{file}. Skipping")
        except EOFError:
            print(f"Failed to pickle file:{file}. Skipping")
    return train_dataset,test_dataset
def loadDataset():
    print("loading...",end="")
    dataset_files=glob("./dataset/*.dat")
    dataset=None
    if len(dataset_files)<1:
        raise FileNotFoundError("Files that matched the pattern seems to be none.\n",
                                "Please check dataset folder.")
    random.shuffle(dataset_files)
    for file in dataset_files:
        try:
            with open(file,"rb") as f:
                try:
                    data=pickle.load(f)
                except EOFError:
                    pass
            if dataset is None:
                dataset=data
            else:
                dataset[0] = np.concatenate([dataset[0],data[0]])
                dataset[1] = np.concatenate([dataset[1],data[1]])
        except pickle.PickleError:
            print(f"Failed to pickle file:{file}. Skipping")
    if dataset is None:
        raise FileNotFoundError("It seems all dataset file(s) have been corruppted\n",
                                "Please Check a dataset folder for more info.",
                                f"dataset_files:{dataset_files}")
    print("Done")
    return dataset

def dataset2tensor(dataset,batch_size,do_shuffle):
    x,y=dataset[0],dataset[1]
    x=np.array(x,dtype="float32")
    y=np.array(y,dtype="float32").reshape(y.shape[0],64)
    print("---Describe of Dataset---")
    print(pd.DataFrame(pd.Series(x[:min(len(x),30000)].ravel()).describe()).transpose())
    print(pd.DataFrame(pd.Series(np.array(y[:min(len(y),30000)],dtype="float32").reshape(min(y.shape[0],30000),64).ravel()).describe()).transpose())
    print("-------Describe End------")
    batch_size=max(128,min(int(2**(int(math.log2(len(x)))-2)),2048))
    if do_shuffle:
        ds=tf.data.Dataset.from_tensor_slices(
            (x, y)
            ).shuffle(25000,reshuffle_each_iteration=True,seed=random.randint(0,2**32)).batch(batch_size)
    else:
        ds = tf.data.Dataset.from_tensor_slices(
                (x,y)
            ).batch(batch_size)
    return ds

def split_array(ary:list|np.ndarray,num):
    result=[]
    array_len=len(ary)//num
    is_ndarray = type(ary) == np.ndarray
    if is_ndarray:
        dtype=ary.dtype
    ary=list(ary)
    for i in range(num):
        result.append(ary[i*array_len:(i+1)*array_len])
    result[-1]+=ary[(i+1)*array_len:]
    if is_ndarray:
        result = np.array(result,dtype=dtype)
    return result




