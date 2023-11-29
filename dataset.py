from concurrent.futures import thread
import datetime
import numpy as np
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from glob import glob
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import random,pickle,math,os,threading,tqdm

os.makedirs("./dataset/test/",exist_ok=True)
os.makedirs("./dataset/train/",exist_ok=True)

def move2board(move,turn):
    a=np.zeros((8,8))
    a[move[0]][move[1]]=1
    return a
def split_datasets(buffer_size=2**12):
    dataset=loadDataset()
    x_train,x_test,y_train,y_test=train_test_split(dataset[0],dataset[1],test_size=0.1,random_state=random.randint(0,2048))
    x_train=np.split(x_train,1+len(x_train)//buffer_size)
    y_train=np.split(y_train,1+len(x_train)//buffer_size)
    x_test=np.split(x_test,1+len(x_test)//buffer_size)
    y_test=np.split(y_test,1+len(y_test)//buffer_size)
    return x_train,y_train,x_test,y_test
def get_dataset_num():
    print("loading...",end="")
    dataset_files=glob("./dataset/*.dat")
    dataset=None
    if len(dataset_files)<1:
        raise FileNotFoundError("Files that matched the pattern seems to be none.\n",
                                "Please check dataset folder.")
    num=0
    for file in dataset_files:
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
def loadDataset():
    print("loading...",end="")
    dataset_files=glob("./dataset/*.dat")
    dataset=None
    if len(dataset_files)<1:
        raise FileNotFoundError("Files that matched the pattern seems to be none.\n",
                                "Please check dataset folder.")
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
    if dataset is None:
        raise FileNotFoundError("It seems all dataset file(s) have been corruppted\n",
                                "Please Check a dataset folder for more info.",
                                f"dataset_files:{dataset_files}")
    print("Done")
    return dataset
def dataset2tensor(dataset,batch_size):
    x,y=dataset[0],dataset[1]
    x=np.array(x,dtype="float32")
    y=np.array(y,dtype="float32").reshape(y.shape[0],64)
    print("---Describe of Dataset---")
    print(pd.DataFrame(pd.Series(x[:min(len(x),30000)].ravel()).describe()).transpose())
    print(pd.DataFrame(pd.Series(np.array(y[:min(len(y),30000)],dtype="float32").reshape(min(y.shape[0],30000),64).ravel()).describe()).transpose())
    print("-------Describe End------")
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=random.randint(0,2048))
    print(x_train.shape)
    print(y_train.shape)
    test_batch_size=max(128,min(int(2**(int(math.log2(len(x_test)))-2)),2048))
    test_ds = tf.data.Dataset.from_tensor_slices(
                (x_test,y_test)
            ).batch(test_batch_size)
    train_ds = tf.data.Dataset.from_tensor_slices(
            (x_train, y_train)).shuffle(100000,reshuffle_each_iteration=True).batch(batch_size)
    return train_ds,test_ds
class gdrive_dataset():
    #Will make this folder public in future
    FOLDER_ID="1ooAjVn2MUGs4fy2FK4wLMK24bm0EfqaJ"
    _FOLDER_ID={"test":"1aVYYViqx4_XA-gVtsVWasX4Q7fNjc56y","train":"1MUqNwie4D_ceNNCe3e-7VD4dQXkb2kAJ"}
    def __init__(self):
        gauth = GoogleAuth(settings_file='setting.yaml')
        gauth.CommandLineAuth()
        self.drive = GoogleDrive(gauth)
    def get_dataset_list(self):
        files={"train":[os.path.split(f)[1] for f in glob("dataset/train/*.dat")],"test":[os.path.split(f)[1] for f in glob("dataset/test/*.dat")]}
        return files
    def get_gdrive_list(self):
        files={}
        for name,f_id in self._FOLDER_ID:
            query=f'"{f_id} in parents'
            f_list = self.drive.ListFile({'q':query}).GetList()
            files[name]=[f["title"] for f in f_list]
        return files
    def transfer_dataset(self):
        """
            Transfers all dataset into Google Drive
        """        
        dataset=self.get_dataset_list()
        gdrive_d=self.get_gdrive_list()
        for n,data in dataset.items():
            for d in data:
                if not d in gdrive_d[n]:
                    file=self.drive.CreateFile({"parents": [{"id": self._FOLDER_ID[n]}]})
                    file.SetContentFile(f"dataset/{n}/"+d)
                    print("Uploading:",d)
                    file.Upload()
    def _get_dataset(self):
        """
            Gets all dataset from Google Drive
        """        
        query=f'"{self.FOLDER_ID}" in parents'
        files = self.drive.ListFile({'q':query}).GetList()
        for f in files:
            print("Downloading:",f["title"])
            f.GetContentFile("./dataset/"+f["title"])
    
    def get_dataset_thread(self,thread_num=4):
        files=self.get_gdrive_list()
        f_list=dict()
        final_fnum=0
        for name,file in files.items():
            f_num=len(file)
            f_list[name] = [file[i*f_num:min((i+1)*f_num,len(file))] for i in range(thread_num//2)]
            final_fnum+=fnum
        th_array=[]
        self.finish_num=[0 for _ in range(thread_num)]
        tqdm_obj=tqdm.tqdm_notebook(range(final_fnum))
        thread_id=0
        for name,files in f_list.items():
            for file in files:
                th_array.append(threading.Thread(target=self._files_getter,args=(file,name,thread_id)))
                thread_id+=1
        for th in th_array:
            th.start()
        while True in [t.is_alive() for t in th_array]:
            fnum=sum(self.finish_num)
            tqdm_obj.update(fnum-tqdm_obj.n)
        for th in th_array:
            th.join()
        tqdm_obj.close()
    def _files_getter(self,files,name,t_n):
        for f in files:
            Download_flg=False
            Fail_count=0
            while (not Download_flg) and Fail_count<10:
                try:
                    f.GetContentFile(f"./dataset/{name}/"+f["title"])
                except Exception:
                    Fail_count+=1
                else:
                    Download_flg=True
            if Fail_count>=10:
                print("Download failed:",f["title"])
            self.finish_num[t_n]+=1


