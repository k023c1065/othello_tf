from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from glob import glob
import os,time,tqdm,threading
def split_array(ary,num):
    result=[]
    array_len=len(ary)//num
    for i in range(num):
        result.append(ary[i*array_len:(i+1)*array_len])
    result[-1]+=ary[(i+1)*array_len:]
    return result


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
        for name,f_id in self._FOLDER_ID.items():
            query=f'"{f_id}" in parents'
            f_list = self.fetch_flist(query=query)
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
        files = self.fetch_flist(query=query)
        for f in files:
            print("Downloading:",f["title"])
            f.GetContentFile("./dataset/"+f["title"])
    def fetch_flist(self,query):
        while True:
            try:
                file = self.drive.ListFile({'q':query}).GetList()
                return file
            except TimeoutError:
                time.sleep(3)
                pass
    def get_dataset_thread(self,thread_num=4):
        files=self.get_gdrive_list()
        f_list=dict()
        final_fnum=0
        for name,file in files.items():
            f_num=len(file)
            f_list[name] = [file[i*f_num:min((i+1)*f_num,len(file))] for i in range(thread_num//2)]
            final_fnum+=f_num
        th_array=[]
        self.finish_num=[0 for _ in range(thread_num*2)]
        tqdm_obj=tqdm.tqdm_notebook(range(final_fnum))
        thread_id=0
        for name,id in self._FOLDER_ID.items():
            query=f'"{id}" in parents'
            file = self.fetch_flist(query=query)
            file = split_array(file,thread_num)
            f_list[name]=file
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