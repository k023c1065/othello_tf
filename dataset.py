import numpy as np
import pydrive2
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from glob import glob
import os
def move2board(move,turn):
    a=np.zeros((8,8))
    a[move[0]][move[1]]=1
    return a

class gdrive_dataset():
    #Will make this folder public in futre
    FOLDER_ID="1ooAjVn2MUGs4fy2FK4wLMK24bm0EfqaJ"
    def __init__(self):
        gauth = GoogleAuth()
        gauth.CommandLineAuth()
        self.drive = GoogleDrive(gauth)
    def get_dataset_list(self):
        files=glob("dataset/*.dat")
        files=[os.path.split(f)[1] for f in files]
        return files
    def get_gdrive_list(self):
        query=f'"{self.FOLDER_ID}" in parents'
        files = self.drive.ListFile({'q':query}).GetList()
        files=[f["title"] for f in files]
        return files
    def transfer_dataset(self):
        dataset=self.get_dataset_list()
        gdrive_d=self.get_gdrive_list()
        for d in dataset:
            if not d in gdrive_d:
                file=self.drive.CreateFile({"parents": [{"id": self.FOLDER_ID}]})
                file.SetContentFile("dataset/"+d)
                file.upload()

