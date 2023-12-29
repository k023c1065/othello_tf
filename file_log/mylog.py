import logging
import datetime,os
from multiprocessing import Lock
print(__file__,os.path.dirname(__file__))
if not os.path.dirname(__file__) == "":
    os.chdir(os.path.dirname(__file__))
if not "filename" in globals():
    filename=f"./log/{str(datetime.datetime.now())}.log".replace(" ","").replace(":","_")
logging.basicConfig(filename=filename,level=logging.INFO,
                    format='%(asctime)s %(message)s')
log_level=[]
lock=Lock()

class mylog:
    @classmethod
    def add_log(cls,msg,level=0):
        with lock:
            logging.info(msg)
    
    @classmethod
    def get_log_name(cls):
        return filename
    
    @classmethod
    def define_config(cls,filename):
        logging.basicConfig(filename=filename,level=logging.INFO,
                    format='%(asctime)s %(message)s')