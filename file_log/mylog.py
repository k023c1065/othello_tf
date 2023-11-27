import logging
import datetime,os
from multiprocessing import Lock
filename=f"./log/{str(datetime.datetime.now())}.log".replace(" ","").replace(":","_")
logging.basicConfig(filename=filename,level=logging.INFO,
                    format='%(asctime)s %(message)s')
log_level=[]
lock=Lock()

class mylog:
    if not os.path.dirname(__file__) == "":
        os.chdir(os.path.dirname(__file__))
    @classmethod
    def add_log(cls,msg,level=0):
        with lock:
            logging.info(msg)