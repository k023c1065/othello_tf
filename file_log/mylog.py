import logging
import datetime,os
from multiprocessing import Lock
print(__file__,os.path.dirname(__file__))
if not os.path.dirname(__file__) == "":
    os.chdir(os.path.dirname(__file__))
log_level=[]
lock=Lock()
filename=None
class mylog:
    @classmethod
    def add_log(cls,msg,level=0):
        with lock:
            logging.info(msg)
    
    @classmethod
    def get_log_name(cls):
        return filename
    
    @classmethod
    def define_config(cls,fn):
        global filename
        filename=fn
        logging.basicConfig(filename=filename,level=logging.INFO,
                    format='%(asctime)s %(message)s')