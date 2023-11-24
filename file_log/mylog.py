import logging
import datetime,os
from multiprocessing import Lock
class mylog:
    if not os.path.dirname(__file__) == "":
        os.chdir(os.path.dirname(__file__))
    filename=f"./log/{str(datetime.datetime.now())}.log".replace(" ","").replace(":","_")
    logging.basicConfig(filename=filename,
                        format='%(asctime)s %(message)s')
    log_level=[]
    lock=Lock()
    @classmethod
    def add_log(cls,msg,level=0):
        with cls.lock:
            logging.info(msg)