import multiprocessing
import subprocess
import os

def func():
    r=subprocess.run(["python","-c","import time;time.sleep(5);print(\"ENDENDEND\")"],shell=True,stdout=subprocess.PIPE,close_fds=True)
    with open("test.txt","w") as f:
        f.write(r.stdout.decode())
    return r


if __name__=="__main__":
    print(os.path.dirname(__file__))
    exit()
    multiprocessing.freeze_support()
    p=multiprocessing.Process(target=func,close_fds=True)
    
    p.start()
    print(p)
    p.join()