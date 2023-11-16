from math import e
from othello import *
import random
from dataset import *
import network
import pickle
import time
from tqdm import tqdm
import multiprocessing
import tensorflow as tf
import datetime
import sys
class local_locker():
    def __init__(self):
        self.lock=False
    def get_lock(self,id,time_out=0.1):
        s_t=time.time()
        while self.lock!=0 and (time_out<0 or time.time()-s_t<=time_out):
            pass
        if self.lock==0:
            self.lock=id
            return True
        else:
            return False
    def release_lock(self,id,time_out=0.1):
        if self.lock==id:
            self.lock=0
predict_module=None
def main(proc_num=None,play_num=8192,expand_rate=1,sub_play_count=1024,isModel=False,ForceUseMulti=False,isGDrive=False,time_limit=-1):
    IS_MULTI=True
    global predict_module
    if proc_num is None or proc_num<1:
        proc_num=multiprocessing.cpu_count()
    if proc_num==1 and not ForceUseMulti:
        IS_MULTI=False
    if play_num is None:
        play_num=sub_play_count*proc_num
    i=0
    if isGDrive:
        gdrive=gdrive_dataset()
    else:
        gdrive=None
    s_t=time.time()
    fname=None
    multi_man=multiprocessing.Manager()
    gpm_inputs=multi_man.list()
    gpm_outputs=multi_man.dict()
    gpm_outputs[-1]=False
    locker=multiprocessing.Lock()
    baseline_model=None
    while (isGDrive or i==0) and \
    (time_limit<0 or time.time()-s_t<=time_limit):
        refreshflg=False
        if isModel:
            model,tmp_f=network.raw_load_model(needsName=True)
            if fname is None or fname != tmp_f:
                refreshflg=True
                mcts=MCTS(game_cond(),model)
                predict_module=network.global_pred_module(model,max_size=32,inputspipe=gpm_inputs,outputspipe=gpm_outputs,lock=locker)
                
            fname=tmp_f
            if IS_MULTI and len(tf.config.list_physical_devices('GPU'))<1 and not ForceUseMulti:
                print("Multi processing feature will be ignored")
                # Neural network will use full cpu cores 
                # and multiprocessing will be bad in this situation
                IS_MULTI=False 
            predict_module.start_worker()
        else:
            model=None    
            
        i+=1
        dataset=[[],[]]
        if IS_MULTI:
            multiprocessing.freeze_support()
            with multiprocessing.Pool(proc_num,initializer=global_pred_initer,initargs=(locker,)) as p:
                pool_result=p.starmap(sub_create_dataset,
                                      [(play_num//proc_num,expand_rate,p_num+1,model,baseline_model,s_t,-1,mcts,gpm_inputs,gpm_outputs) for p_num in range(proc_num)],
                                      )
            for r in pool_result:
                if len(dataset[0])<1:
                    dataset[0]=r[0]
                    dataset[1]=r[1]
                else:
                    dataset[0]=dataset[0]+r[0]
                    dataset[1]=np.concatenate([dataset[1], r[1]])
        else:
            dataset=sub_create_dataset(play_num,expand_rate,None,None,model,mcts)
        dataset[1]=np.array(dataset[1],dtype="float32")
        dataset=[np.array(np.transpose(dataset[0],[0,2,3,1]),dtype=bool),dataset[1]]
        print(dataset[0].shape,dataset[1].shape)
        d=str(datetime.datetime.now()).replace(" ","_").replace(":","_")
        with open(f"./dataset/data_{d}.dat","wb") as f:
            pickle.dump(dataset,f)
        if isGDrive:
            gdrive.transfer_dataset()
    return dataset
lock=None
def global_pred_initer(lock_param):
    global lock
    print("before:",lock)
    lock=lock_param
    print("after:",lock)
def sub_create_dataset(
    play_num,expand_rate,
    p_num,
    model=None,baseline_model=None,
    s_t=0,time_limit=-1,
    mcts=None,
    inputs_pipe=None,outputs_pipe=None
    ):
    global predict_module,lock
    pmodule=predict_module
    print(pmodule)
    dataset=[[],[]]
    if model is None:
        isModel=False
    else:
        isModel=True
        if mcts is None:
            mcts=MCTS(game_cond(),model)
        minimax=minimax_search()
    if baseline_model is not None:
        baseline_mcts=MCTS(game_cond(),baseline_model)
    tqdm_obj=tqdm(range(play_num),position=0,leave=False)
    win_rate=[0,0,0]
    for _ in tqdm_obj:
        if (not time_limit<0) and time.time()-s_t>time_limit:
            break
        model_usage=[0,0]
        if isModel:
            model_usage=[(_%4)//2+1,(_%4)%2+1]
        model_usage=[0 if i==1 and baseline_model is None else i for i in model_usage]
        cond=game_cond()
        data=[[],[]]
        end_flg=0
        s_t=time.time()
        while not(cond.isEnd() or end_flg>=2):
            if time.time()-s_t>10 and not isModel:
                cond.show()
                print(cond.isEnd(),end_flg<2)
                raise TimeoutError("Took too much time to finish the round")
            poss=[]
            for p in cond.placable:
                if cond.is_movable(p[0],p[1]):
                    poss.append(p)
            if len(poss)>0:
                end_flg=0
                if model_usage[cond.turn]==0:
                    next_move=random.choice(poss)
                elif model_usage[cond.turn]==1:
                    if cond.board[0].sum()+cond.board[1].sum()>=56:
                        _,next_move=minimax.get_move(cond)
                    else:
                        next_move=baseline_mcts.get_next_move(cond,lock,inputs_pipe,outputs_pipe)[0]  
                elif model_usage[cond.turn]==2:
                    if cond.board[0].sum()+cond.board[1].sum()>=56:
                        s,next_move=minimax.get_move(cond)
                    else:
                        next_move=mcts.get_next_move(cond,lock,inputs_pipe,outputs_pipe)[0]
                data[cond.turn].append([cond.board,next_move,poss])
                cond.move(next_move[0],next_move[1])
            else:
                end_flg+=1
            cond.flip_board()
        score=list(cond.get_score())
        if model_usage[0]!=model_usage[1]:
            win_rate[model_usage[np.argmax(score)]]+=1
        for i,s in enumerate(score):
            for d in data[i]:
                board=d[0].repeat(expand_rate, axis=1).repeat(expand_rate, axis=2)
                dataset[0].append(np.array(board,dtype=bool))
                a=np.zeros((8,8))
                for p in d[2]:
                    if p==d[1]:
                        a[d[1][0]][d[1][1]]=score[i]/sum(score)
                    else:
                        a[p[0]][p[1]]=1-score[i]/sum(score)
                a/=max(0.001,a.reshape(64).sum())
                dataset[1].append(a.reshape(64))
        # if Lock.get_lock(p_num,time_out=-1):
        #     tqdm_obj.update(1)
        #     Lock.release_lock(p_num)
    tqdm_obj.close()
    # print("softmax")
    # dataset[1]=sfmax(dataset[1])
    # print("Done")
    print(win_rate)
    return dataset

def sfmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--num",type=int,default=1024)
    parser.add_argument("--pnum",type=int,default=1)
    parser.add_argument("--model","-m",action="store_true",default=False)
    parser.add_argument("--transflg","-gt",default=False,action="store_true")
    parser.add_argument("--gdrive","-g",default=False,action="store_true")
    parser.add_argument("--time","-t",type=int,default=-1)
    parser=parser.parse_args()
    proc_num=parser.pnum
    play_num=parser.num
    transflg=parser.transflg
    mflg=parser.model
    gflg=parser.gdrive
    time_limit=parser.time
    if not transflg:
        main(proc_num,play_num,isModel=mflg,isGDrive=gflg,time_limit=time_limit,ForceUseMulti=True)
    else:
        gdrive=gdrive_dataset()
        gdrive.get_dataset()
        gdrive.transfer_dataset()
        
        
