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
def main(proc_num=None,play_num=8192,expand_rate=1,sub_play_count=1024,isModel=False,ForceUseMulti=False):
    IS_MULTI=True
    if proc_num is None:
        proc_num=multiprocessing.cpu_count()
    if proc_num==1 and not ForceUseMulti:
        IS_MULTI=False
    if play_num is None:
        play_num=sub_play_count*proc_num
    dataset=[[],[]]
    if isModel:
        model=network.raw_load_model()
        if IS_MULTI and len(tf.config.list_physical_devices('GPU'))<1 and not ForceUseMulti:
            print("Multi processing feature will be ignored")
            IS_MULTI=False #Neural network will use full cpu cores and multiprocessing will be bad in this situation
    else:
        model=None
    if IS_MULTI:
        multiprocessing.freeze_support()
        Lock=None
        with multiprocessing.Pool(proc_num) as p:
            pool_result=p.starmap(sub_create_dataset,[(play_num//proc_num,expand_rate,p_num,Lock,model) for p_num in range(proc_num)])
        for r in pool_result:
            if len(dataset[0])<1:
                dataset[0]=r[0]
                dataset[1]=r[1]
            else:
                dataset[0]=dataset[0]+r[0]
                dataset[1]=np.concatenate([dataset[1], r[1]])
    else:
        dataset=sub_create_dataset(play_num,expand_rate,None,None,model)
    dataset[1]=np.array(dataset[1],dtype="float32")
    dataset=[np.array(np.transpose(dataset[0],[0,2,3,1]),dtype=bool),dataset[1]]
    print(dataset[0].shape,dataset[1].shape)
    with open(f"./dataset/data{datetime.datetime.now()}.dat","wb") as f:
        pickle.dump(dataset,f)
    return dataset
def sub_create_dataset(play_num,expand_rate,p_num,Lock,model=None):
    dataset=[[],[]]
    if model is None:
        isModel=False
    else:
        isModel=True
        mcts=MCTS(game_cond(),model)
        minimax=minimax_search()
    for _ in tqdm(range(play_num)):
        model_usage=[0,0]
        if isModel:
            model_usage=[(_%4)//2,(_%4)%2]
        cond=game_cond()
        data=[[],[]]
        end_flg=0
        s_t=time.time()
        while not(cond.isEnd() or end_flg>=2):
            if time.time()-s_t>10 and not isModel:
                cond.show()
                print(cond.isEnd(),end_flg<2)
                time.sleep(10)
            poss=[]
            for p in cond.placable:
                if cond.is_movable(p[0],p[1]):
                    poss.append(p)
            if len(poss)>0:
                end_flg=0
                if model_usage[cond.turn]==0:
                    next_move=random.choice(poss)
                else:
                    if cond.board[0].sum()+cond.board[1].sum()>=56:
                        s,next_move=minimax.get_move(cond)
                    else:
                        next_move=mcts.get_next_move(cond)[0]
                data[cond.turn].append([cond.board,next_move,poss])
                cond.move(next_move[0],next_move[1])
            else:
                end_flg+=1
            cond.flip_board()
        score=list(cond.get_score())
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
                a/=a.reshape(64).sum()
                dataset[1].append(a.reshape(64))
    # print("softmax")
    # dataset[1]=sfmax(dataset[1])
    # print("Done")
    return dataset

def sfmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

if __name__=="__main__":
    main(proc_num=6,play_num=1024,isModel=True,ForceUseMulti=False)
