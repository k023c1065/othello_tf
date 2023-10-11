from othello import *
import random
from dataset import *
import network
import pickle
import time
from tqdm import tqdm
import multiprocessing
import cv2
import tensorflow as tf
def main(proc_num=None,play_num=10000,expand_rate=1,isModel=False):
    IS_MULTI=True
    if proc_num is None:
        proc_num=multiprocessing.cpu_count()
    if proc_num==1:
        IS_MULTI=False
    dataset=[[],[]]
    if isModel:
        model=network.raw_load_model()
        if IS_MULTI and not tf.test.is_gpu_available():
            print("Multi processing feature will be ignored")
            IS_MULTI=False #Neural network will use full cpu cores and multiprocessing will be bad in this situation
    else:
        model=None
    if IS_MULTI:
        multiprocessing.freeze_support()
        Lock=multiprocessing.Manager().Lock()
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
    # dataset[1]=(dataset[1]-dataset[1].mean())/dataset[1].std()
    # dataset[1]=dataset[1]-dataset[1].min()
    # dataset[1]=dataset[1]/dataset[1].max()
    dataset=[np.array(np.transpose(dataset[0],[0,2,3,1]),dtype=bool),dataset[1]]
    print(dataset[0].shape,dataset[1].shape)
    with open("./dataset/data.dat","wb") as f:
        pickle.dump(dataset,f)
    return dataset
def sub_create_dataset(play_num,expand_rate,p_num,Lock,model=None):
    dataset=[[],[]]
    if model is None:
        isModel=False
    else:
        isModel=True
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
                    mcts=MCTS(cond,model)
                    next_move=mcts.get_move()[0]
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
    main(proc_num=4,play_num=200,)