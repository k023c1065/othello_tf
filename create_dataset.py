from othello import *
import random
from dataset import *
import pickle
import time
from tqdm import tqdm
def main():
    dataset=[[],[]]
    for _ in tqdm(range(10000)):
        cond=game_cond()
        data=[[],[]]
        end_flg=0
        s_t=time.time()
        while not(cond.isEnd() or end_flg>=2):
            if time.time()-s_t>4:
                cond.show()
                print(cond.isEnd(),end_flg<2)
                time.sleep(10)
            poss=[]
            for p in cond.placable:
                if cond.is_movable(p[0],p[1]):
                    poss.append(p)
            if len(poss)>0:
                end_flg=0
                next_move=random.choice(poss)
                data[cond.turn].append([cond.board,next_move])
                cond.move(next_move[0],next_move[1])
            else:
                end_flg+=1
            cond.flip_board()
        score=list(cond.get_score())
        for i,s in enumerate(score):
            for d in data[i]:
                dataset[0].append(d[0])
                a=move2board(d[1],i)
                dataset[1].append(a*score[i]/sum(score))
    dataset=[np.array(dataset[0],dtype=bool),np.array(dataset[1],dtype="float16")]
    print(dataset[0].shape,dataset[1].shape)
    with open("./dataset/data.dat","wb") as f:
        pickle.dump(dataset,f)
    
if __name__=="__main__":
    main()