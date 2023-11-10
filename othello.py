import glob
import numpy as np
from collections import deque
import sys

from network import raw_load_model
sys.setrecursionlimit(10**6)
class ALLWAYSFALSE:
    def __init__(self):
        pass
    def __call__(self):
        return False
    def __eq__(self,other):
        return False
    def __ne__(self,other):
        return False
    
const_direction=[
    [-1, 0], # 左
    [-1, 1], # 左下
    [0, 1], # 下
    [1, 1], # 右下
    [1, 0], # 右
    [1, -1], # 右上
    [0, -1], # 上
    [-1, -1] # 左上
]
const_direction=deque(const_direction)
const_direction.rotate(4)
const_direction=list(const_direction)
from copy import copy,deepcopy
base_2=[2**i for i in range(128)]
class game_cond:
    def __init__(self,cond=None):
        self.moveflg=False
        if cond is None:
            self.board=np.zeros((2,8,8),dtype=bool)
            self.board[0][3][3]=True
            self.board[0][4][4]=True
            self.board[1][3][4]=True
            self.board[1][4][3]=True
            
            self.placable={(2,2),(5,5),(2,5),(5,2),(2,3),(3,2),(4,5),(5,4),(3,5),(2,4),(5,3),(4,2)}
            self.turn=0
            self.notChangedCount=0
            self.moveflg=False
        else:
            self.board=deepcopy(cond.board)
            self.placable=deepcopy(cond.placable)
            self.turn=copy(cond.turn)
            self.notChangedCount=copy(cond.notChangedCount)
    def hash(self):
        flatten_board=self.board.flatten()
        hash=flatten_board*base_2
        hash=np.base_repr(hash.sum(),36)+str(self.turn)
        return hash
    def is_movable(self,i,j):
        if not (i,j) in self.placable:
            return False
        r=self.OpponentAround(i,j,(self.turn+1)%2)
        target_axis=[1,1]
        if i==0:
            target_axis[0]=0
        if j==0:
            target_axis[1]=0
        poss=[]
        #Looks if the cell is empty or not
        for h,a in enumerate(r):
            for k,data in enumerate(a):
                if not(h==target_axis[0] and k==target_axis[1]):
                    if data:
                        poss.append([h,k])
        for p in poss:
            test_axis=[i+(p[0]-target_axis[0]),j+(p[1]-target_axis[1])]

            while self.isLegal2(test_axis) and self.get_board(test_axis,(self.turn+1)%2)==True:
                test_axis[0]+=(p[0]-target_axis[0])
                test_axis[1]+=(p[1]-target_axis[1])

            if self.isLegal2(test_axis) and self.get_board(test_axis,self.turn)==True:
                return True
        return False
    def get_board(self,axis,turn):
        return self.board[turn][axis[0]][axis[1]]       
    def get_board2(self,axis,sub_axis,turn):
        new_axis=[axis[0]+sub_axis[0],axis[1]+sub_axis[1]]
        new_axis[0]=min(7,max(0,new_axis[0]))
        new_axis[1]=min(7,max(0,new_axis[1]))
        return self.get_board(new_axis,turn)
    
    def move(self,i,j):
        # Special if function for turn skip
        if i==-1 and j==-1:
            return
        r=self.OpponentAround(i,j,(self.turn+1)%2)
        target_axis=[1,1]
        if i==0:
            target_axis[0]=0
        if j==0:
            target_axis[1]=0
        poss=[]
        #Looks if the cell is empty or not
        for h,a in enumerate(r):
            for k,data in enumerate(a):
                if not(h==target_axis[0] and k==target_axis[1]):
                    if data:
                        poss.append([h,k])
        
        for p in poss:
            test_axis=[i+(p[0]-target_axis[0]),j+(p[1]-target_axis[1])]
            count=1
            while self.isLegal2(test_axis) and self.get_board(test_axis,(self.turn+1)%2)==True:
                test_axis[0]+=(p[0]-target_axis[0])
                test_axis[1]+=(p[1]-target_axis[1])
                count+=1
            if self.isLegal2(test_axis) and self.get_board(test_axis,self.turn)==True:
                count-=1
                for _ in range(0,count+1):
                    new_axis=[i+_*(p[0]-target_axis[0]),j+_*(p[1]-target_axis[1])]
                    self.board[(self.turn+1)%2][new_axis[0]][new_axis[1]]=0
                    self.board[(self.turn)][new_axis[0]][new_axis[1]]=1
                    if tuple(new_axis) in self.placable:
                        self.placable.remove(tuple(new_axis))
                    for x in const_direction:
                        if not(self.get_board2(new_axis,x,0) or self.get_board2(new_axis,x,1)):
                            self.placeable_add(new_axis,x)
            self.notChangedCount=0
            self.moveflg=True
    def placeable_add(self,axis,sub_axis):
        new_axis=[axis[0]+sub_axis[0],axis[1]+sub_axis[1]]
        new_axis[0]=min(7,max(0,new_axis[0]))
        new_axis[1]=min(7,max(0,new_axis[1]))
        self.placable.add(tuple(new_axis))
    def _is_movable(self,i,j):
        result=[]
        if (i,j) in self.placable:
            r=self.OpponentAround(i,j,(self.turn+1)%2)
            target_axis=[1,1]
            if i==0:
                target_axis[0]=0
            if j==0:
                target_axis[1]=0
            poss=[]
            #Looks if the cell is empty or not
            for h,a in enumerate(r):
                for k,data in enumerate(a):
                    if not(h==target_axis[0] and k==target_axis[1]):
                        if data:
                            poss.append([h,k])
            
            for p in poss:
                axis_diff=abs(target_axis[0]-p[0])+abs(target_axis[1]-p[1])
                if axis_diff==1:
                    end_point=[min(7,max(0,i-(target_axis[0]-p[0])*8)),min(7,max(0,j-(target_axis[1]-p[1])*8))]
                    Line_Ally=self.board[0][min(i+1,end_point[0]):1+max(i-1,end_point[0]),min(j+1,end_point[1]):1+max(j-1,end_point[1])]
                    
                    Line_Ally=Line_Ally.reshape(1,Line_Ally.shape[0]*Line_Ally.shape[1])[0]
                    
                    if (target_axis[0]-p[0])>0 or (target_axis[1]-p[1])>0:
                        Line_Ally=np.flip(Line_Ally)
                    print(Line_Ally)
                    next_ally_index=np.where(Line_Ally==True)[0]
                    if len(next_ally_index)>0:
                        next_ally_index=next_ally_index[0]
                        Line_opponent=self.board[1][min(i+1,end_point[0]):1+max(i-1,end_point[0]),min(j+1,end_point[1]):1+max(j-1,end_point[1])]
                        Line_opponent=Line_opponent.reshape(1,Line_opponent.shape[0]*Line_opponent.shape[1])[0]
                        if (target_axis[0]-p[0])>0 or (target_axis[1]-p[1])>0:
                            Line_opponent=np.flip(Line_opponent)
                        print(Line_opponent[:next_ally_index])
                        if Line_opponent[:next_ally_index].all():
                            return True
        return False
    def flip_board(self):
        if not self.moveflg:
            self.notChangedCount+=1
        self.moveflg=False
        self.turn=(self.turn+1)%2
    def isLegal(self,i,j):
        return i>=0 and i<8 and j>=0 and j<8   
    def isLegal2(self,axis):
        return self.isLegal(axis[0],axis[1])   
    def OpponentAround(self,i,j,turn):
        return self.board[turn][max(0,i-1):min(8,i+2),max(0,j-1):min(8,j+2)]
    def safeboard(self,i,j):
        if self.isLegal(i,j):
            return self.board
        else:
            return ALLWAYSFALSE()
    def isEnd(self):
        return len(self.placable)<1 or self.board[0].sum()<1 or self.board[1].sum()<1 or self.notChangedCount>=2
    def isBlank(self,i,j):
        return not(self.board[0][i][j] or self.board[1][i][j])
    def get_score(self):
        return self.board[0].sum(),self.board[1].sum()
    def get_score_float(self):
        s=list(self.get_score())
        return s[(self.turn+1)%2]/sum(s)
    def show(self):
        print(" 0 1 2 3 4 5 6 7")
        for i in range(8):
            print(i,end="",flush=False)
            for j in range(8):
                if self.board[0][i][j]:
                    print("○ ",end="",flush=False)
                elif self.board[1][i][j]:
                    print("● ",end="",flush=False)
                else:
                    print("□ ",end="",flush=False)
            print(flush=False)
        print()

if __name__=="__main__":
    pass
import random,time
from collections import deque

        
from search.ab import minimax_search
from search.mcts import MCTS
from tqdm import tqdm_gui   
class test_play():
    def __init__(self,players=["Model","Random"],model=[None,None],game_count=100,isDebug=None,Doshuffle=False):  
            self.minimax=minimax_search()
            self.isDebug=isDebug
            self.players=players
            self.shuffle=Doshuffle
            for i,player in enumerate(self.players):
                if player=="Model":
                    if model[i]==None:
                        self.players[i]="Random"
                        print(f"Player No:{i+1} has been changed to Random.Please specify the model.")
                    if type(model[i]) is str:
                        print(f"Model file:{model[i]} has been specified as model for Player No:{i+1}")
                        model[i]=raw_load_model(model[i])
            self.mcts=[MCTS(game_cond(),model[i]) if self.players[i]=="Model" else None for i in range(2) ]            
            self.game_count=game_count
            self.turn_append=0
            if self.isDebug is None:
                if game_count==1:
                    self.isDebug=True
                else:
                    self.isDebug=False
    def loop_game(self):
        win_count=[0,0]
        
        for _ in range(self.game_count):
            if self.shuffle:
                self.turn_append=random.randint(0,1)
            print(f"Target model Player No.:{self.turn_append+1}")
            ab_s=[]
            try:
                cond=game_cond()
                if self.isDebug:cond.show()
                end_flg=0
                print()
                while not(cond.isEnd() or end_flg>=2):
                    print("\rGame count:",_+1,"space left:",64-(cond.board[0]+cond.board[1]).sum(),end="        ")
                    poss=[]
                    for p in cond.placable:
                        if cond.is_movable(p[0],p[1]):
                            poss.append(p)
                    if len(poss)>0:
                        end_flg=0
                        next_move,ab_s=self.get_next_move(cond,poss,ab_s,self.players[cond.turn])
                        if self.isDebug:print("move:",next_move)
                        cond.move(next_move[0],next_move[1])
                    else:
                        end_flg+=1
                    if self.game_count==1:
                        cond.show()
                    cond.flip_board()
            except KeyboardInterrupt:
                cond.show()
                raise KeyboardInterrupt()
            score=list(cond.get_score())
            win_side=np.argmax(score)
            win_count[(win_side+self.turn_append)%2]+=1
            if win_side==1:
                print("ab_s:",ab_s)
            print("win_count:",win_count)
            
        return win_count
    def get_next_move(self,cond:game_cond,poss,ab_s,player):
        if player == "Model":
            if self.isDebug:print("Executing model")
            if 64-(cond.board[0].sum()+cond.board[1].sum())<=8:
                s,next_move=self.minimax.get_move(cond)
                ab_s.append(s)
                if self.isDebug:print("s:",s)
            else:
                next_move=self.mcts[(cond.turn+self.turn_append)%2].get_next_move(cond)[0]
        elif player == "Random":
            if self.isDebug:print("Executing random")
            r=[1 for p in poss]
            next_move=random.choices(poss,weights=r)[0]
        elif player == "Human":
            if self.isDebug:print("Executing Human inputs")
            axis=(-1,-1)
            while not axis in poss:
                print("Allowed Move:",poss)
                inp_axis=input("Please Enter your next Move:")
                axis=[int(i) for i in inp_axis.split()]
                axis=tuple(axis)
            next_move=axis
        return next_move,ab_s
import os
if __name__=="__main__":
    fs=glob.glob("model/*")
    fs=sorted(fs,key=os.path.getmtime)
    f=fs[-1]
    import network
    target_model=network.miniResNet((8,8,2),64)
    target_model(np.empty((1,8,8,2)))
    target_model.load_weights(f)
    print(f)
    if len(fs)>1:
        baseline_model=network.raw_load_model(fs[-2])
        print(fs[-2])
        
    tp=test_play(players=["Model","Model"],model=[target_model,baseline_model],game_count=20,Doshuffle=False)
    score=tp.loop_game()
    tp=test_play(players=["Model","Model"],model=[baseline_model,target_model],game_count=20,Doshuffle=False)
    score2=tp.loop_game()
    score2[0],score2[1]=score2[1],score2[0]
    score=np.array(score)+np.array(score2)
    print((score[0]*100)/score.sum(),"%")