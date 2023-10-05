import numpy as np
from collections import deque
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

class game_cond:
    def __init__(self):
        self.board=np.zeros((2,8,8),dtype=bool)
        
        self.board[0][3][3]=True
        self.board[0][4][4]=True
        self.board[1][3][4]=True
        self.board[1][4][3]=True
        
        self.placable={(2,2),(5,5),(2,5),(5,2),(2,3),(3,2),(4,5),(5,4),(3,5),(2,4),(5,3),(4,2)}
        self.turn=0
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
        return not(len(self.placable)>0 and self.board[0].sum()>0 and self.board[1].sum()>0)
    def isBlank(self,i,j):
        return not(self.board[0][i][j] or self.board[1][i][j])
    def get_score(self):
        return self.board[0].sum(),self.board[1].sum()
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
    import random,traceback,time
    from learning import ResNet
    model=ResNet((2,8,8),64)
    while True:
        try:
            s_t=time.time()
            cond=game_cond()
            #cond.show()
            end_flg=0
            while not cond.isEnd():
                poss=[]
                for p in cond.placable:
                    if cond.is_movable(p[0],p[1]):
                        poss.append(p)
                if len(poss)>0:
                    end_flg=0
                    next_move=random.choice(poss)
                    cond.move(next_move[0],next_move[1])
                else:
                    end_flg+=1
                cond.show()
                cond.flip_board()
                if time.time()-s_t>5:
                    cond.show()
                    time.sleep(1)
            print("\r",cond.board[0].sum(),cond.board[1].sum(),"time_taken:",round(time.time()-s_t,5),end="                           ")
        except KeyboardInterrupt:
            print(traceback.format_exc())
            exit()
        except Exception:
            print(traceback.format_exc())
            cond.show()
            exit()
    
    
def test_play(model,game_count=100):
    win_count=[0,0]
    for _ in range(game_count):
        cond=game_cond()
        end_flg=0
        
        while not(cond.isEnd() or end_flg>=2):
            poss=[]
            for p in cond.placable:
                if cond.is_movable(p[0],p[1]):
                    poss.append(p)
            if len(poss)>0:
                end_flg=0
                
                if game_cond.turn==0:
                    r=model(np.transpose(cond.board[np.newaxis],[0,2,3,1]))[0]
                    r=[r[p[0]][p[1]] for p in poss]
                else:
                    r=[1 for p in poss]
                next_move=random.choices(poss,weights=r)[0]
                cond.move(next_move[0],next_move[1])
            else:
                end_flg+=1
            cond.flip_board()
        score=list(cond.get_score())
        win_count[np.argmax(score)]+=1
    return win_count