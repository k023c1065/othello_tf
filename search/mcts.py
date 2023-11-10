from othello import game_cond
import random,time
import numpy as np
class MCTS():
    def __init__(self,cond:game_cond,model,search_rate=1):
        self.model=model
        self.uct_c=2**0.5
        self.qc_limit=100
        self.roll_out_limit=100
        self.time_limit=15
        self.q_board_dict={}
        self.search_rate=1
    def change_search_rate(self,rate):
        self.search_rate=rate
    def get_next_move(self,cond,locker,ipipe,opipe):
        qc_score=[]
        self.play_count={}
        self.qdict={}
        self.move_poss_dict={}
        self.init_cond=game_cond(cond)
        grand_parent_key=self.init_cond.hash() #Am I going to use this?
        # Defines parent leaf
        key=0
        n_moves=[]
        counter=0
        s_t=time.time()
        long_flg=True
        while len(qc_score)<self.qc_limit and counter<self.roll_out_limit and time.time()-s_t<self.time_limit:
            counter+=1
            # Moves to parent leaf
            cond=game_cond(self.init_cond)
            for move in n_moves:
                cond.move(move[0],move[1])
                cond.flip_board()
            # Expand
            poss=[]
            for p in cond.placable:
                if cond.is_movable(p[0],p[1]):
                    poss.append(p)
            self.move_poss_dict[key]=poss
            target_key=key
            while target_key>0:
                self.play_count[target_key]+=len(poss)
                target_key//=64
            for p in poss:
                k=key*64+self.sub_key(p)
                self.play_count[k]=1
            
            # Evaluate Q score for current leaf
            if cond.hash() in self.q_board_dict:
                r=self.q_board_dict[cond.hash()]
            elif ipipe is None:
                r=np.array(self.model(np.transpose(cond.board[np.newaxis],[0,2,3,1]),training=False)[0]).reshape(8,8)
                self.q_board_dict[cond.hash()]=r
            else:
                try:
                    with locker:
                        print("Sending Data...",end="")
                        ipipe.append(np.transpose(cond.board,[1,2,0]))
                        id=len(ipipe)-1
                        print("Done")
                except AttributeError:
                    print("\n"*30,"locker:",locker)
                    input("Waiting....")
                    raise AttributeError()
                print("Getting data...",end="")
                while (not opipe[-1]) or (not id in opipe.keys()) :
                    pass
                print(opipe.items())
                r=opipe[id]
                try:
                    assert(type(r) is np.ndarray and r.shape=(1,8,8))
                print("Done")
                self.q_board_dict[cond.hash()]=r
            for p in poss:
                q=r[p[0]][p[1]]
                self.qdict[key*64+self.sub_key(p)]=q
                qc_score.append([q,key*64+self.sub_key(p)])
                
            # Evaluate Q score for other leafs
            for qc in qc_score:
                if qc[0]==0:
                    qc[0]=self.qdict[qc[1]]
            
            # Evaluate C score
            for qc in qc_score:
                N=self.play_count[qc[1]]
                N_sum=0
                parent_key=qc[1]//64
                try:
                    #print("parent moves:",self.get_moves(parent_key))
                    for move in self.move_poss_dict[parent_key]:
                        N_sum+=self.play_count[parent_key*64+self.sub_key(move)]
                except KeyError:
                    print(qc[1])
                    print(parent_key)
                    raise KeyError()
                c_score=self.uct_c*np.sqrt(np.log(N_sum)/N)
                qc[0]+=c_score
            #Finds best qc score leaf and set it as new parent leaf
            qc_score.sort()
            if len(qc_score)==0:
                break
            n_moves=self.get_moves(qc_score[-1][1])
            key=int(qc_score[-1][1])
            qc_score.pop()
            for qc in qc_score:
                qc[0]=0
        r=[None,0]
        poss=[]
        poss=self.move_poss_dict[0]
        poss_qc=[]
        for p in poss:
            key=self.sub_key(p)
            poss_qc.append(self.play_count[key]**(1/self.search_rate))
        r=random.choices(poss,weights=poss_qc,k=1)[0]
        return [r,0]
        
            
    def get_key(self,moves:list):
        key=0
        for move in enumerate(moves):
            key*=64
            key+=self.sub_key(move)
        return key
    def sub_key(self,move):
        return move[0]+8*move[1]
    def get_moves(self,key):
        moves=[]
        init_key=key
        while key>0:
            moves.append(key%8)
            key//=8
        if len(moves)>0:
            if len(moves)%2==1:
                moves.append(0)
            moves=np.array(moves).reshape(len(moves)//2,2)
            moves=list(moves)
            moves.reverse()
        return moves