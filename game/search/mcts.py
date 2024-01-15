import math
from ..othello import game_cond,single_play_test
import numpy as np
import time
import random
# class single_play_test():
#     def __init__(self,a):
#         pass
    
#     def start_play(self):
#         return 
import tqdm    

base_value = np.array([
    [120,-20,20,5,5,20,-20,120],
    [-20,-40,-5,-5,-5,-5,-40,-20],
    [20,-5,15,3,3,15,-5,20],
    [5,-5,3,3,3,3,-5,5],
    [5,-5,3,3,3,3,-5,5],
    [20,-5,15,3,3,15,-5,20],
    [-20,-40,-5,-5,-5,-5,-40,-20],
    [120,-20,20,5,5,20,-20,120],
])

base_value = (base_value-base_value.min())/(base_value.max()-base_value.min())
class MCTS():
    def __init__(self, cond: game_cond, model, search_rate=1):
        self.model = model
        self.uct_c = 2**0.5
        self.qc_limit = 10000
        self.roll_out_limit = 1000
        self.time_limit = 20
        self.q_board_dict = {}
        self.search_rate = 1
        self.parent_set = set()
    def change_search_rate(self, rate):
        self.search_rate = rate
    
    def get_q_value(self,cond,key):
        poss = self.move_poss_dict[key]
        result = {}
        if cond.hash() in self.q_board_dict:
            r = self.q_board_dict[cond.hash()]
        else:
            r = base_value
            r = np.array(self.model(np.transpose(cond.board[np.newaxis], [
                            0, 2, 3, 1]), training=False)[0]).reshape(8, 8)
            
            self.q_board_dict[cond.hash()] = r
        for p in poss:
            q = r[p[0]][p[1]]
            #print("q score:",p,q)
            #print("parent key:",key,"main key:",int(self.sub_key(p)))
            #print("move:",p)
            self.qdict[int(key)*64+int(self.sub_key(p))] = q
            result[int(key)*64+int(self.sub_key(p))] = q
        return result
    
    def expand(self,cond:game_cond,key):
        poss = []
        for p in cond.placable:
            if cond.is_movable(p[0], p[1]):
                poss.append(p)
        self.move_poss_dict[key] = poss
        self.parent_set.add(key)
        target_key = key
        while target_key > 0:
            self.play_count[target_key] += len(poss)
            target_key //= 64
        #print("expand poss:",poss)
        for p in poss:
            
            k = int(key)*64+int(self.sub_key(p))
            if not k in self.play_count:
                self.play_count[k] = 1
            else:
                self.play_count[k] += 1
        return key,cond
    def get_next_move(self, cond:game_cond):
        """_summary_
        Returns:
            [r,0]
            r : best move predicted
        """
        qc_score = {}
        self.play_count = {}
        self.qdict = {}
        self.move_poss_dict = {}
        self.init_cond = game_cond(cond)
        grand_parent_key = self.init_cond.hash()  # Am I going to use this?
        grand_parent_key_level = np.sum(cond.board[0])+np.sum(cond.board[1])-4
        # Defines parent leaf
        key = 0
        n_moves = []
        counter = 0
        s_t = time.time()
        long_flg = True
        while len(qc_score) < self.qc_limit and counter < self.roll_out_limit and time.time()-s_t < self.time_limit:
            counter += 1
            
            # Moves to parent leaf
            cond = game_cond(self.init_cond)
            for move in n_moves:
                cond.move(move[0], move[1])
                cond.flip_board()
            #cond.show()
            key = self.get_key(n_moves)
            #print("n_moves:",n_moves)
            #print("key now:",key)
            
            # Expand
            key,cond = self.expand(cond,key)

            # Evaluate Q score for current leaf
            q_result = self.get_q_value(cond,key)
            for k,r in q_result.items():
                if not k in qc_score:
                    qc_score[k]=r
            
            # Evaluate Q score for other leafs
            for k in qc_score.keys():
                if qc_score[k] == 0:
                    qc_score[k] = self.qdict[k]

            # Evaluate C score
            #print("Getting C Score")
            gpc=set()
            c_score_sample=[]
            for key,qc in qc_score.items():
                N = self.play_count[key]
                # if not key in self.parent_set:
                #     print("key:",key,"play_count",N)
                N_sum = 0
                parent_key = key//64
                for move in self.move_poss_dict[parent_key]:
                    move_key=int(parent_key)*64+int(self.sub_key(move))
                    N_sum += self.play_count[move_key]
                
                c_score = self.uct_c*np.sqrt(np.log(N_sum)/N)
                c_score_sample.append(c_score)
                key_level=int(math.log(key)//math.log(64))
                pk = (key//(64**(key_level)))%64
                # print(f"gpk:{grand_parent_key_level}",
                #       f"key_level:{key_level}",
                #       "grand parent key:",pk,
                #       " key:",key,
                #       "  c_score:",c_score)
                gpc.add(pk)
                qc_score[key] *= c_score
            #print("c_score max:",max(c_score_sample))
            #print("c_score min:",min(c_score_sample))
            #print("q_score max:",max(list(self.qdict.values())))
            #print("q_score min:",min(list(self.qdict.values())))
            #print("gpc:",gpc)
            #print("Done")
            # Finds best qc score leaf and set it as new parent leaf
            array_qc = list(qc_score.items())
            array_qc.sort(key=lambda x:x[1])
            #print("qc_score:",qc_score)
            if len(array_qc) == 0:
                break
            pure_qc=np.array([i[1] for i in array_qc])**self.search_rate
            elem = random.choices([i for i in range(len(array_qc))],weights=pure_qc,k=1)[0]
            key = int(array_qc[elem][0])
            #print("next key:",key)
            #print(len(qc_score) < self.qc_limit,counter < self.roll_out_limit,time.time()-s_t < self.time_limit)
            #key = int(array_qc[-1][0])
            n_moves = self.get_moves(key)
            #qc_score.pop() なにこれ？
            for k in qc_score.keys():
                qc_score[k] = 0
        r = [None, 0]
        poss = []
        poss = self.move_poss_dict[0]
        random.shuffle(poss)
        poss_qc = []
        N_array=[]
        print("play count:",counter)
        best_play_count=0
        for p in poss:
            k = self.sub_key(p)
            play_count=self.play_count[k]
            poss_qc.append(play_count)
            if best_play_count<play_count:
                r=p
        print("mcts move:",poss)
        print("mcts key:",[i[0]*8+i[1] for i in poss])
        print("mcts score:",poss_qc)
        #print("mcts qdict:",self.qdict)
        return [r, 0]

    def get_key(self, moves: list):
        key = 0
        #print("moves:",moves)
        for move in moves:
            #print("move:",move)
            key *= 64
            key += self.sub_key(move)
        return key

    def sub_key(self, move):
        return move[0]*8+move[1]

    def get_moves(self, key):
        moves = []
        init_key = key
        while key > 0:
            moves.append(key % 8)
            key //= 8

        if len(moves) > 0:
            if len(moves) % 2 == 1:
                moves.append(0)
            moves.reverse()
            moves = np.array(moves).reshape(len(moves)//2, 2)
            moves = moves.tolist()
            #print("g_moves moves:",moves)
        return moves


class randomMCTS(MCTS):
    def __init__(self, cond: game_cond, model, search_rate=1,random_play_num=50000):
        super().__init__(cond,model,search_rate)
        self.play_num=random_play_num
        
    def get_q_value(self, cond:game_cond, poss, key):
        turn_now=cond.turn
        r = [0,0]
        for _ in tqdm.tqdm(range(self.play_num)):
            spt = single_play_test(cond,show=False)
            stone_num = cond.board[0].sum()+cond.board[1].sum()
            result = spt.start_play(end_num=min(64,stone_num+15))
            r[0]+=result[0]
            r[1]+=result[1]
        return r[turn_now]/(sum(r))
