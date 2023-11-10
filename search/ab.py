from othello import game_cond

class minimax_search():
    def __init__(self):
        self.cache={}
    #From https://zero2one.jp/learningblog/mini-max-alpha-beta/
    #Thank You!
    def get_move(self,board:game_cond):
        if board.hash() in self.cache:
            return self.cache[board.hash()]
        if board.isEnd():
            return board.get_score_float(),None
        best_score = 10
        best_move=[]
        # 全ての可能な手について評価関数を計算
        poss=[]
        for p in board.placable:
            if board.is_movable(p[0],p[1]):
                poss.append(p)
        #　打つ手がない場合のみ手番をスキップすることができる
        if len(poss)==0:
            b=game_cond(board)
            b.flip_board()
            score,_ = self.get_move(b)
            score=1-score
            if score < best_score:
                best_score = score
                best_move=[-1,-1]
        else:
            for move in poss:
                b=game_cond(board)
                b.move(move[0],move[1])
                b.flip_board()
                #　スコアは必ず[0,1]となるので1-scoreとすることで相手のスコアを取得することができる
                score,_ = self.get_move(b)
                score = 1-score
                if score < best_score:
                    best_score = score
                    best_move=move
        self.cache[board.hash()]=(best_score,best_move)
        return best_score,best_move
    