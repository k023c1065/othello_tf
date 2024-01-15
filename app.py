from re import search
from flask import Flask,render_template,request,make_response
import numpy as np

import random
from datetime import timedelta

import game.othello as othello
import network.network as network

app = Flask(__name__,
            static_folder="./web_viewer/static",
            template_folder="./web_viewer/templates")

#app.secret_key=format(random.randint(0,2**128),'x')

model,model_fn = network.raw_load_model(get_filename=True)
print("model file name is ",model_fn)

othello_searcher=othello.simple_model_search(model=model,search_rate=1)
othello_mcts = othello.MCTS(othello.game_cond(),model)
othello_minimax = othello.minimax_search()
@app.route("/")
def init_page():
    response = make_response(render_template("index.html"))
    return response
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
@app.route("/model",methods=["GET","POST"])
def post_page():
    global othello_searcher
    r=request.form["board"]
    r = export_board(r)
    board = single2double(r)
    cond=othello.game_cond()
    cond.board=board
    cond.update_placable()
    cond.turn=1
    cond.show()
    print("placable:",cond.placable)
    if 64-(cond.board[0].sum()+cond.board[1].sum()) <= 8:
        move = othello_minimax.get_move(cond)[1]
    else:
        move = othello_mcts.get_next_move(cond)[0]
    print("move:",move)
    cond.move(move[0],move[1])
    cond.flip_board()
    poss = []
    for p in cond.placable:
        if cond.is_movable(p[0], p[1]):
            poss.append(p)
    get_next_move=[]
    score = -200
    for p in poss:
        if base_value[p[0]][p[1]]>score:
            score = base_value[p[0]][p[1]]
            get_next_move = p
    print("Your Move:",get_next_move)
    return f"{move[0]},{move[1]}"


def export_board(data):
    r=data.split(",")
    r=[int(i) for i in r]
    r = np.array(r)
    r = r.reshape((8,8))    
    return r

def single2double(board):
    result = np.zeros((2,8,8),dtype=bool)
    for i in range(8):
        for j in range(8):
            if board[i][j]!=0:
                result[(1+board[i][j])%2][i][j]=1
    return result
if __name__ == "__main__":
    app.run("localhost",port="25565")