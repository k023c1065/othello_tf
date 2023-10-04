import numpy as np
import random
def move2board(move,turn):
    a=np.zeros((8,8))
    a[move[0]][move[1]]=1
    return a

