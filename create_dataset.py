import ctypes
from game.othello import *
import random
from dataset import *
import network.network as network
import pickle
import time
from tqdm import tqdm
import multiprocessing
import tensorflow as tf
import datetime
import os
import numpy as np
import traceback
from file_log.mylog import mylog
btqdm_flg = False
log_file_name=f"./log/{str(datetime.datetime.now())}.log".replace(" ","").replace(":","_")

def main(proc_num=None, play_num=8192, expand_rate=1, sub_play_count=1024, isModel=False,
         ForceUseMulti=False,
         isGDrive=False,
         time_limit=-1, mcts_flg=True):
    IS_MULTI = True
    if proc_num is None:
        proc_num = multiprocessing.cpu_count()
    if proc_num == 1 and not ForceUseMulti:
        IS_MULTI = False
    if play_num is None:
        play_num = sub_play_count*proc_num
    i = 0
    if isGDrive:
        gdrive = gdrive_dataset()
    else:
        gdrive = None
    s_t = time.time()
    last_fn = None
    SMSearch=[None,None]
    while (isGDrive or i == 0) and (time_limit < 0 or time.time()-s_t <= time_limit):
        if isModel:
            model, fn = network.network.raw_load_model(get_filename=True)
            print("Target model:",fn)
            if last_fn is None or fn != last_fn:
                last_fn = fn
                SMSearch[0] = simple_model_search(model,search_rate=0.75)
            if IS_MULTI and len(tf.config.list_physical_devices('GPU')) < 1 and not ForceUseMulti:
                print("Multi processing feature will be ignored")
                # Neural network will use full cpu cores
                # and multiprocessing will be bad in this situation
                IS_MULTI = False
            if os.path.isfile("./model/baseline.h5"):
                baseline_model = network.network.raw_load_model("model/baseline.h5")
                if SMSearch[1] is None:
                    baseline_SMSearch = simple_model_search(baseline_model,search_rate=0.75)
                    SMSearch[1]=baseline_SMSearch
                print("Loading baseline model:baseline.h5")
            else:
                baseline_model= None
                SMSearch=[SMSearch]
                print("Not loading Baseline model.\
                      Changed the model file name into 'baseline.h5' to apply baseline model.")
        else:
            model = None
            baseline_model = None
        i += 1
        dataset = [[], []]
        if IS_MULTI:
            multiprocessing.freeze_support()
            Lock = multiprocessing.Lock()
            end_num = multiprocessing.Value(ctypes.c_int, 0)
            end_num.value = 0
            with multiprocessing.Pool(proc_num, initializer=scd_initer, initargs=(Lock, end_num)) as p:
                pool_result = p.starmap(sub_create_dataset,
                                        [(play_num//proc_num, expand_rate, p_num+1, model,
                                          baseline_model, 0, -1, mcts_flg, SMSearch,  proc_num, log_file_name)
                                         for p_num in range(proc_num)],
                                        )
            for r in pool_result:
                if len(dataset[0]) < 1:
                    dataset[0] = r[0]
                    dataset[1] = r[1]
                else:
                    dataset[0] = dataset[0]+r[0]
                    dataset[1] = np.concatenate([dataset[1], r[1]])
            mylog.add_log(f"Valid dataset:{(len(dataset[0])*100/play_num)/64}%")
        else:
            dataset = sub_create_dataset(
                play_num, expand_rate, None, None, model)
        dataset[1] = np.array(dataset[1], dtype="float32")
        dataset = [np.array(np.transpose(
            dataset[0], [0, 2, 3, 1]), dtype=bool), dataset[1]]
        print(dataset[0].shape, dataset[1].shape)
        d = str(datetime.datetime.now()).replace(" ", "_").replace(":", "_")
        with open(f"./dataset/data_{d}.dat", "wb") as f:
            pickle.dump(dataset, f)
        dataset_file = glob("./dataset/*.dat")
        if len(dataset_file) > 10:
            split_datasets(file_num=4)
        if isGDrive:
            gdrive.transfer_dataset()
    return dataset


Lock = None
end_num = None


def scd_initer(lock, em):
    global Lock, end_num
    Lock = lock
    end_num = em


def sub_create_dataset(
    play_num, expand_rate,
    p_num,
    model=None, baseline_model=None,
    s_t=0, time_limit=-1,
    mcts_flg=True, sms=None,
    pnum=1,
    log_file_name=f"./log/{str(datetime.datetime.now())}.log".replace(" ","").replace(":","_")
):
    mylog.define_config(log_file_name)
    #print("bm:", baseline_model)
    global btqdm_flg
    global Lock, end_num
    dataset = [[], []]
    if model is None:
        isModel = False
    else:
        isModel = True
        if mcts_flg:
            mcts = MCTS(game_cond(), model)
        minimax = minimax_search()
    if baseline_model is not None:
        if mcts_flg:
            baseline_mcts = MCTS(game_cond(), baseline_model)
    tqdm_obj = tqdm(range(play_num*pnum), position=0,
                    leave=False, dynamic_ncols=btqdm_flg)
    win_rate = [0, 0, 0]
    for _ in range(play_num):
        if (not time_limit < 0) and time.time()-s_t > time_limit:
            break
        model_usage = [0, 0]
        if isModel:
            model_usage = [(_ % 4)//2+1, (_ % 4) % 2+1]
        if not ((model_usage[0] == 0 and model_usage[0] == 0) or
                (model_usage[0] == 1 and model_usage[1] == 1 and mcts_flg == False)):
            model_usage = [
                0 if i == 1 and baseline_model is None else i for i in model_usage]
            cond = game_cond()
            data = [[], []]
            end_flg = 0
            s_t = time.time()
            while not (cond.isEnd() or end_flg >= 2):
                # tqdm_obj.set_description(f"{np.sum(cond.board[0]+cond.board[1])}/64")
                if time.time()-s_t > 10 and not isModel:
                    cond.show()
                    print(cond.isEnd(), end_flg < 2)
                    raise TimeoutError(
                        "Took too much time to finish the round")
                poss = []
                for p in cond.placable:
                    if cond.is_movable(p[0], p[1]):
                        poss.append(p)
                if len(poss) > 0:
                    end_flg = 0
                    if model_usage[cond.turn] == 0: #Random
                        next_move = random.choice(poss)
                    elif model_usage[cond.turn] == 1: #Baseline model
                        if cond.board[0].sum()+cond.board[1].sum() >= 56:
                            _, next_move = minimax.get_move(cond)
                        else:
                            if mcts_flg:
                                next_move = baseline_mcts.get_next_move(cond)[
                                    0]
                            else:
                                next_move = sms[1].search(cond)
                    elif model_usage[cond.turn] == 2: #Main Model
                        if cond.board[0].sum()+cond.board[1].sum() >= 56:
                            s, next_move = minimax.get_move(cond)
                        else:
                            if mcts_flg:
                                next_move = mcts.get_next_move(cond)[0]
                            else:
                                next_move = sms[0].search(cond)
                    data[cond.turn].append([cond.board, next_move, poss])
                    cond.move(next_move[0], next_move[1])
                else:
                    end_flg += 1
                cond.flip_board()
            score = list(cond.get_score())
            if model_usage[0] != model_usage[1]:
                win_rate[model_usage[np.argmax(score)]] += 1
            if (baseline_model is not None and model is not None ) or (model_usage[0] == model_usage[1] or score[model_usage.index(0)] == 1):
                for i, s in enumerate(score):
                    for d in data[i]:
                        board = d[0]
                        if expand_rate != 1:
                            board = board.repeat(expand_rate, axis=1).repeat(
                                expand_rate, axis=2)
                        dataset[0].append(board.astype(bool))
                        a = np.zeros((8, 8))
                        for p in d[2]:
                            if p == d[1]:
                                a[d[1][0]][d[1][1]] = score[i]/sum(score)
                            else:
                                a[p[0]][p[1]] = 1-score[i]/sum(score)
                        a /= max(0.001, a.reshape(64).sum())
                        dataset[1].append(a.reshape(64))
            # if Lock.get_lock(p_num,time_out=-1):
            #     tqdm_obj.update(1)
            #     Lock.release_lock(p_num)
            with Lock:
                end_num.value += 1
                tqdm_obj.update(end_num.value-tqdm_obj.n)
                tqdm_obj.display()
    print(win_rate)
    mylog.add_log(f"win_rate:{win_rate}")
    return dataset


def sfmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

def init_parser(parser):
    parser.add_argument("--num", type=int, default=1024)
    parser.add_argument("--pnum", type=int, default=1)
    parser.add_argument("--model", "-m", action="store_true", default=False)
    parser.add_argument("--transflg", "-gt",
                        default=False, action="store_true")
    parser.add_argument("--gdrive", "-g", default=False, action="store_true")
    parser.add_argument("--time", "-t", type=int, default=-1)
    parser.add_argument("--mcts", default=True, action="store_false")
    parser.add_argument("--btqdm", action="store_false")
    return parser
def do_parse(parser):
    parser = parser.parse_args()
    proc_num = parser.pnum
    play_num = parser.num
    transflg = parser.transflg
    mflg = parser.model
    gflg = parser.gdrive
    time_limit = parser.time
    mcts_flg = parser.mcts
    btqdm_flg = parser.btqdm
    fum = proc_num > 1
    return locals()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser=init_parser(parser)
    proc_num, play_num, transflg, mflg, gflg, time_limit, mcts_flg, btqdm_flg, fum = 0, 0, 0, 0, 0, 0, 0, 0, 0
    parse_variable=do_parse(parser=parser)
    globals().update(parse_variable)
    assert play_num>0
    print(__file__, os.path.dirname(__file__))
    if not os.path.dirname(__file__) == "":
        os.chdir(os.path.dirname(__file__))
    if not transflg:
        try:
            main(proc_num, play_num,
                 isModel=mflg, isGDrive=gflg,
                 time_limit=time_limit, ForceUseMulti=fum,
                 mcts_flg=mcts_flg)
        except Exception as e:
            print(traceback.format_exc())
            mylog.add_log(traceback.format_exc())
            input(f"log file name:{mylog.get_log_name()} \r\nWaiting...")
            raise Exception(f"Unexpected Error,Please check {mylog.get_log_name()}")
    else:
        gdrive = gdrive_dataset()
        gdrive.get_dataset()
        gdrive.transfer_dataset()
