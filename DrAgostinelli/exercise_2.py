from typing import List, Tuple
import numpy as np

import torch
from torch import nn

from environments.environment_abstract import Environment, State
from utils import env_utils
from utils.misc_utils import evaluate_cost_to_go
from utils.nnet_utils import states_nnet_to_pytorch_input
from utils.misc_utils import flatten

import pickle

from to_implement.functions import get_nnet_model, train_nnet, value_iteration
import pdb, traceback, sys

#from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt

from numba import jit, njit, vectorize, cuda, uint32, f8, uint8

def main():
    torch.set_num_threads(1)

    #Set up TesnorBoard writer
    #writer = SummaryWriter()

    # get environment
    env: Environment = env_utils.get_environment("puzzle8")

    # get nnet model
    nnet: nn.Module = get_nnet_model()
    device = torch.device('cpu')
    batch_size: int = 100
    num_itrs_per_vi_update: int = 200
    num_vi_updates: int = 1
    #num_vi_updates: int = 50

    # get data
    print("Preparing Data\n")
    data = pickle.load(open("data/data.pkl", "rb"))


    # train with supervised learning
    print("Training DNN\n")
    train_itr: int = 0
    for vi_update in range(num_vi_updates):
    #for vi_update in range(1):
        print("--- Value Iteration Update: %i ---" % vi_update)
        states: List[State] = env.generate_states(batch_size*num_itrs_per_vi_update, (0, 500))
        #states: List[State] = env.generate_states(200, (0, 500))

        states_nnet: np.ndarray = env.state_to_nnet_input(states)

        inputs_tensor = torch.from_numpy(states_nnet).float()
        #writer.add_graph(nnet, inputs_tensor)
       

        outputs_np = value_iteration(nnet, device, env, states)
        outputs = np.expand_dims(np.array(outputs_np), 1)

        nnet.train()
        train_nnet(nnet, states_nnet, outputs, batch_size, num_itrs_per_vi_update, train_itr)

        nnet.eval()
        evaluate_cost_to_go(nnet, device, env, data["states"], data["output"])

        #pdb.set_trace()

        train_itr = train_itr + num_itrs_per_vi_update

    #writer.close()
    #pdb.set_trace()
    generate_plot(nnet, device, env, data["states"], data["output"])

def generate_plot(nnet: nn.Module(), device, env: Environment, states: List[State], outputs: np.array):
    nnet.eval()

    states_targ_nnet: np.ndarray = env.state_to_nnet_input(states)
    out_nnet = nnet(states_nnet_to_pytorch_input(states_targ_nnet, device).float()).cpu().data.numpy()
    out_nnet, _ = flatten(out_nnet)
    outputs, _ = flatten(outputs)

    out_nnet_array = np.array(out_nnet)
    outputs_array = np.array(outputs)
    #pdb.set_trace()
    
    h_new: np.ndarray = approx_admissable_conv(env, outputs_array, outputs_array, states)

    before, after = plt.subplots()
    before.scatter(outputs_array, out_nnet_array, c = '000000', linewidths = 0.1)
    #plt.plot([0,0],[30,30], c = 'g')
    before.axline([0,0],[30,30], linewidth =3, c = 'g')
    before.ylabel('NNet output')
    before.xlabel('Expected value')
    before.title("Output vs Expected")
    before.show()

    after.scatter(outputs_array, h_new, c = '000000', linewidths = 0.1)
    after.axline([0,0],[30,30], linewidth =3, c = 'g')
    after.ylabel('Converted output')
    after.xlabel('Expected value')
    after.title("Converted Output vs Expected")
    after.show()    
    

def approx_admissable_conv(env: Environment, nnet_output: np.ndarray, optimal_output: np.ndarray, states: List[State]) -> np.ndarray:
    admissible_heur: np.ndarray =  np.zeros_like(nnet_output)
    

    h_new: np.ndarray = np.copy(admissible_heur)
    b = 3.0

    #create user defined set of cutoffs
    cut_offs: np.ndarray = np.array([])
    i = 0
    max_h = max(nnet_output)
    while i < max_h:
        cut_offs = np.append(cut_offs, [i])
        i = i+1
    cut_offs = np.append(cut_offs, max_h)

    o_c_max: np.ndarray = np.zeros_like(cut_offs)

    #while x is an element of the representative set and is_solved == false
    x = 0

    while x < len(nnet_output): 

        #adjust inadmissible hueristic 
        h_new = adjust_inadmissible_huerisitc(h_new, cut_offs, o_c_max, nnet_output, admissible_heur, b)

        #for x element X do -> h^a(x), is_solved(x) <- A*(x,h',h^a(x)+n)
        eta = 10.0
        for x in range(len(nnet_output)):
            #pdb.set_trace()
            print(x)
            admissible_heur[x], solved = a_star_update(env, states[x], states, h_new, admissible_heur[x]+eta)
        
        if solved == 1.0: #1.0 == True
            break

        if x % 100 == 0:
            print("Current progress: "+(x/len(solved))*100+"%")

        x += 1
    
    #adjust h' one last time before returning
    h_new = adjust_inadmissible_huerisitc(h_new, cut_offs, o_c_max, nnet_output, admissible_heur, b)

    return h_new

#@vectorize(["np.ndarray(np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float)"], target ='cuda')
def adjust_inadmissible_huerisitc(h_new: np.ndarray, cut_offs: np.ndarray, o_c_max: np.ndarray, nnet_output: np.ndarray, admissible_heur: np.ndarray, b: float) -> np.ndarray:
    #get o_c_max
    for i, c in enumerate(cut_offs):
        #pdb.set_trace()
        for j in range(len(nnet_output)):
            if nnet_output[j] <= c:
                #pdb.set_trace()
                o_c_max[i] = max(o_c_max[i], nnet_output[j]-admissible_heur[j])
        
    for i in range(len(cut_offs)):
        o_c_max[i] = max(o_c_max[i] - b, 0)
        
    #set h'
    for i in range(len(h_new)):
        for j, c in enumerate(cut_offs):
            if nnet_output[i] <= c:
                #pdb.set_trace()
                h_new[i] = nnet_output[i] - o_c_max[j]
                break 

    #pdb.set_trace()
    return h_new

#@jit(nopython = True)
def a_star_update(env: Environment, state: State, states: List[State], h_new: np.ndarray, max_step) -> Tuple[float, float]:
    children = env.expand(states)
    opened = np.empty(shape=(1,2), dtype = object)
    closed = np.empty(shape=(1,2), dtype = object)
    
    opened = np.append(opened, [[state, h_new[states.index(state)]]], axis = 0)
    opened = np.delete(opened, 0, 0)

    #pdb.set_trace()

    max_cost_state: float = 0.0
    try:
        while len(opened) > 0:
            #pdb.set_trace()
            lowest_cost_state = opened[np.argmin(opened[:,1])]

            #check if lowest cost state is goal state
            if env.is_solved([lowest_cost_state[0]])[0] == True:
                return lowest_cost_state[1], 1.0

            #Check if over max step limit
            if lowest_cost_state[1] > max_step:
                return lowest_cost_state[1], 0.0

            #check if greater than max cost state:
            if lowest_cost_state[1] > max_cost_state:
                max_cost_state = lowest_cost_state[1]

            #add children of state to opened
            index_of_LCS = states.index(lowest_cost_state[0])
            for c in range(len(children[0][index_of_LCS])):
                #check if child in closed:
                new_child = [children[0][index_of_LCS][c], lowest_cost_state[1]+children[1][index_of_LCS][c]]
                #pdb.set_trace()
                if np.isin(new_child[0], closed) == False:
                    opened = np.append(opened, [new_child], axis = 0)
                else:
                    #if lower cost state found add back to open, not sure if theres a built in function to do this
                    #pdb.set_trace()
                    for i, x in enumerate(closed):
                        #pdb.set_trace()
                        if type(x[0]) != type(None) and new_child[0] == x[0] and new_child[1] < x[1]:
                            opened = np.append(opened, [new_child], axis = 0)
                            closed = np.delete(closed, i, 0)

            #add to closed and delete from opened
            closed = np.append(closed, [lowest_cost_state], axis = 0)
            #pdb.set_trace()
            if len(closed) > 0 and np.isin(True, np.isin(closed, None)) == True: #really dumb work around, not sure how else to solve it tho
                closed = np.delete(closed, 0, 0)
                #pdb.set_trace() #check closed
            #pdb.set_trace()

            for i, x in enumerate(opened):
                #pdb.set_trace()
                if all(x == lowest_cost_state):
                    #pdb.set_trace()
                    opened = np.delete(opened, i, 0)
                    #pdb.set_trace()

        return max_cost_state, 0.0
    except:
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)



    '''
    #opened: np.array(Tuple[State, float]) = [[]]
    opened = np.empty(shape=(1,2), dtype = object)

    #set initial state into opened
    #pdb.set_trace() #make sure np.where works :)
    opened = np.append(opened, [[state, h_new[states.index(state)]]], axis = 0)
    opened = np.delete(opened, 0, 0)

    closed = np.empty(shape=(1,2), dtype = object)

    max_cost_state: float = 0.0
    while opened.size > 0:
        #Get state in opened with lowest cost
        lowest_cost_state = opened[0]
        for x in opened:
            #pdb.set_trace()
            if x[1] < lowest_cost_state[1]:
                lowest_cost_state = x

        #Check if goal state
        if env.is_solved([lowest_cost_state[0]])[0] == True:
            return lowest_cost_state[1], 1.0

        #Check if over max step limit
        if lowest_cost_state[1] > max_step:
            return lowest_cost_state[1], 0.0

        #check if greater than max cost state:
        if lowest_cost_state[1] > max_cost_state:
            max_cost_state = lowest_cost_state[1]

        #Get the children of the state with the lowest cost
        children = env.expand([lowest_cost_state[0]])
        #set path cost of children
        for i in range(len(children[1][0])):
            #pdb.set_trace()
            children[1][0][i] = lowest_cost_state[1] + h_new[states.index(children[0][0][i])]

        for i in range(len(children[0])):
            opened = np.append(opened, [[children[0][0][i], children[1][0][i]]], axis = 0)

        #add to closed and delete from opened
        closed = np.append(closed, lowest_cost_state)
        if closed[0] == [None, None]:
            closed = np.delete(closed, 0, 0)
        #pdb.set_trace() #check if np.delete works
        #pdb.set_trace()
        opened = np.delete(opened, np.where(opened == lowest_cost_state), axis = 0)

    return max_cost_state, 0.0
    '''
if __name__ == "__main__":
    main()
