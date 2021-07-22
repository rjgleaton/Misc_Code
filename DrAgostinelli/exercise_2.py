from typing import List, Tuple, Set, Dict
import numpy as np
from environments.n_puzzle import NPuzzle, NPuzzleState

import torch
from torch import nn

from environments.environment_abstract import Environment, State
from utils import env_utils
from utils.misc_utils import evaluate_cost_to_go
from utils.nnet_utils import states_nnet_to_pytorch_input
from utils.misc_utils import flatten

import pickle, random, time, threading, multiprocessing

from to_implement.functions import get_nnet_model, train_nnet, value_iteration
import pdb, traceback, sys


from matplotlib import pyplot as plt

from numba import jit, njit, vectorize, cuda, uint32, f8, uint8

def main():
    torch.set_num_threads(4)

    #Set up TesnorBoard writer
    #writer = SummaryWriter()

    # get environment
    env: Environment = env_utils.get_environment("puzzle8")

    # get nnet model
    nnet: nn.Module = get_nnet_model()
    device = torch.device('cpu')
    batch_size: int = 100
    num_itrs_per_vi_update: int = 200
    #num_vi_updates: int = 1
    num_vi_updates: int = 50

    # get data
    print("Preparing Data\n")
    data = pickle.load(open("data/data.pkl", "rb"))

    '''
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
    FILE = "model.pth"
    torch.save(nnet.state_dict(), FILE)
    '''
    FILE = "model.pth"
    nnet.load_state_dict(torch.load(FILE))
    generate_plot(nnet, device, env, data["states"], data["output"])

def generate_plot(nnet: nn.Module(), device, env: Environment, states: List[State], outputs: np.array):
    nnet.eval()

    states_targ_nnet: np.ndarray = env.state_to_nnet_input(states)
    out_nnet = nnet(states_nnet_to_pytorch_input(states_targ_nnet, device).float()).cpu().data.numpy()
    out_nnet, _ = flatten(out_nnet)
    outputs, _ = flatten(outputs)

    out_nnet_array = np.array(out_nnet)
    outputs_array = np.array(outputs)

    random_indexs = list(range(len(out_nnet_array)))
    random.shuffle(random_indexs)

    random_states: np.ndarray = []
    sample_expected: np.ndarray = []
    sample_outputs: np.ndarray = []

    for i in range(100):
        random_states.append(states[random_indexs[i]])
        sample_expected.append(outputs_array[random_indexs[i]])
        sample_outputs.append(out_nnet_array[random_indexs[i]])

    h_new: np.ndarray = approx_admissible_conv(env, nnet, out_nnet_array, outputs_array, states, random_states, sample_outputs, sample_expected)

    #before, after = plt.subplots()
    plt.scatter(sample_expected, sample_outputs, c = '000000', linewidths = 0.1)
    #plt.plot([0,0],[30,30], c = 'g')
    plt.axline([0,0],[30,30], linewidth =3, c = 'g')
    plt.ylabel('NNet output')
    plt.xlabel('Expected value')
    plt.title("Output vs Expected")
    plt.show()
    #before.savefig("preconversion.pdf")

    
    plt.scatter(sample_expected, h_new, c = '000000', linewidths = 0.1)
    plt.axline([0,0],[30,30], linewidth =3, c = 'g')
    plt.ylabel('Converted output')
    plt.xlabel('Expected value')
    plt.title("Converted Output vs Expected")
    plt.show() 
       

def approx_admissible_conv(env: Environment, nnet: nn.Module(), nnet_output: np.ndarray, optimal_output: np.ndarray, states: List[State], sample_states: np.ndarray, sample_outputs: np.ndarray, sample_optimal: np.ndarray) -> np.ndarray:

    try:
        start = time.time()
        h_admissible: Dict[State, float] = {sample_states[i]: 0 for i in range(len(sample_states))}
        is_solved: Dict[State, bool] = {sample_states[i]: False for i in range(len(sample_states))}

        complete_solve: bool = False
        while(complete_solve == False):
            h_new: Dict[State, float] = adjust_h_new(h_admissible, nnet_output, states, sample_states, sample_outputs)

            eta = 5.0
            count = 0
            for state in sample_states:
                if is_solved[state] is not True: 
                    h_admissible[state], is_solved[state] = a_star_update(env, nnet, state, h_new, h_admissible[state]+eta, states, nnet_output)
                count += 1
                print(count)

            if False not in is_solved.values():
                complete_solve = True

            unsolved_count = 0
            for value in list(is_solved.values()):
                if value == False:
                    unsolved_count += 1
            
            print("Number of unsolved states: ",unsolved_count)
            #if unsolved_count == 100 or unsolved_count < 10:
            '''
            plt.scatter(sample_optimal, np.array(list(h_new.values())), c = '000000', linewidths = 0.1)
            plt.axline([0,0],[30,30], linewidth =3, c = 'g')
            plt.ylabel('Converted output')
            plt.xlabel('Expected value')
            plt.title("Converted Output vs Expected")
            plt.show() 
            '''
        h_new: Dict[int, float] = adjust_h_new_final(h_admissible, nnet_output, states, sample_states, sample_outputs)

        end = time.time()
        print('Total time: ', end-start)
        return np.array(list(h_new.values()))


    except:
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)


def adjust_h_new(h_admissible: Dict, nnet_output: np.ndarray, states: List, sample_states: np.ndarray, sample_outputs: np.ndarray) -> Dict[State, float]:

    #Get set of cut offs
    cutoffs: List = np.arange(0, int(max(sample_outputs))+1, 1).tolist()
    cutoffs.append(max(sample_outputs))
    
    #o_c_max is list of max overestimations for each cut off provided the output is less than the cut off
    #same indexs as cutoffs
    o_c_max: List = get_o_c_max(cutoffs, nnet_output, sample_outputs, sample_states, h_admissible, states)

    #eq (4) h'(x) = h(x) - o_c_max

    h_new: Dict[int, float] = {hash(sample_states[i]): sample_outputs[i] - o_c_max[np.argwhere(sample_outputs[i] <= cutoffs).flatten()[0]] for i in range(len(sample_states))}
    #pdb.set_trace()
    return h_new

def adjust_h_new_final(h_admissible: Dict, nnet_output: np.ndarray, states: List, sample_states: np.ndarray, sample_outputs: np.ndarray) -> Dict[State, float]:

    #Get set of cut offs
    cutoffs: List = np.arange(0, int(max(sample_outputs))+1, 1).tolist()
    cutoffs.append(max(sample_outputs))
    
    #o_c_max is list of max overestimations for each cut off provided the output is less than the cut off
    #same indexs as cutoffs
    o_c_max: List = get_o_c_max_final(cutoffs, nnet_output, sample_outputs, sample_states, h_admissible, states)

    #eq (4) h'(x) = h(x) - o_c_max

    h_new: Dict[int, float] = {hash(sample_states[i]): sample_outputs[i] - o_c_max[np.argwhere(sample_outputs[i] <= cutoffs).flatten()[0]] for i in range(len(sample_states))}
    #pdb.set_trace()
    return h_new

def get_o_c_max(cutoffs: List, nnet_output: np.ndarray, sample_outputs: np.ndarray, sample_states: np.ndarray, h_admissible: Dict, states: List) -> List:

    o_c_max: List = []

    for i in range(len(cutoffs)):
        curr_max = float('-inf')
        for j in np.argwhere(np.asarray(sample_outputs) < cutoffs[i]).flatten():
            #o_c_max = max(xEX|h(x)<c) (h(x) - h_a(x))
            curr_max = max(curr_max, sample_outputs[j] - h_admissible[sample_states[j]])
        o_c_max.append(curr_max)
    
    return o_c_max

def get_o_c_max_final(cutoffs: List, nnet_output: np.ndarray, sample_outputs: np.ndarray, sample_states: np.ndarray, h_admissible: Dict, states: List) -> List:

    o_c_max: List = []

    for i in range(len(cutoffs)):
        curr_max = float('-inf')
        for j in np.argwhere(np.asarray(sample_outputs) < cutoffs[i]).flatten():
            #o_c_max = max(xEX|h(x)<c) (h(x) - h_a(x))
            curr_max = max(curr_max, sample_outputs[j] - h_admissible[sample_states[j]])
        o_c_max.append(curr_max)

    #pdb.set_trace()
    #eq (7)
    b = 0
    for i in range(len(o_c_max)):
        o_c_max[i] = max(o_c_max[i] - b, 0)

    return o_c_max

#TODO might have to change it so it ALWAYS returns the max cost of all nodes expanded
def a_star_update(env: Environment, nnet: nn.Module(), state: State, h_new: Dict[State, float], max_step: float, states: List[State], nnet_output: np.ndarray) -> Tuple[float, bool]:
    update = AStarUpdate(env, nnet, state, h_new[hash(state)], max_step, states, nnet_output)
    max_cost_state = float('-inf')

    if env.is_solved([state])[0] == True:
        return update.opened[update.curr_state][0], True
    
    while len(update.opened) > 0:
        update.step()

        if update.env.is_solved([update.curr_state])[0] == True:
            return update.opened[update.curr_state][1], True

        elif update.opened[update.curr_state][0] + update.opened[update.curr_state][1] > max_step:
            return update.opened[update.curr_state][0] + update.opened[update.curr_state][1], False   
        
        elif update.opened[update.curr_state][0] + update.opened[update.curr_state][1] > max_cost_state:
            max_cost_state = update.opened[update.curr_state][0] + update.opened[update.curr_state][1]
        
        update.delete()

    #Not sure if this should be considered solved or not    
    del update
    return max_cost_state, True



class AStarUpdate:

    def __init__(self, env: Environment, nnet: nn.Module(), state: State, h_new: float, max_step: float, states: List[State], nnet_output: np.ndarray):
        self.env: Environment = env
        self.nnet: nn.Module() = nnet
        self.curr_state: State = state
        self.states: List[State] = states
        self.max_step: float = max_step
        
        #First float is heuristic, second is path cost/length
        self.opened: Dict[State, Tuple[float, float]] = {state: [h_new, 0.0]}
        self.closed: Dict[State, Tuple[float, float]] = dict()

    def step(self):
        self.nnet.eval()
        #Find lowest cost state
        lowest_cost_state: State = list(self.opened.keys())[0]
        for item in self.opened:
            if self.opened[lowest_cost_state][0] + self.opened[lowest_cost_state][1] > self.opened[item][0] + self.opened[item][1]:
                lowest_cost_state = item
        self.curr_state = lowest_cost_state

        #Add Children to open
        children = self.env.expand([lowest_cost_state])[0][0]
        children_output = self.nnet(states_nnet_to_pytorch_input(self.env.state_to_nnet_input(children), 'cpu').float()).cpu().data.numpy()
        for i, child in enumerate(children):
            if child in self.closed:
                if (children_output[i][0] + self.opened[lowest_cost_state][1]+1.0) < (self.closed[child][0] + self.closed[child][1]):
                    self.opened[child] = children_output[i][0], self.opened[lowest_cost_state][1]+1.0
                else:
                    continue
            else:
                self.opened[child] = children_output[i][0], self.opened[lowest_cost_state][1]+1.0
        

    def delete(self):
        #add to closed delete from opened
        self.closed[self.curr_state] = self.opened[self.curr_state]
        del self.opened[self.curr_state]



if __name__ == "__main__":
    main()
