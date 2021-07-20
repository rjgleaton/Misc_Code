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

import pickle, random, time

from to_implement.functions import get_nnet_model, train_nnet, value_iteration
import pdb, traceback, sys

#from torch.utils.tensorboard import SummaryWriter
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


    # train with supervised learning
    print("Training DNN\n")
    train_itr: int = 0
    #for vi_update in range(num_vi_updates):
    for vi_update in range(1):
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

    random_indexs = list(range(len(out_nnet_array)))
    random.shuffle(random_indexs)

    random_states: np.ndarray = []
    sample_expected: np.ndarray = []
    sample_outputs: np.ndarray = []

    for i in range(100):
        random_states.append(states[random_indexs[i]])
        sample_expected.append(outputs_array[random_indexs[i]])
        sample_outputs.append(out_nnet_array[random_indexs[i]])

    h_new: np.ndarray = approx_admissible_conv(env, out_nnet_array, outputs_array, states, random_states, sample_outputs, sample_expected)

    converted_sample: np.ndarray = []
    for i in range(100):
        converted_sample.append(h_new[random_indexs[i]])

    #before, after = plt.subplots()
    plt.scatter(sample_expected, sample_outputs, c = '000000', linewidths = 0.1)
    #plt.plot([0,0],[30,30], c = 'g')
    plt.axline([0,0],[30,30], linewidth =3, c = 'g')
    plt.ylabel('NNet output')
    plt.xlabel('Expected value')
    plt.title("Output vs Expected")
    plt.show()
    #before.savefig("preconversion.pdf")

    
    plt.scatter(sample_expected, converted_sample, c = '000000', linewidths = 0.1)
    plt.axline([0,0],[30,30], linewidth =3, c = 'g')
    plt.ylabel('Converted output')
    plt.xlabel('Expected value')
    plt.title("Converted Output vs Expected")
    plt.show() 
       

def approx_admissible_conv(env: Environment, nnet_output: np.ndarray, optimal_output: np.ndarray, states: List[State], sample_states: np.ndarray, sample_outputs: np.ndarray, sample_optimal: np.ndarray):

    try:

        h_admissible: Dict[State, float] = {sample_states[i]: 0 for i in range(len(sample_states))}
        is_solved: Dict[State, bool] = {sample_states[i]: False for i in range(len(sample_states))}

        complete_solve: bool = False
        while(complete_solve == False):
            h_new: Dict[State, float] = adjust_h_new(h_admissible, nnet_output, states, sample_states, sample_outputs)

            eta = 3.0
            count = 0
            for state in sample_states: 
                h_admissible[state], is_solved[state] = a_star_update(env, state, h_new, h_admissible[state]+eta, states, nnet_output)
                count += 1
                print(count)

            if False not in is_solved.values():
                complete_solve = True
        
        h_new: Dict[int, float] = adjust_h_new(h_admissible, nnet_output, states, sample_states, sample_outputs)

        return np.array(list(h_new.values()))


    except:
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)


def adjust_h_new(h_admissible: Dict, nnet_output: np.ndarray, states: List, sample_states: np.ndarray, sample_outputs: np.ndarray) -> Dict:

    #Get set of cut offs
    cutoffs: List = np.arange(0, int(max(sample_outputs))+1, 1).tolist()
    cutoffs.append(max(sample_outputs))
    
    #o_c_max is list of max overestimations for each cut off provided the output is less than the cut off
    #same indexs as cutoffs
    o_c_max: List = get_o_c_max(cutoffs, nnet_output, sample_outputs, sample_states, h_admissible, states)

    #eq (4) h'(x) = h(x) - o_c_max
    h_new: Dict[int, float] = {hash(sample_states[i]): sample_outputs[i] - o_c_max[np.argmin(np.argwhere(sample_outputs[i] <= cutoffs))] for i in range(len(sample_states))}

    return h_new

def get_o_c_max(cutoffs: List, nnet_output: np.ndarray, sample_outputs: np.ndarray, sample_states: np.ndarray, h_admissible: Dict, states: List) -> List:

    o_c_max: List = []

    for i in range(len(cutoffs)):
        curr_max = float('-inf')
        for j in np.argwhere(np.asarray(sample_outputs) < cutoffs[i]).flatten():
            #o_c_max = max(xEX|h(x)<c) (h(x) - h_a(x))
            curr_max = max(curr_max, sample_outputs[j] - h_admissible[sample_states[j]])
        o_c_max.append(curr_max)

    #eq (7)
    b = 0
    for i in range(len(o_c_max)):
        o_c_max[i] = max(o_c_max[i] - b, 0)

    return o_c_max

def a_star_update(env: Environment, state: State, h_new: Dict[State, float], max_step: float, states: List[State], nnet_output: np.ndarray) -> Tuple[float, bool]:
    update = AStarUpdate(env, state, h_new, max_step, states, nnet_output)
    max_cost_state = float('-inf')

    if update.is_solved_dict[update.curr_state] == True:
        return update.opened[update.curr_state], True
    
    while len(update.opened) > 0:
        update.step()

        if update.is_solved_dict[update.curr_state] == True:
            return update.opened[update.curr_state], True

        elif update.opened[update.curr_state] + update.h_new[update.curr_state] > max_step:
            return update.opened[update.curr_state], False

        elif update.opened[update.curr_state] + update.h_new[update.curr_state] > max_cost_state:
            max_cost_state = update.opened[update.curr_state] + update.h_new[update.curr_state]
        
        update.delete()
        
    return max_cost_state, False



class AStarUpdate:

    def __init__(self, env: Environment, state: State, h_new: Dict[State, float], max_step: float, states: List[State], nnet_output: np.ndarray):
        self.env: Environment = env
        self.curr_state: int = hash(state)
        self.states: List[State] = states
        #TODO Change all dictionary instances to have the hash of the state as the key, not the state itself
        self.h_new: Dict[int, float] = h_new
        self.nnet_output: np.ndarray = nnet_output
        self.max_step: float = max_step

        start = time.time()
        get_children: List[List[State]] = env.expand(self.states)[0]
        #self.children_list: Dict[int, List[State]] = {hash(states[i]): get_children[i] for i in range(len(states))}
        self.children_list: Dict[int, List[int]] = dict()
        for i, child_list in enumerate(get_children):
            self.children_list[hash(states[i])] = []
            for child in child_list:
                self.children_list[hash(states[i])].append(hash(self.states[self.states.index(child)]))
        end = time.time()
        print(end-start)

        self.nnet_output_dict: Dict[int, float] = {hash(states[i]): nnet_output[i] for i in range(len(states))}
        solved_list: np.ndarray = env.is_solved(states)
        self.is_solved_dict: Dict[int, bool]= {hash(states[i]): solved_list[i] for i in range(len(solved_list))}
        self.states_to_hash: Dict[State, int] = {self.states[i]: hash(self.states[i]) for i in range(len(self.states))}

        self.opened: Dict[int, float] = dict()
        self.closed: Dict[int, float] = dict()

        self.opened[self.curr_state] = 0.0


    def step(self):
        #Find smallest value in opened and add children to opened
        #TODO check is state in h_new, if not, use nnet_output
        self.curr_state = list(self.opened.items())[0][0]
        for key in self.opened:
            if key in self.h_new:
                if (self.opened[key] + self.h_new[key]) < (self.opened[self.curr_state] + self.h_new[self.curr_state]):
                    self.curr_state = key
            else:
                if (self.opened[key] + self.nnet_output_dict[key]) < (self.opened[self.curr_state] + self.nnet_output_dict[self.curr_state]):
                    self.curr_state = key

        valueOfMin = self.opened[self.curr_state]
        for child in self.children_list[self.curr_state]:
            pdb.set_trace()
            self.opened[hash(child)] = valueOfMin+1
        
    def delete(self):
        #add to closed delete from opened
        self.closed[self.curr_state] = self.opened[self.curr_state]
        del self.opened[self.curr_state]


@jit(nopython = True)
def get_children_list(get_children: List[List[State]], states: List[State])->Dict[int, List[int]]:
    children_list: Dict[int, List[int]] = dict()
    for i, child_list in enumerate(get_children):
        children_list[hash(states[i])] = []
        for child in child_list:
            children_list[hash(states[i])].append(hash(states[states.index(child)]))
    return children_list




'''
def approx_admissable_conv(env: Environment, nnet_output: np.ndarray, optimal_output: np.ndarray, states: List[State], sample_states: List, sample_expected: np.ndarray) -> np.ndarray:
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
    solved: np.ndarray = np.zeros_like(nnet_output) #0.0 for false, and 1.0 for true

    #while x is an element of the representative set and is_solved == false
    found_false = False
    x = 0
    while True:
        #if every state is solved break out, else start over loop
        if x >= len(solved):
            if found_false == False:
                break
            found_false = False
            x = 0
            
        if solved[x] == 0.0:
            found_false = True
            #adjust inadmissible hueristic 
            h_new = adjust_inadmissible_huerisitc(h_new, cut_offs, o_c_max, nnet_output, admissible_heur, b)

            #for x element X do -> h^a(x), is_solved(x) <- A*(x,h',h^a(x)+n)
            eta = 3.0
            
            poop = 0
            for i in range(len(nnet_output)):
                if states[i] in sample_states:
                    admissible_heur[i], solved[i] = a_star_update(env, states[i], states, h_new, admissible_heur[i]+eta)
                    print(poop)
                    poop += 1
                else:
                    solved[i] = 1.0 #super dumb work around


        x += 1
    
    #adjust h' one last time before returning
    h_new = adjust_inadmissible_huerisitc(h_new, cut_offs, o_c_max, nnet_output, admissible_heur, b)

    return h_new

#@vectorize(["np.ndarray(np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float)"], target ='cuda')
def adjust_inadmissible_huerisitc(h_new: np.ndarray, cut_offs: np.ndarray, o_c_max: np.ndarray, nnet_output: np.ndarray, admissible_heur: np.ndarray, b: float) -> np.ndarray:
    #get o_c_max
    o_c_max = get_oc_max(o_c_max, cut_offs, nnet_output, admissible_heur)
        
    for i in range(len(cut_offs)):
        o_c_max[i] = max(o_c_max[i] - b, 0)
        
    #set h'
    h_new = get_h_new(h_new, o_c_max, cut_offs, nnet_output)

    return h_new


@jit(nopython = True)
def get_oc_max(o_c_max: np.ndarray, cut_offs: np.ndarray, nnet_output: np.ndarray, admissible_heur: np.ndarray) -> np.ndarray:
    for i, c in enumerate(cut_offs):
        for j in range(len(nnet_output)):
            if nnet_output[j] <= c:
                o_c_max[i] = max(o_c_max[i], nnet_output[j]-admissible_heur[j])

    return o_c_max


@jit(nopython = True)
def get_h_new(h_new: np.ndarray, o_c_max: np.ndarray, cut_offs: np.ndarray, nnet_output: np.ndarray) -> np.ndarray:
    for i in range(len(h_new)):
        for j, c in enumerate(cut_offs):
            if nnet_output[i] <= c:
                h_new[i] = nnet_output[i] - o_c_max[j]
                break

    return h_new     


def a_star_update(env: Environment, state: State, states: List[State], h_new: np.ndarray, max_step: float) -> Tuple[float, float]:
    #pdb.set_trace()
    update = AStarUpdate(env, state, states, h_new, max_step)
    update.step()
    max_cost_state: State = state

    while len(update.opened) > 0:
        #check if goal state
        if env.is_solved([update.curr_state])[0] == True:
            return update.h_new_dict[states[states.index(update.curr_state)]], 1.0 
        #check if over max step
        elif update.closed[update.curr_state] + update.h_new_dict[states[states.index(update.curr_state)]] > max_step:
            #return update.h_new_dict[states[states.index(update.curr_state)]], 0.0
            return update.h_new_dict[states[states.index(update.curr_state)]], 0.0
        #check if greater than max cost state
        elif (update.closed[update.curr_state] + update.h_new_dict[states[states.index(update.curr_state)]]) > (update.closed[max_cost_state] + update.h_new_dict[states[states.index(max_cost_state)]]):
            max_cost_state = update.curr_state

        
        else:
            update.step()
    
    return max_cost_state, 1.0
    
    




class AStarUpdate:
    def __init__(self, env: Environment, state: State, states: List[State], h_new: np.ndarray, max_step: float):
        self.env: Environment = env
        self.states: List[State] = states
        self.max_step: float = max_step
        self.solved: bool = False
        self.curr_state: State = state

        #pdb.set_trace()
        self.h_new_dict: Dict[State, float] = {self.states[i]: h_new[i] for i in range(len(states))}
        self.opened: Dict[State, float] = dict()
        self.closed: Dict[State, float] = dict()

        self.opened[state] = 0.0

    def step(self):
        try:
            self.curr_state: State = list(self.opened.keys())[0]
            for s in self.opened:
                if (self.opened[s] + self.h_new_dict[self.states[self.states.index(s)]]) < (self.opened[self.curr_state] + self.h_new_dict[self.states[self.states.index(self.curr_state)]]):
                    self.curr_state = s

            #children to opened
            for child in self.env.expand([self.curr_state])[0][0]:
                #pdb.set_trace()
                if child not in self.closed:
                    self.opened[child] = self.opened[self.curr_state] + 1.0
                else:
                    if (self.opened[self.curr_state] + 1.0) < self.closed[child]:
                        del self.closed[child]
                        self.opened[child] = self.opened[self.curr_state] + 1.0
            #add to closed and delete from opened
            self.closed[self.curr_state] = self.opened[self.curr_state]
            del self.opened[self.curr_state]
        except:
            extype, value, tb = sys.exc_info()
            traceback.print_exc()
            pdb.post_mortem(tb)

'''
if __name__ == "__main__":
    main()

