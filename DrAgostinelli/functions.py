from typing import List
from torch import nn
import torch.nn.functional as F
import numpy as np
import torch
import time
import pdb

from environments.environment_abstract import Environment, State


class Cost2Go(nn.Module):
    def __init__(self):
        super(Cost2Go, self).__init__()
        #values chosen are arbitrary besides input and output
        self.fc1 = nn.Linear(81, 40)
        self.fc2 = nn.Linear(40, 20)
        self.fc3 = nn.Linear(20, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def get_nnet_model() -> nn.Module:
    return Cost2Go()


def train_nnet(nnet: nn.Module, states_nnet: np.ndarray, outputs: np.ndarray, batchsize: int, num_itrs: int,
               train_itr: int):

    start = time.time()
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(nnet.parameters(), lr=0.001, momentum = 0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=100, gamma=0.996)

    #get values to compare results against and tensor of inputs
    inputs_tensor = torch.from_numpy(states_nnet).float()
    target = torch.from_numpy(outputs).float()

    #data loader, breaks into batches of size 100
    data = torch.utils.data.TensorDataset(inputs_tensor, target)
    data_loader = torch.utils.data.DataLoader(
       data, batch_size=batchsize, shuffle=True, num_workers=4)

    for epoch, to_load in enumerate(data_loader, 0):
        nnet.train()
        
        inputs, labels = to_load
        pred = nnet(inputs)

        loss = criterion(pred, labels)
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        scheduler.step()

        if epoch % 100 == 0:
            print("Itr: ", epoch, ", loss: ", loss.item(), ", lr: ",
                  optimizer.param_groups[0]['lr'], ", target_ctg: ", labels.mean().item(), ", esnnet_ctg: ", pred.mean().item(), ", Time: ", time.time()-start)
            start = time.time()
pass

def value_iteration(nnet, device, env: Environment, states: List[State]) -> List[float]:
    theta = 1
    gamma = 0.9 #discount factor
    prob = 0.8 #probability of going to next state given action 
    children = env.expand(states) #tuple of list of children followed by list of path cost for each state
    rewards = np.invert(env.is_solved(states).astype(int)) #0 path cost for solved states, 1 for unsolved

    while True:
        delta = 0

        for i, v in enumerate(rewards):
            if v != 0:
                print("Ruh Roh Raggy!")
                next_state_rewards = []
                rewards_children = np.invert(env.is_solved(children[0][i]).astype(int))
                #Track each child of that state
                for j, child in enumerate(children[0][i]):
                    if rewards_children[j] == 0:
                        print("OMG!") 
                    #Find child in actual state and reward list
                    next_state_rewards.append(rewards_children[j]+children[1][i][j])
                    #pdb.set_trace()
                
                if len(next_state_rewards) != 0:
                    rewards[i] = min(next_state_rewards)
                    delta = max(delta, abs(v-rewards[i]))
            else:
                print("Zoinks scoob!")
        
        print(delta)
        if delta < theta:
            break
    
    return rewards
    
pass        
