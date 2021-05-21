from typing import List
from torch import nn
import numpy as np
import torch
import time
import pdb

from torch.nn.modules.batchnorm import BatchNorm1d
from environments.environment_abstract import Environment, State
from utils.misc_utils import flatten, unflatten


class Cost2Go(nn.Module):
    def __init__(self, num_inputs = 81, hidden_size1 = 100, hidden_size2 = 40, num_outputs = 1):
        super(Cost2Go, self).__init__()
        self.linear_block1 = nn.Sequential(
            nn.Linear(num_inputs, hidden_size1, bias=False),
            #nn.Dropout(p=0.2),
            nn.BatchNorm1d(hidden_size1),
            nn.ReLU(),
            #nn.Dropout(p=0.4)
        )
        self.linear_block2 = nn.Sequential(
            nn.Linear(hidden_size1, hidden_size2, bias=False),
            #nn.Dropout(p=0.2),
            nn.BatchNorm1d(hidden_size2),
            nn.ReLU(),
            #nn.Dropout(p=0.4)
        )
        self.output_block = nn.Sequential(
            nn.Linear(hidden_size2, num_outputs)
        )

    def forward(self, x):
        x = self.linear_block1(x)
        x = self.linear_block2(x)
        x = self.output_block(x)
        return x


def get_nnet_model() -> nn.Module:
    return Cost2Go()


def train_nnet(nnet: nn.Module, states_nnet: np.ndarray, outputs: np.ndarray, batchsize: int, num_itrs: int,
               train_itr: int):

    nnet.train()

    start = time.time()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(nnet.parameters(), lr=0.001)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.996)


    #convert inputs and outputs to tensors
    inputs_tensor = torch.from_numpy(states_nnet).float()
    target = torch.from_numpy(outputs).float()

    #data loader, breaks into batches of size 100
    data = torch.utils.data.TensorDataset(inputs_tensor, target)
    data_loader = torch.utils.data.DataLoader(
       data, batch_size=batchsize, shuffle=True, num_workers=4)

    #training loop
    for epoch, to_load in enumerate(data_loader, 0):

        inputs, labels = to_load
        pred = nnet(inputs)

        loss = criterion(pred, labels)
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        #scheduler.step()

        #if epoch % 100 == 0:
        #    print("Itr: ", train_itr+epoch, ", loss: ", loss.item(), ", lr: ",
        #          optimizer.param_groups[0]['lr'], ", target_ctg: ", labels.mean().item(), ", nnet_ctg: ", pred.mean().item(), ", Time: ", time.time()-start)
        #    start = time.time()
            

def value_iteration(nnet, device, env: Environment, states: List[State]) -> List[float]:
    
    children_cost_list = env.expand(states)
    children = children_cost_list[0]
    transition_costs = children_cost_list[1]

    flat_children, index_children = flatten(children)    

    #pdb.set_trace()
    inputs_tensor = torch.from_numpy(env.state_to_nnet_input(flat_children)).float()
    output_tensor = nnet(inputs_tensor).data.numpy()[:,0] 

    #pdb.set_trace()
    targets_list =  np.ndarray = np.array(transition_costs) + np.array(unflatten(list(output_tensor), index_children))
    targets = np.min(targets_list, axis = 1)

    #pdb.set_trace()
    is_solved: np.array = np.array(env.is_solved(states))

    #pdb.set_trace()
    targets = targets * np.logical_not(is_solved)

    #pdb.set_trace()
    return targets
   
pass   
