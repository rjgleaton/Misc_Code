from typing import List
from torch import nn
import numpy as np
import torch
import time
import pdb

from torch.nn.modules.batchnorm import BatchNorm1d

from environments.environment_abstract import Environment, State


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
    optimizer = torch.optim.SGD(nnet.parameters(), lr=0.00001, momentum = 0.9)
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
    nnet.eval()
    rewards = []
    #pdb.set_trace()
    #Check if is goal state
    for i, elem in enumerate(env.is_solved(states)):
        if elem == True:
            rewards.append(0.0) #Set value to 0 if goal state
        else:
            next_state_rewards = []

            expand = env.expand([states[i]])

            #pdb.set_trace()
            inputs = env.state_to_nnet_input(expand[0][0])
            inputs_tensor = torch.from_numpy(inputs).float()
            output = nnet(inputs_tensor)

            child = expand[0]
            costs = expand[1]

            for j in range(len(child)):
                #pdb.set_trace()
                if env.is_solved([child[0][j]]) == True: 
                    next_state_rewards.append(0.0+output[j][0].item())
                    #pdb.set_trace()
                else:
                    next_state_rewards.append(1.0+output[j][0].item())
                #next_state_rewards.append(1.0+output[j][0].item())
            rewards.append(min(next_state_rewards))
        #pdb.set_trace()
    #pdb.set_trace()
    return rewards
   
pass        
