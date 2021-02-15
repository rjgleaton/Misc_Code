from torch import nn
import torch.nn.functional as F
import numpy as np
import torch
import time


def get_nnet_model() -> nn.Module:
    return nn.Linear(81, 1)


def train_nnet(nnet: nn.Module, states_nnet: np.ndarray, outputs: np.ndarray, batch_size: int, num_itrs: int,
               train_itr: int):

    start = time.time()
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(nnet.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=100, gamma=0.996)

    target = torch.from_numpy(outputs).float()

    for epoch in range(num_itrs):

            outputs_tensor = torch.from_numpy(states_nnet).float()
            pred = nnet(outputs_tensor)

            loss = criterion(pred, target)
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()
            scheduler.step()

            if epoch % 100 == 0:
                print("Itr: ", epoch, ", loss: ", loss, ", lr: ",
                      optimizer.param_groups[0]['lr'], ", target_ctg: ", target.mean(), ", nnet_ctg: ", pred.mean(), ", Time: ", time.time()-start)


