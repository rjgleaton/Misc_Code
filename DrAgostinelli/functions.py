from torch import nn
import torch.nn.functional as F
import numpy as np
import torch
import time

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
    optimizer = torch.optim.SGD(nnet.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=100, gamma=0.996)

    #get values to compare results against and tensor of inputs
    inputs_tensor = torch.from_numpy(states_nnet).float()
    target = torch.from_numpy(outputs).float()

    #data loader, breaks into batches of size 100
    data = torch.utils.data.TensorDataset(inputs_tensor, target)
    dataLoader = torch.utils.data.DataLoader(
       data, batch_size=batchsize, shuffle=True, num_workers=10)

    for epoch in range(num_itrs):
        nnet.train()
        
        for toLoad in dataLoader:
            inputs, labels = toLoad
            pred = nnet(inputs)

            loss = criterion(pred, labels)
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

        scheduler.step()
        if epoch % 100 == 0:
            print("Itr: ", epoch, ", loss: ", loss, ", lr: ",
                  optimizer.param_groups[0]['lr'], ", target_ctg: ", target.mean(), ", nnet_ctg: ", pred.mean(), ", Time: ", time.time()-start)


