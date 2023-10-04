import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from general_processor import Utils
import snntorch as snn
from snntorch import surrogate


from sklearn.model_selection import KFold
from tqdm import tqdm


from module import SGLnet
from prepro import adjacency_matrix, split_time, CustomDataset
from engines import train, evaluate
from utils import accuracy, load_checkpoint, save_checkpoint
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--title", type=str, default="Vit")
    parser.add_argument("--device", type=str, default="cuda:1")
    

    parser.add_argument("--batch_size", type=int, default=64)

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()


    np.random.seed(31415)
    torch.manual_seed(4)
    #################불러오기#####################

    base_path = './pre_data/'
    # exclude =  [38, 88, 89, 92, 100, 104]
    exclude =  [88,89,92,100]
    subjects = [n for n in np.arange(1,25) if n not in exclude]
    x, y = Utils.load(subjects, base_path=base_path)
    y_one_hot  = Utils.to_one_hot(y)
    dataset = CustomDataset(x, y_one_hot)    

    k_folds    = 5
    num_epochs = 100
    results_train = pd.DataFrame(columns=["0", "1", "2" ,"3" ,"4"], index=range(num_epochs))
    results_valid = pd.DataFrame(columns=["0", "1", "2" ,"3" ,"4"], index=range(num_epochs))
    

    kfold   = KFold(n_splits=k_folds, shuffle=True)
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        print(f'FOLD {fold}')
        print('--------------------------------')
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler  = torch.utils.data.SubsetRandomSampler(test_ids)
        
        trainloader = torch.utils.data.DataLoader(dataset, 
                        batch_size=args.batch_size, sampler=train_subsampler)
        testloader = torch.utils.data.DataLoader(dataset,
                        batch_size=args.batch_size, sampler=test_subsampler)
        
    
        model = SGLnet(8)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs * len(trainloader))\
        ###SPike LOSS##########
        loss_fn = nn.CrossEntropyLoss()
        
        metric_fn = accuracy
        torch.autograd.set_detect_anomaly(True)
        for epoch in tqdm(range(0, num_epochs), total = num_epochs):
            train_summary = train(trainloader, model, optimizer, scheduler, loss_fn, metric_fn, args.device)
            print(train_summary)

            # val_summary = evaluate(val_loader, model, loss_fn, metric_fn, args.device)


            
        
    








