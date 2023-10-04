import torch

from utils import AverageMeter

from prepro import adjacency_matrix, split_time

def train(loader, model, optimizer, scheduler, loss_fn, metric_fn, device):
    model.train()
    model.cuda(device)
    loss_mean = AverageMeter()
    metric_mean = AverageMeter()
    
    for idx, (inputs, targets) in enumerate(loader):
 
        adj_matrix = adjacency_matrix(inputs, 6)
        print(adj_matrix)
        inputs = split_time(inputs)
        adj_matrix = adj_matrix.cuda(device)
        inputs = inputs.cuda(device)
        targets = targets.cuda(device)
        
        spk, mem = model(inputs, adj_matrix)
        
        for j in range(8):
            loss = loss_fn(mem[j], targets)
        metric = metric_fn(spk, targets)

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        # loss.backward()
        optimizer.step()
        
        loss_mean.update(loss.to('cpu'))
        
        metric_mean.update(metric)
        # print("loss_mean" , metric_mean.avg)
        scheduler.step()

    summary = {'loss': loss_mean.avg, 'metric': metric_mean.avg}
    

    return summary


def evaluate(loader, model, loss_fn, metric_fn, device):
    model.eval()
    loss_mean = AverageMeter()
    metric_mean = AverageMeter()
    
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        with torch.no_grad():
            outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        metric = metric_fn(outputs, targets)

        loss_mean.update(loss.to('cpu'))
        metric_mean.update(metric)
    
    summary = {'loss': loss_mean.avg, 'metric': metric_mean.avg}

    return summary