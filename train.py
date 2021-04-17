import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy
import pickle
from model import *
import argparse
from utils import load_citation, accuracy
import torch.optim as optim
import nni
import time

parser = argparse.ArgumentParser()
parser.add_argument('--data', default='cora', help='dateset')
parser.add_argument('--seed', type=int, default=2021, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate.')
parser.add_argument('--layer', type=int, default=16, help='Number of layers.')
# parser.add_argument('--hidden', type=int, default=512, help='hidden dimensions.')
parser.add_argument('--patience', type=int, default=200, help='Patience')  # 100
parser.add_argument('--dev', type=int, default=0, help='device id')
parser.add_argument('--variant', action='store_true', default=False, help='GCN* model.')
parser.add_argument('--test', action='store_true', default=False, help='evaluation on test set.')
args = parser.parse_args()
print(args)

support00 = np.load('supports/'+args.data+'_support00.npy')
support01 = np.load('supports/'+args.data+'_support01.npy')
support10 = np.load('supports/'+args.data+'_support10.npy')
support11 = np.load('supports/'+args.data+'_support11.npy')

adj, features, labels, idx_train, idx_val, idx_test = load_citation(args.data)
cudaid = "cuda:"+str(args.dev)
device = torch.device(cudaid)
features = features.to(device)
adj = adj.to(device)
support0 = torch.sparse_coo_tensor(support00, support01, (adj.shape[0], adj.shape[0]))
support1 = torch.sparse_coo_tensor(support10, support11, (adj.shape[0], adj.shape[0]))
support0 = support0.to(device).to(torch.float32)
support1 = support1.to(device).to_dense().to(torch.float32)

def build_and_train(hype_space):
    modelname = 'nof'
    modeltype = {'nof':nof,'GWCNII': GWCNII, 'GCNII': GCNII, 'combine': combinemodel, }
    model = modeltype[modelname](
        mydev=device,
        myf=hype_space['myf'],
        support0=support0,
        support1=support1,
        adj=adj,
        gamma=hype_space['gamma'],
        nnode=features.shape[0],
        nfeat=features.shape[1],
        nlayers=args.layer,
        nhidden=hype_space['hidden'],
        nclass=int(labels.max()) + 1,
        dropout=hype_space['dropout'],
        lamda=hype_space['lambda'],
        alpha=hype_space['alpha'],
        variant=args.variant).to(device)

    optimizer = optim.Adam([{'params': model.params1, 'weight_decay': hype_space['wd1']},
                            {'params': model.params2, 'weight_decay': hype_space['wd2']},],
                           lr=0.1*hype_space['lr_rate_mul'])

    def train():
        model.train()
        optimizer.zero_grad()
        output=model(features)
        acc_train = accuracy(output[idx_train], labels[idx_train].to(device))
        loss_train = F.nll_loss(output[idx_train], labels[idx_train].to(device))
        loss_train.backward()
        optimizer.step()
        return loss_train.item(), acc_train.item()

    def validate():
        model.eval()
        with torch.no_grad():
            output=model(features)
            loss_val = F.nll_loss(output[idx_val], labels[idx_val].to(device))
            acc_val = accuracy(output[idx_val], labels[idx_val].to(device))
            return loss_val.item(), acc_val.item()

    def test():
        # model.load_state_dict(torch.load(checkpt_file))
        model.eval()
        with torch.no_grad():
            output=model(features)
            loss_test = F.nll_loss(output[idx_test], labels[idx_test].to(device))
            acc_test = accuracy(output[idx_test], labels[idx_test].to(device))
            return loss_test.item(), acc_test.item()

    bad_counter = 0
    best = 999999999
    best_test_acc = 0
    lrnow = optimizer.state_dict()['param_groups'][0]['lr']
    for epoch in range(args.epochs):
        loss_tra, acc_tra = train()
        loss_val,acc_val = validate()
        loss_tes,acc_tes = test()
        # nni.report_intermediate_result(acc_tes)

        if epoch==0 or loss_val < best:
            best = loss_val
            best_epoch = epoch
            acc = acc_val
            bad_counter = 0
        else:
            bad_counter += 1

        if acc_tes > best_test_acc:
            best_test_acc = acc_tes
            best_test_epoch = epoch

        print('Epoch:%4d|train loss:%.3f acc:%.2f|val loss:%.3f acc:%.2f|test loss:%.3f acc:%.2f|best acc:%.2f epoch:%d'
                %(epoch + 1,loss_tra,acc_tra * 100,loss_val,acc_val * 100,loss_tes,acc_tes*100,best_test_acc*100,best_test_epoch))
        if bad_counter == args.patience:
            print('early stopping at epoch %d'%epoch)
            break
    # nni.report_final_result(best_test_acc)
    return

if __name__ == "__main__":
    t1=time.time()
    params={"lr_rate_mul":0.1,"alpha":0.3,"lambda":0.4,"gamma":0.2,"wd1":0.05,"wd2":0.001,"dropout":0.7,"hidden":64,"myf":1.2}
    build_and_train(params)
    t2=time.time()
    print('time= %.2f s'%(t2-t1))
    # params=nni.get_next_parameter()
    # build_and_train(params)
