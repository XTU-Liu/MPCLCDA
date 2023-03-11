import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from model import GTNCL
import pdb
import pickle
import argparse
from units import *
import datetime

import datetime
import os
import time



if __name__ == '__main__':

    t=datetime.datetime.now()

    _savePath = "terminal/%d" % time.time()
    os.mkdir(_savePath)

    parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset', type=str,
    #                     help='Dataset')
    parser.add_argument('--epoch', type=int, default=300,
                        help='Training Epochs')
    parser.add_argument('--node_dim', type=int, default=256,
                        help='Node dimension')
    parser.add_argument('--num_channels', type=int, default=2,
                        help='number of channels')
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-10,
                        help='l2 reg')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layer')
    parser.add_argument('--norm', type=str, default='true',
                        help='normalization')
    parser.add_argument('--adaptive_lr', type=str, default='false',
                        help='adaptive learning rate')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout')
    parser.add_argument('--dataset',type=str,default='CircR2Disease')



    args = parser.parse_args()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(args)
    dataset=args.dataset
    epochs = args.epoch
    node_dim = args.node_dim
    num_channels = args.num_channels
    lr = args.lr
    weight_decay = args.weight_decay
    num_layers = args.num_layers
    norm = args.norm
    dropout=args.dropout
    adaptive_lr = args.adaptive_lr

    data,node_features,edges=load_data1(dataset)
    cl=get_clGraph(data,"circd").to(device)

    lable=torch.tensor(data[:, 2:3]).to(device)


    num_nodes = edges[0].shape[0]

    for i,edge in enumerate(edges):
        if i ==0:
            A = torch.from_numpy(edge.todense()).type(torch.FloatTensor).unsqueeze(-1)
        else:
            A = torch.cat([A,torch.from_numpy(edge.todense()).type(torch.FloatTensor).unsqueeze(-1)], dim=-1)
    A = torch.cat([A,torch.eye(num_nodes).type(torch.FloatTensor).unsqueeze(-1)], dim=-1)

    node_features = torch.from_numpy(node_features).type(torch.FloatTensor)

    train,test=get_cross(data)



    num_classes = torch.max(lable).item()+1

    all_acc = []
    all_roc = []
    all_f1 = []
    pred = []
    pred_label = []
    best_test_auc = 0
    for i in range(len(train)):
        print('fold: ',i+1)

        train_index = train[i]
        test_index = test[i]

        model = GTNCL(num_edge=A.shape[-1],
                    num_channels=num_channels,
                    w_in=node_features.shape[1],
                    w_out=node_dim,
                    num_class=num_classes,
                    num_layers=num_layers,
                    norm=norm,
                    dropout=dropout)
        if adaptive_lr == 'false':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            optimizer = torch.optim.Adam([{'params': model.weight},
                                          {'params': model.linear1.parameters()},
                                          {'params': model.linear2.parameters()},
                                          {"params": model.layers.parameters(), "lr": 0.5}
                                          ], lr=0.005, weight_decay=0.005)
#            lr = 0.005, weight_decay = 0.001)
        loss = nn.CrossEntropyLoss()

        final_f1 = 0
        best_test_loss = 10000
        best_train_loss = 10000
        best_train_f1 = 0
        best_test_f1 = 0
        best_test_auc=0
        pred.append([])
        pred_label.append([])

        for j in range(epochs):
            for param_group in optimizer.param_groups:
                if param_group['lr'] > 0.05:
                    param_group['lr'] = param_group['lr'] * 0.9
            print('Epoch:  ',j+1)
            model.zero_grad()
            model.train()
            loss,y_train,Ws = model(data,A, node_features, train_index, lable[train_index],cl)
            loss=loss+0.01*get_L2reg(model.parameters())
            train_f1 = torch.mean(f1_score(torch.argmax(y_train.detach(),dim=1), lable[train_index], num_classes=num_classes)).cpu().numpy()
            train_auc,train_fpr,train_tpr=get_roc(torch.argmax(y_train.detach(),dim=1), lable[train_index])

            print('Train - Loss: {}, Macro_F1: {},auc:{}'.format(loss.detach().cpu().numpy(), train_f1,train_auc))
            loss.backward()
            optimizer.step()
            model.eval()
            # Valid
            with torch.no_grad():
                test_loss, y_test, W = model.forward(data,A, node_features, test_index, lable[test_index],cl)
                test_loss=test_loss+0.01*get_L2reg(model.parameters())
                test_f1 = torch.mean(
                    f1_score(torch.argmax(y_test, dim=1),lable[test_index], num_classes=num_classes)).cpu().numpy()
                test_auc,test_fpr,train_tpr = get_roc(torch.argmax(y_test.detach(), dim=1), lable[test_index])
                print('Test - Loss: {}, Macro_F1: {},auc:{}\n'.format(test_loss.detach().cpu().numpy(), test_f1,test_auc))
            if test_auc > best_test_auc:
                best_test_loss = test_loss.detach().cpu().numpy()
                best_train_loss = loss.detach().cpu().numpy()
                best_train_f1 = train_f1
                best_test_f1 = test_f1
                best_test_auc=test_auc
                pred[i]= y_test
                pred_label[i]=lable[test_index]
        print('---------------Best Results--------------------')
        print('Train - Loss: {}, Macro_F1: {}'.format(best_train_loss, best_train_f1))
        print('Test - Loss: {}, Macro_F1: {},auc:{}'.format(best_test_loss, best_test_f1,best_test_auc))
        df=np.hstack((pred[i].detach().cpu().numpy(), pred_label[i].detach().cpu().numpy()))
        df = pd.DataFrame(df)
        df.to_csv('./'+_savePath+'/roc'+str(i)+'.csv', mode='a',header=None, index=None)
        final_f1 += best_test_f1
        all_roc.append(best_test_auc)


    mean_roc=sum(all_roc)/len(all_roc)
    f = open('./'+_savePath+"/record.txt","a+",encoding="utf-8")
    f.write(f"\n--------tiem:{t}---lr:{lr}---weight_decay:{weight_decay}---------\n")
    f.write(f"{args}\n")
    f.write(f"negative1 all_roc:{all_roc}\n")
    f.write(f"negative1 mean_roc:{mean_roc}\n")
    f.close()
    print("all roc:",all_roc)
    print("mean:",mean_roc)



