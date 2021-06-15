import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import numpy as np
import csv
import os

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
 
def get_device():
    '''Get device (if GPU is available, use GPU) '''
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def plot_learning_curve(loss_record, title=''):
    ''' Plot learning curve of your DNN (train & dev loss) '''
    total_steps = len(loss_record['train'])
    x_1 = range(total_steps)
    x_2 = x_1[::len(loss_record['train']) // len(loss_record['dev'])]
    figure(figsize=(6, 4))
    plt.plot(x_1, loss_record['train'], c='tab:red', label='train')
    plt.plot(x_2, loss_record['dev'], c='tab:cyan', label='dev')
    plt.ylim(0.0, 5.)
    plt.xlabel('Training steps')
    plt.ylabel('MSE loss')
    plt.title('Learning curve of {}'.format(title))
    plt.legend()
    plt.show()

def plot_pred(dv_set, model, device, lim=35., preds=None, targets=None):
    ''' Plot prediction of your DNN '''
    if preds is None or targets is None:
        model.eval()
        preds, targets = [], []
        for x, y in dv_set:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                preds.append(pred.detach().cpu())
                targets.append(y.detach().cpu())
        preds = torch.cat(preds, dim=0).numpy()
        targets = torch.cat(targets, dim=0).numpy()

    figure(figsize=(5, 5))
    plt.scatter(targets, preds, c='r', alpha=0.5)
    plt.plot([-0.2, lim], [-0.2, lim], c='b')
    plt.xlim(-0.2, lim)
    plt.ylim(-0.2, lim)
    plt.xlabel('ground truth value')
    plt.ylabel('predicted value')
    plt.title('Ground Truth v.s. Prediction')
    plt.show()

class COVID19DDataset(Dataset):
    def __init__(self, path, mode='train', target_only=False):
        self.mode = mode
        with open(path, 'r') as fp:
            data = list(csv.reader(fp))
            data = np.array(data[1:])[:,1:].astype(float)

        if not target_only:
            feats = list(range(93))
        else:
            feats = list(range(40))
            feats.append(57)
            feats.append(75)

        if mode =='test':
            data = data[:, feats] #93
            self.data = torch.FloatTensor(data)
        else:
            target = data[:, -1] #94
            data = data[:, feats]

            if mode == 'train':
                indices = [i for i in range(len(data)) if i%10 != 0]#每10个人中第一个人加入dev
            if mode == 'dev':
                indices = [i for i in range(len(data)) if i%10 == 0]

            self.data = torch.FloatTensor(data[indices])
            self.target = torch.FloatTensor(target[indices])

        self.data[:, 40:] = (self.data[:, 40:] - self.data[:, 40:].mean(dim=0, keepdim=True))/(self.data[:, 40:].std(dim=0, keepdim=True))
        self.dim = self.data.shape[1]
        print("Finished reading the {} set of COVID19 Dataset ({} samples found,each dim = {})".format(mode, len(self.data), self.dim))
    def __getitem__(self, index):
        if self.mode in ["train", "dev"]:
            return self.data[index], self.target[index]
        else:
            return self.data[index]
    def __len__(self):
        return len(self.data)

def pre_dataloader(path, mode, batch_size, n_jobs=0, target_only=False):
    """
    pin_memory就是锁页内存，创建DataLoader时，设置pin_memory=True，则意味着生成的Tensor数据最开始是属于内存中的锁页内存，这样将内存的Tensor转义到GPU的显存就会更快一些。
    主机中的内存，有两种存在方式，一是锁页，二是不锁页，锁页内存存放的内容在任何情况下都不会与主机的虚拟内存进行交换（注：虚拟内存就是硬盘），而不锁页内存在主机内存不足时，数据会存放在虚拟内存中。
    而显卡中的显存全部是锁页内存！
    当计算机的内存充足的时候，可以设置pin_memory=True。当系统卡住，或者交换内存使用过多的时候，设置pin_memory=False。因为pin_memory与电脑硬件性能有关，pytorch开发者不能确保每一个炼丹玩家都有高端设备，因此pin_memory默认为False。
    """
    dataset = COVID19DDataset(path, mode=mode, target_only=target_only)
    datalaoder = DataLoader(
            dataset,
            batch_size,
            shuffle=(mode=="train"),
            drop_last=False,#将不足一个batch的数据丢弃
            num_workers=n_jobs,
            pin_memory=True)
    return datalaoder

class NeuralNet(nn.Module):
    def __init__(self, input_dim):
        super(NeuralNet, self).__init__()#继承父类中的所有self

        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        self.criterion = nn.MSELoss(reduction='mean')
    def forward(self, x):
        return self.net(x).squeeze(1)
        #batch = 64 ([64,1]) ->(64,)

    def cal_loss(self, pred, target):
        return self.criterion(pred, target)

def val(dev_set, model, device):
    model.eval()
    total_loss = 0
    for data, label in dev_set:
        data, label = data.to(device), label.to(device)
        with torch.no_grad():
            pred = model(data)
            mse_loss = model.cal_loss(pred, label)
        total_loss += mse_loss.detach().cpu().item() * len(data)
    total_loss = total_loss/len(dev_set.dataset)

    return total_loss

def train(tr_set, dev_set, model, config, device):
    n_epochs = config['n_epochs']
    #获得torch.optim.optimizer属性  ==  torch.optim.SGD
    #*args表示任何多个无名参数，它本质是一个tuple
    #** kwargs表示关键字参数，它本质上是一个dic

    optimizer = getattr(torch.optim, config['optimizer'])(model.parameters(), **config['optim_hparas'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1, last_epoch=-1)

    min_mse = 1000.  #初始化最小的损失，保存在验证集下最小的损失的模型
    loss_record = {'train': [], 'dev':[]}
    early_stop_cnt = 0 #设置连续多少个epoch，没更新最优模型就提前停止训练
    epoch = 0
    while epoch < n_epochs:
        model.train()
        for data, label in tr_set:
            optimizer.zero_grad()
            data, label = data.to(device), label.to(device)
            pred = model(data)
            mse_loss = model.cal_loss(pred, label)
            mse_loss.backward()
            optimizer.step()
            loss_record['train'].append(mse_loss.detach().cpu().item())

        dev_mse = val(dev_set, model, device)
        if dev_mse < min_mse:
            min_mse = dev_mse
            print("Saving model (epoch = {:4d}, loss={:.4f})".format(epoch+1, min_mse))
            torch.save(model.state_dict(), config['save_path'])
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1

        epoch += 1
        loss_record['dev'].append(dev_mse)
        if early_stop_cnt > config['early_stop']:
            break
        scheduler.step()

    print('Finish training after {} epochs'.format(epoch))
    return min_mse, loss_record

def test(tt_set, model, device):
    model.eval()
    preds = []
    for data in tt_set:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data)
            preds.append(pred.detach().cpu())
    preds = torch.cat(preds, dim=0, ).numpy()
    return preds

def save_pred(pres, file):
    print("Saving results to {}".format(file))
    with open(file, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'tested_positive'])
        for i, p in enumerate(preds):
            writer.writerow([i, p])



if __name__ == '__main__':
    '''
    myseed = 0
    np.random.seed(myseed)
    torch.manual_seed(myseed)
    torch.cuda.manual_seed_all(myseed)
    
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        print(torch.cuda.device_count())
        print(torch.cuda.get_device_name(0))
        print(torch.cuda.current_device())
    '''
    device = get_device()
    os.makedirs('models', exist_ok=True)
    target_only = False
    config = {
        'n_epochs': 6000,
        'batch_size': 320,
        'optimizer': 'SGD',
        'optim_hparas': {
            'lr': 0.0001,
            'momentum': 0.9,
            'weight_decay': 0.01
        },
        'early_stop': 1000,
        'save_path': 'models/model.pth'
    }

    tr_path = 'C:\\Users\\A539\\Desktop\\covid.train.csv'
    tt_path = 'C:\\Users\\A539\\Desktop\\covid.test.csv'

    tr_set = pre_dataloader(tr_path, 'train', config['batch_size'], target_only=True)
    dv_set = pre_dataloader(tr_path, 'dev', config['batch_size'], target_only=True)
    tt_set = pre_dataloader(tt_path, 'test', config['batch_size'], target_only=True)

    model = NeuralNet(tr_set.dataset.dim).to(device)

    model_loss, model_loss_record = train(tr_set, dv_set, model, config, device)

    plot_learning_curve(model_loss_record, title='deep model')

    del model
    model = NeuralNet(tr_set.dataset.dim).to(device)
    ckpt = torch.load(config['save_path'], map_location='cpu')
    model.load_state_dict(ckpt)
    plot_pred(dv_set, model, device)

    preds = test(tt_set, model, device)
    save_pred(preds, 'pred.csv')























