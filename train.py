from torch.autograd import Variable
import torch.autograd as autograd
import torch
import torch.optim as optim
from helper import write_log, write_figures, savefig, savewavefig, cal_running_TMQI
# from hdr_loss import HDport numpy as np
from dataset import get_loader
import torch.nn as nn
from ICCV_model_sk_ablation_no import init_net, model
from tqdm import tqdm
from torchvision import transforms as T
# from msssim import MSSSIM
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
# from attention_unpool import Model
from loss import MEF_SSIM_Loss

import numpy as np
class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]
def sgn(x):
    return torch.abs(x) / (x+1e-12)
def mu_law(x, device):
    mu = torch.ones(x.shape,device=device)*5000
    return sgn(x) * (torch.log(torch.ones(x.shape,device=device) + mu*torch.abs(x))) / (torch.log(torch.ones(x.shape,device=device) + mu))


#-----多GPU训练的模型读取的代码，multi-gpu training---------
def load_network(network):
    save_path = 'output/weight_sum_blur.pth'
    state_dict = torch.load(save_path)
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        namekey = k[7:] # remove `module.`
        new_state_dict[namekey] = v
    # load params
    network.load_state_dict(new_state_dict)
    return network

def fit(epoch, model, optimizer, criterion, msssim,  device, data_loader, phase='training'):
    if phase == 'training':
        model.train()
    if phase == 'training' and epoch==0:
        print(123)
        # model.load_state_dict(torch.load(
        #         'output/weight_best.pth', map_location=device))
    else:
        model.eval()
    show = 0
    running_loss = 0
    total_loss_d = 0
    count = 0
    running_Q = 0
    running_S = 0
    running_N = 0

    for low, high, groundtruth, filename in tqdm(data_loader):
        # with torch.no_grad():
        # inputs = out[:,0:1,:,:].to(device)
        low = low.to(device)
        high = high.to(device)

        targets_y = groundtruth[:,0:1,:,:].to(device)
        low_y = low[:,0:1,:,:].to(device)
        high_y = high[:,0:1,:,:].to(device)

        targets_cbcr = groundtruth[:,1:3,:,:].to(device)
        targets_cb = groundtruth[:,1:2,:,:].to(device)
        targets_cr = groundtruth[:,2:3,:,:].to(device)

        low_cbcr = low[:,1:3,:,:].to(device)
        high_cbcr = high[:,1:3,:,:].to(device)

        low_cb = low[:,1:2,:,:].to(device)
        high_cb = high[:,1:2,:,:].to(device)

        low_cr = low[:,2:3,:,:].to(device)
        high_cr = high[:,2:3,:,:].to(device)

        if phase == 'predict':
            model.eval()
            model.load_state_dict(torch.load(
                'output/weight_best.pth', map_location=device))
            # load_network(model)

        if phase == 'training':
            optimizer.zero_grad()

        else:
            model.eval()

        # with torch.no_grad():

        outputs_y, outputs_cbcr = model(low, high)

        outputs_cb = outputs_cbcr[:,0:1,:,:]
        outputs_cr = outputs_cbcr[:,1:2,:,:]


        # X: (N,3,H,W) a batch of normalized images (-1 ~ 1)
        # Y: (N,3,H,W)
        low_y_msssim = (low_y + 1) / 2  # [-1, 1] => [0, 1]
        high_y_msssim = (high_y + 1) / 2  # [-1, 1] => [0, 1]
        outputs_y_msssim = (outputs_y + 1) / 2  # [-1, 1] => [0, 1]

        low_cbcr_msssim = (low_cbcr + 1) / 2  # [-1, 1] => [0, 1]
        high_cbcr_msssim = (high_cbcr + 1) / 2  # [-1, 1] => [0, 1]
        outputs_cb_msssim = (outputs_cb + 1) / 2  # [-1, 1] => [0, 1]
        outputs_cr_msssim = (outputs_cr + 1) / 2  # [-1, 1] => [0, 1]
        targets_y_msssim = (targets_y + 1) / 2
        targets_cb_msssim = (targets_cb + 1) / 2
        targets_cr_msssim = (targets_cr + 1) / 2
        # lossm_y,_ = msssim(low_y_msssim,high_y_msssim,outputs_y_msssim)
        # lossm_cbcr,_ = msssim(low_cbcr_msssim,high_cbcr_msssim,outputs_cbcr_msssim)


        lossL1_y = criterion((outputs_y), (targets_y))
        lossL1_cb = criterion((outputs_cb),( targets_cb))
        lossL1_cr = criterion((outputs_cr),( targets_cr))

        lossL1 = lossL1_y + (lossL1_cb + lossL1_cr) * 0.5
        # lossm = lossm_y #+ lossm_cbcr

        loss = lossL1
        # print(' lossL1: %.4f  lossM: %.4f ' %(lossL1.item(), lossm.item()))
        #+ lossMSSIM
        running_loss += lossL1.item()# + lossm.item()

        print(' lossL1: %.4f  ' %(lossL1.item()))
        # savefig(epoch, out[:,0:1,:,:].cpu(), out[:,1:3,:,:].cpu(), filename)
        if phase == 'training':
            show = 0
            loss.backward()
            optimizer.step()

        elif phase == 'validation' and show <= 20:
            savefig(epoch, outputs_y.cpu().detach(), outputs_cbcr.cpu().detach(), filename)
            show += 1
        elif phase == 'validation' and show > 20:
            break
        elif phase == 'predict':
            savefig(epoch, outputs_y.cpu(), outputs_cbcr.cpu(), filename)

    epoch_loss = running_loss / len(data_loader.dataset)
    torch.cuda.empty_cache()
    return epoch_loss


def train(root, device, model, epochs, bs, lr):
    # print('start training ...........')
    train_loader, val_loader = get_loader(
        root=root, batch_size=bs, shuffle=True)

    criterion = nn.MSELoss().to(device)
    # criterionMSE = nn.MSELoss().to(device)
    msssim = MEF_SSIM_Loss().to(device)

    train_losses, val_losses,total_Q,total_S,total_N= [
    ], [], [], [], []


    for epoch in range(epochs):
        # if epoch % 20 == 0 and epoch !=0:
        #     lr /= 10
        optimizer = optim.Adam(model.parameters(), lr=lr)
        train_epoch_loss = fit(epoch, model,optimizer, criterion,   msssim, device, train_loader, phase='training')
        val_epoch_loss = fit(epoch, model,optimizer, criterion,  msssim, device, val_loader, phase='validation')
        # val_epoch_loss = fit(epoch, model,optimizer, criterion,  msssim, device, val_loader, phase='predict')

        print('-----------------------------------------')

        if epoch %10 == 0 and epoch != 0 :
            torch.save(model.state_dict(), 'output/weight_{}.pth'.format(epoch))

        if (epoch == 0 or val_epoch_loss <= np.min(val_losses) ) :
            torch.save(model.state_dict(), 'output/weight_best.pth')

        train_losses.append(train_epoch_loss)
        val_losses.append(val_epoch_loss)

        write_figures('output', train_losses, val_losses,total_Q,total_S,total_N)
        write_log('output', epoch, train_epoch_loss, val_epoch_loss)


if __name__ == "__main__":
    # device = torch.device("cpu")
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    model = model().to(device)
    init_net(model)
    batch_size = 6
    num_epochs = 200
    learning_rate = 0.0001
    root = 'data/train'
    train(root, device, model,
          num_epochs, batch_size, learning_rate)
