import matplotlib.pyplot as plt

# import torch.nn as nn
# import torch.nn.functional as F
# import torch.autograd as autograd
import torch
import numpy as np
import os
from torchvision import  transforms as T
from skimage.color import ycbcr2rgb
from skimage.io import imsave
from skimage import img_as_ubyte
from TMQI import TMQI, TMQIr

def write_figures(location, train_losses, val_losses,total_Q,total_S,total_N):
    plt.plot(train_losses, label='training loss')
    plt.plot(val_losses, label='validation loss')
    plt.plot(total_Q, label='Q')
    plt.plot(total_S, label='S')
    plt.plot(total_N, label='N')
    plt.legend()
    plt.savefig(location + '/loss.png')
    plt.close('all')


def write_log(location, epoch, train_loss, val_loss):
    if epoch == 0:
        f = open(location + '/log.txt', 'w+')
        f.write('epoch\t\ttrain_loss\t\tval_loss\n')
    else:
        f = open(location + '/log.txt', 'a+')

    f.write(str(epoch) + '\t' + str(train_loss) + '\t' + str(val_loss) + '\n')

    f.close()

def savefig(epoch,  outputs, cbcr, filename):
    test_mean, test_std = torch.tensor([0.5 ,0.5 ,0.5]), torch.tensor([0.5 ,0.5, 0.5])
    test_mean_Y, test_std_Y = torch.tensor([0.5 ]), torch.tensor([0.5 ])
    if epoch %1 == 0:
        # np.clip(outputs,-1,1)
        # outputs.data.clamp_(-1,1)

        print(outputs.shape)
        print(cbcr.shape)
        y = outputs
        outputs = torch.cat([outputs,cbcr],dim=1)
        print(y.shape)
        Y = y * test_std_Y.view(1,1,1) + test_mean_Y.view(1,1,1)
        print(Y.shape)
        prediction = outputs * test_std.view(3,1,1) + test_mean.view(3,1,1)

        # Y = Y * 255
        prediction = prediction * 255
        # prediction = outputs#[:,0:1,:,:]
        # print(prediction.shape)
        # prediction.data.clamp_(0,1)
        #ycbcr to rgb
        prediction = ycbcr2rgb(prediction[0].detach().numpy().transpose(1,2,0))
        Y = Y[0].detach().numpy().transpose(1,2,0)
        print(Y.shape)
        Y = np.clip(Y, -1, 1)
        prediction = np.clip(prediction, -1, 1)
        filename = filename[0].replace(' ','').split('\\')[-2]
        print(filename)
        imsave(".\\result\\%s_fused.png" % (filename), img_as_ubyte(prediction))
        # imsave(".\\result_Y\\%s_fused.png" % (filename), img_as_ubyte(Y))
        # image_list = T.ToPILImage()(prediction).convert('RGB')
        # # image_list = T.ToPILImage()(prediction[0].cpu()).convert('RGB')
        # filepath = '.\\Result'
        # if not os.path.exists(filepath):
        #     os.mkdir(filepath,7777)
        # print('=============>save image to  ',filepath + r'\{}.png'.format(filename[0].replace('.png','').split('\\')[-1].split('\\')[0]))
        # image_list.save(filepath + '\\{}.png'.format(filename[0].split('\\')[-2]),'png')

def predict(model, low, high):
    model.eval()
    model.load_state_dict(torch.load('output/weight.pth', map_location=device))
    skips = wave_model(low, high)
    outputs, d2, d3, d4, d5, d6, db = model(inputs, skips)

def cal_TMQI(hdr,ldr):
    Q, S, N, s_local, s_maps = TMQIr()(hdr, ldr)
    return Q, S, N, s_local, s_maps

def cal_running_TMQI(hdr1,hdr2,ldr1,ldr2):
    test_mean, test_std = torch.tensor([0.5 ,0.5 ,0.5]), torch.tensor([0.5 ,0.5, 0.5])
    hdr = torch.cat([hdr1,hdr2],dim=1)
    hdr = hdr * test_std.view(3,1,1) + test_mean.view(3,1,1)
    hdr = hdr * 255
    #ycbcr to rgb
    hdr = ycbcr2rgb(hdr[0].detach().numpy().transpose(1,2,0))
    hdr = np.clip(hdr, -1, 1)

    ldr = torch.cat([ldr1,ldr2],dim=1)
    ldr = ldr * test_std.view(3,1,1) + test_mean.view(3,1,1)
    ldr = ldr * 255
    #ycbcr to rgb
    ldr = ycbcr2rgb(ldr[0].detach().numpy().transpose(1,2,0))
    ldr = np.clip(ldr, -1, 1)

    Q, S, N, s_local, s_maps = TMQIr()(hdr, ldr)
    return Q, S, N, s_local, s_maps


def savewavefig(epoch, outputs, cbcr, filename):
    test_mean, test_std = torch.tensor([0.5 ,0.5 ,0.5]), torch.tensor([0.5 ,0.5, 0.5])
    if epoch %1 == 0:
        # np.clip(outputs,-1,1)
        # outputs.data.clamp_(-1,1)
        # outputs = torch.cat([outputs,cbcr],dim=1)

        # prediction = outputs * test_std.view(3,1,1) + test_mean.view(3,1,1)
        # prediction = prediction * 255
        # prediction = outputs#[:,0:1,:,:]
        # print(prediction.shape)
        # prediction.data.clamp_(0,1)


        #ycbcr to rgb

        # prediction = ycbcr2gray(prediction[0].numpy().transpose(1,2,0))
        outputs = np.transpose(outputs[0],(1,2,0))
        outputs = np.clip(outputs, -1, 1)
        print(outputs.shape)
        # print(prediction)
        imsave(".\\Result_wavelet2\\%06s_fused.png" % (filename), img_as_ubyte(outputs))



def rescue_img_mask_name():
    import cv2
    img_root = r'.\result2\\'
    img_path = os.listdir(img_root)

    for i in img_path:
        path = img_root +i
        # print(path)
        img = cv2.imread(path)
        i = i.replace(' ','')
        print(i)
        # print(img)
        path = img_root +i
        print(path)
        cv2.imwrite(path,img)

if __name__ == '__main__':
    rescue_img_mask_name()