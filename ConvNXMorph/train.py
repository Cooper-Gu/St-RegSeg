import glob
from torch.utils.tensorboard import SummaryWriter
import os, losses, utils
import sys
from torch.utils.data import DataLoader
from data import datasets, trans
import numpy as np
import torch
from torchvision import transforms
from torch import optim
import torch.nn as nn
import matplotlib.pyplot as plt
from natsort import natsorted
from convnxmorph import ConvNXMorph
from sklearn.model_selection import train_test_split
import torch.utils.data as Data

class Logger(object):
    def __init__(self, save_dir):
        self.terminal = sys.stdout
        self.log = open(save_dir+"logfile.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def main():
    lr = 0.0002
    epoch_start = 0
    max_epoch = 200
    img_size = (144, 224, 224)
    cont_training = False
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    min_eval_loss = 1000
    batch_size = 1


    list_filename = list(x for x in os.listdir('/home/ISLES_preprocess/ISLES_data/train/ADC/'))

    fixed_1_path = '/home/ISLES_preprocess/ISLES_data/train/ADC'
    fixed_2_path = '/home/ISLES_preprocess/ISLES_data/train/DWI'
    moving_path = '/home/ISLES_preprocess/ISLES_data/train/FLAIR'

    weights = [0.5, 0.5, 1.5]  # loss weights
    save_dir = 'vxm_three_ncc_{}_{}_diffusion_{}_lr_{}/'.format(weights[0], weights[1], weights[2], lr)
    if not os.path.exists('experiments/' + save_dir):
        os.makedirs('experiments/' + save_dir)
    if not os.path.exists('logs/' + save_dir):
        os.makedirs('logs/' + save_dir)
    sys.stdout = Logger('logs/' + save_dir)

    '''
    Initialize model
    '''
    model = ConvNXMorph(img_size, batch_size)
    model = model.to(device)



    '''
    If continue from previous training
    '''
    if cont_training:
        epoch_start = 0
        model_dir = 'experiments/'+save_dir
        updated_lr = round(lr * np.power(1 - (epoch_start) / max_epoch,0.9),8)
        best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[-3])['state_dict']
        model.load_state_dict(best_model)
    else:
        updated_lr = lr

    '''
    Initialize training
    '''
    train_composed = transforms.Compose([trans.RandomFlip(0),
                                         trans.NumpyType((np.float32, np.float32)),
                                         ])


    train_set = datasets.Dataset(list_filename, fixed_1_path, fixed_2_path, moving_path,transforms = train_composed)
    train_dataloader = Data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)


    # optimizer = optim.Adam(model.parameters(), lr=updated_lr, weight_decay=0, amsgrad=True)
    optimizer = optim.AdamW(model.parameters(), lr=updated_lr, weight_decay=5e-5, amsgrad=True)
    criterion = losses.NCC_vxm()
    # criterion = losses.MIND_loss()
    criterions = [criterion]
    criterions += [criterion]
    criterions += [losses.Grad3d(penalty='l2')]

    writer = SummaryWriter(log_dir='logs/'+save_dir)
    for epoch in range(epoch_start, max_epoch):
        print('Training Starts')
        '''
        Training
        '''
        loss_all = utils.AverageMeter()
        idx = 0
        for i, (moving_image, fixed_1_image, fixed_2_image) in enumerate(train_dataloader):
            idx += 1
            model.train()
            adjust_learning_rate(optimizer, epoch, max_epoch, lr)
            moving_image = moving_image.to(device)
            fixed_1_image = fixed_1_image.to(device)
            fixed_2_image = fixed_2_image.to(device)

            input = torch.cat((moving_image, fixed_1_image, fixed_2_image), dim=1).to(device)
            output = model(input)    # output include two parts: moved_image, pos_flow
            output = list(output)
            moved_image = output[0]
            output.insert(1, moved_image)
            fixed_image_list = [fixed_1_image, fixed_2_image, fixed_1_image]
            loss = 0
            loss_vals = []
            for n, loss_function in enumerate(criterions):
                curr_loss = loss_function(output[n], fixed_image_list[n]) * weights[n]
                loss_vals.append(curr_loss)
                loss += curr_loss
            loss_all.update(loss.item(), fixed_1_image.numel())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('Iter {} of {} loss {:.4f}, Img Sim: {:.6f}, Reg: {:.6f}'.format(idx, len(train_dataloader), loss.item(), loss_vals[0].item(), loss_vals[1].item()))

        writer.add_scalar('Loss/train', loss_all.avg, epoch)
        print('Epoch {} loss {:.4f}'.format(epoch, loss_all.avg))

        if loss_all.avg < min_eval_loss:
            min_eval_loss = loss_all.avg
            model_path = os.path.join('./experiments', save_dir, 'min_eval_loss{:.3f}.pth'.format(min_eval_loss))
            torch.save(model.state_dict(), model_path)

        loss_all.reset()
    writer.close()

def comput_fig(img):
    img = img.detach().cpu().numpy()[0, 0, 48:64, :, :]
    fig = plt.figure(figsize=(12, 12), dpi=180)
    for i in range(img.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.axis('off')
        plt.imshow(img[i, :, :], cmap='gray')
    fig.subplots_adjust(wspace=0, hspace=0)
    return fig

def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(INIT_LR * np.power(1 - (epoch) / MAX_EPOCHES, power), 8)

def mk_grid_img(grid_step, line_thickness=1, grid_sz=(160, 192, 224)):
    grid_img = np.zeros(grid_sz)
    for j in range(0, grid_img.shape[1], grid_step):
        grid_img[:, j+line_thickness-1, :] = 1
    for i in range(0, grid_img.shape[2], grid_step):
        grid_img[:, :, i+line_thickness-1] = 1
    grid_img = grid_img[None, None, ...]
    grid_img = torch.from_numpy(grid_img).cuda()
    return grid_img

def save_checkpoint(state, save_dir='models', filename='checkpoint.pth.tar', max_model_num=8):
    torch.save(state, save_dir+filename)
    model_lists = natsorted(glob.glob(save_dir + '*'))
    while len(model_lists) > max_model_num:
        os.remove(model_lists[0])
        model_lists = natsorted(glob.glob(save_dir + '*'))

if __name__ == '__main__':
    main()