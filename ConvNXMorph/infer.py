import glob
import os, losses, utils
import time
import datetime
from torch.utils.data import DataLoader
from data import datasets, trans
import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from natsort import natsorted
from models import VxmDense_1, VxmDense_2, VxmDense_huge
from convnxmorph import ConvNXMorph
import torch.nn as nn
import torch.utils.data as Data
import SimpleITK as sitk

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.vals = []
        self.std = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.vals.append(val)
        self.std = np.std(self.vals)

def MSE_torch(x, y):
    return torch.mean((x - y) ** 2)

def MAE_torch(x, y):
    return torch.mean(torch.abs(x - y))

def main():
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    fixed_path = './ISLES_data/ADC'
    moving_path = './ISLES_data_F_inverse_bbox_resize_2niigz/FLAIR'
    img_size = (144, 224, 224)
    batch_size = 1
    model_path = '/home/ConvNextMorph/experiments/vxm_1_ncc_1_diffusion_1.5_lr_0.0002/min_eval_loss-0.191.pth'
    list_filename = list(x for x in os.listdir('./ISLES_data/ADC'))


    model = ConvNXMorph(img_size,batch_size)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)



    test_composed = transforms.Compose([trans.NumpyType((np.float32, np.int16))])
    test_set = datasets.TestDataset(list_filename, fixed_path, moving_path, transforms=test_composed)
    test_dataloader = Data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0,
                                      drop_last=True)

    result_path = './Result'
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    result_FLAIR_def_path = './Result/FLAIR_def'
    if not os.path.exists(result_FLAIR_def_path):
        os.makedirs(result_FLAIR_def_path)

    result_flow_path = './Result/flow'
    if not os.path.exists(result_flow_path):
        os.makedirs(result_flow_path)

    eval_dsc_def = AverageMeter()
    eval_dsc_raw = AverageMeter()
    eval_det = AverageMeter()
    start_time = time.time()
    with torch.no_grad():
        stdy_idx = 0

        for i, (moving_image, fixed_image, filename) in enumerate(test_dataloader):
            model.eval()

            moving_image = moving_image.to(device)
            fixed_image = fixed_image.to(device)
            filename = filename[0]

            input = torch.cat((moving_image, fixed_image),dim=1)
            moving_image_def, flow = model(input)

            moving_image_path = os.path.join('./ISLES_data/FLAIR', filename)
            raw_data = sitk.ReadImage(moving_image_path)
            origin = raw_data.GetOrigin()
            direction = raw_data.GetDirection()
            space = raw_data.GetSpacing()



            moving_image_def_np = moving_image_def.detach().cpu().squeeze().numpy()
            flow_np = flow.permute(0, 2, 3, 4, 1)[np.newaxis, ...].detach().cpu().squeeze().numpy()

            moving_image_def_nii = sitk.GetImageFromArray(moving_image_def_np)
            moving_image_def_nii.SetOrigin(origin)
            moving_image_def_nii.SetDirection(direction)
            moving_image_def_nii.SetSpacing(space)

            flow_nii = sitk.GetImageFromArray(flow_np)
            flow_nii.SetOrigin(origin)
            flow_nii.SetDirection(direction)
            flow_nii.SetSpacing(space)


            FLAIR_def_path = os.path.join(result_path, 'FLAIR_def', filename)
            flow_path = os.path.join(result_path, 'flow',filename)

            sitk.WriteImage(moving_image_def_nii, FLAIR_def_path)
            sitk.WriteImage(flow_nii, flow_path)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('infering time{}'.format(total_time_str))
        #     tar = y.detach().cpu().numpy()[0, 0, :, :, :]
        #     jac_det = utils.jacobian_determinant_vxm(flow.detach().cpu().numpy()[0, :, :, :, :])
        #     line = utils.dice_val_substruct(def_out.long(), y_seg.long(), stdy_idx)
        #     line = line + ',' + str(np.sum(jac_det <= 0)/np.prod(tar.shape))
        #     csv_writter(line, 'Quantitative_Results/' + csv_name)
        #     eval_det.update(np.sum(jac_det <= 0) / np.prod(tar.shape), x.size(0))
        #
        #     dsc_trans = utils.dice_val(def_out.long(), y_seg.long(), 46)
        #     dsc_raw = utils.dice_val(x_seg.long(), y_seg.long(), 46)
        #     print('Trans dsc: {:.4f}, Raw dsc: {:.4f}'.format(dsc_trans.item(),dsc_raw.item()))
        #     eval_dsc_def.update(dsc_trans.item(), x.size(0))
        #     eval_dsc_raw.update(dsc_raw.item(), x.size(0))
        #     stdy_idx += 1
        # print('Deformed DSC: {:.3f} +- {:.3f}, Affine DSC: {:.3f} +- {:.3f}'.format(eval_dsc_def.avg,
        #                                                                             eval_dsc_def.std,
        #                                                                             eval_dsc_raw.avg,
        #                                                                             eval_dsc_raw.std))
        # print('deformed det: {}, std: {}'.format(eval_det.avg, eval_det.std))

def csv_writter(line, name):
    with open(name+'.csv', 'a') as file:
        file.write(line)
        file.write('\n')

if __name__ == '__main__':
    main()
