import sys
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.optim as optim
import numpy as np
import os
import random
sys.path.append(os.getcwd())
from segmentation.data import IronDataset
import torchvision.transforms as tr
from torch.utils.data import DataLoader
from segmentation.modeltest import DP_UNet
from segmentation.weight_map_loss1 import WeightMapLoss
import matplotlib.pyplot as plt
from IPython.display import clear_output
import time
from skimage import morphology
import argparse

cwd = os.getcwd()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 设置随机种子
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

setup_seed(2024)

def plot_imgs(title, imgs):
    plt.figure(figsize=(30, 30))
    plt.subplot(231)
    plt.imshow(imgs[0], cmap='gray')
    plt.axis('off')
    plt.title(title + '_img-label-out-skle')
    plt.subplot(232)
    plt.imshow(imgs[1], cmap='gray')
    plt.axis('off')
    plt.subplot(233)
    plt.imshow(imgs[2], cmap='gray')
    plt.axis('off')
    plt.subplot(234)
    plt.imshow(imgs[3])
    plt.axis('off')
    plt.subplot(235)
    plt.imshow(imgs[4], cmap='gray')
    plt.axis('off')
    plt.subplot(236)
    plt.imshow(imgs[5], cmap='gray')
    plt.axis('off')
    plt.savefig(checkpoint_path + str(title) + '.png')

def plot(epoch, loss_list, test_loss_list):
    clear_output(True)
    plt.figure()
    plt.title('epoch %s. train loss: %s. val_loss: %s' % (epoch, loss_list[-1], test_loss_list[-1]))
    plt.plot(loss_list, color="r", label="train loss")
    plt.plot(test_loss_list, color="b", label="val loss")
    plt.legend(loc="best")
    plt.savefig(checkpoint_path + 'test_model_loss_state.png')


def check(dataset):  # 检查数据集输入
    print(dataset.__len__())
    image, mask, last, weight = dataset.__getitem__(2)
    print(image.size(), " ", mask.size(), ' ', np.unique(mask), " ", last.size(), " ", np.unique(last), " ", weight.size(), " ", np.unique(weight))

    test_image = image.squeeze().numpy()
    test_mask = mask.squeeze().numpy()
    last_mask = last.squeeze().numpy()
    test_weight = weight.squeeze().numpy()

    print(np.max(test_image), " ", np.min(test_image))

    print(" last mask ", np.unique(last_mask))
    print(" weight ", np.unique(weight[0, :, :]))

    plt.figure(figsize=(30, 30))
    plt.subplot(2, 2, 1), plt.imshow(test_image, cmap="gray")
    plt.subplot(2, 2, 2), plt.imshow(test_mask, cmap="gray")
    plt.subplot(2, 2, 3), plt.imshow(last_mask, cmap="gray")
    plt.subplot(2, 2, 4), plt.imshow(test_weight[1, :, :], vmin=0, vmax=30, cmap="plasma")
    plt.show()


def adjust_learning_rate(optimizer, epoch, learning_rate):
    """设置学习率"""
    lr = learning_rate * (0.8 ** (epoch // 10))
    if not lr < 3e-5:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

# 初始化参数
def weights_init(m):
    classname = m.__class__.__name__  # 得到网络层的名字
    if classname.find('Conv2d') != -1:
        nn.init.xavier_uniform_(m.weight.data)
class Trainer(object):

    def __init__(self, bs=8, loss_type='add', data_dir = ''):
        # parameteres
        self.batch_size = bs
        self.learning_rate =0.0001
        self.loss_type = loss_type
        self.data_dir = data_dir

        self.transform = tr.Compose([
            tr.ToTensor(),
            tr.Normalize(mean = [0.9336267],
                        std = [0.1365774])
        ])

        print(' batch_size is ', self.batch_size, ' learning_rate is ', self.learning_rate, ' loss_type is ', self.loss_type,  ' data_dir is ', self.data_dir)

        # 训练集
        self.train_dataset = IronDataset(self.data_dir, train=True, transform=self.transform, crop=True, crop_size=(256, 256), dilate=5)
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=1)
        # 测试
        self.test_dataset = IronDataset(self.data_dir, train=False, transform=self.transform, crop=False, dilate=5)
        self.test_loader = DataLoader(self.test_dataset, batch_size=1, shuffle=False, num_workers=1)
        print("Training Data : ", len(self.train_loader.dataset))
        print("Test Data :", len(self.test_loader.dataset))
        # set model
        self.model = DP_UNet(num_channels=2, num_classes=2,multi_layer=True)
        self.model.apply(weights_init)

        if torch.cuda.is_available():
            model = nn.DataParallel(self.model).cuda()
        self.criterion=WeightMapLoss(_dilate_cof=5)
        self.method = 5
        self.optimizer = Adam(params=self.model.parameters(), lr=self.learning_rate, weight_decay=0.)
        #self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.learning_rate, alpha=0.9, eps=1e-5)

    def train(self, epoch, loss_list, plotimgs):
        self.model.train()
        adjust_learning_rate(self.optimizer, epoch, learning_rate=self.learning_rate)
        iterations = max(1, int(len(self.train_loader.dataset) / self.batch_size))
        for batch_idx, (image, mask, last, weight) in enumerate(self.train_loader):
            images = []
            label = mask.squeeze(1).long()

            if torch.cuda.is_available():
                img, label, last, weight = image.cuda(), label.cuda(), last.cuda(), weight.cuda()

            self.optimizer.zero_grad()
            # 正则化
            last[last == 255] = -6
            last[last == 0] = 1

            output = self.model(img, last)
            loss = self.criterion(output, label, weight, method=self.method)
            loss.backward()
            # 梯度裁剪
            nn.utils.clip_grad_norm_(self.model.parameters(), 5)
            self.optimizer.step()

            if (batch_idx + 1) % iterations == 0:
                loss_list.append(float(loss.item()))
                # 过程输出
                if plotimgs:
                    images.append(image[0].squeeze().numpy())
                    images.append(mask[0].squeeze().numpy())
                    images.append(last[0].squeeze().cpu().numpy())
                    images.append(weight[0].squeeze().cpu().numpy()[1, :, :])
                    out = torch.sigmoid(output[0])
                    result_npy = out.max(0)[1].data.squeeze().cpu().numpy()
                    result_npy = np.array(result_npy).astype('uint8') * 255
                    images.append(result_npy)
                    result_npy_nc = morphology.skeletonize(result_npy / 255) * 255  # 骨架化
                    images.append(result_npy_nc)
                    plot_imgs(str(epoch).zfill(3) + '_train', images)

    def test(self, test_loss_list):
        self.model.eval()
        test_loss = 0
        images = []
        count = 0
        for batch_idx, (image, mask, last, weight) in enumerate(self.test_loader):
            count += 1
            if torch.cuda.is_available():
                label = mask.squeeze(1).long()
                img, label, last, weight = image.cuda(), label.cuda(), last.cuda(), weight.cuda()
            last[last == 255] = -6
            last[last == 0] = 1

            output = self.model(img, last)
            test_loss += float(self.criterion(output, label, weight, method=self.method).item())


            if batch_idx == 1:
                images.append(image.squeeze().numpy())
                images.append(mask.squeeze().numpy())
                images.append(last.squeeze().cpu().numpy())
                images.append(weight.squeeze().cpu().numpy()[1, :, :])
                out = torch.sigmoid(output)
                result_npy = out.max(1)[1].data.squeeze().cpu().numpy()
                result_npy = np.array(result_npy).astype('uint8') * 255
                images.append(result_npy)
                result_npy_nc = morphology.skeletonize(result_npy / 255) * 255  # 骨架化
                images.append(result_npy_nc)

        # 平均损失
        test_loss /= count
        test_loss_list.append(test_loss)
        return test_loss, images


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BP_UNet training')
    parser.add_argument('--input', type=str, default=os.path.join(cwd, "/", "../datasets", "segmentation", "net_train"), help="dataset dir")
    parser.add_argument("--bs", type=int, default=10, help="batch size")
    parser.add_argument("--loss", type=str, default="add")
    parser.add_argument('--epochs', type=int, default=10, help="epochs")
    parser.add_argument('--ml', help='apply multi_layer', action='store_true')
    args = parser.parse_args()

    loss_list = []
    test_loss_list = []
    test_baseline = 10000
    st_time = time.time()
    # model save address
    checkpoint_path = '/home/user/dongruixuan/BP_UNet/segmentation/parameter/DP_Unet/'
    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)
    trainer = Trainer(bs=args.bs, loss_type=args.loss, data_dir=args.input)
    for i in range(1, args.epochs + 1):
        if (i - 1) % 10 == 0:
            plotimgs = True
        else:
            plotimgs = False
        trainer.train(i, loss_list, plotimgs)
        test_loss, imgs = trainer.test(test_loss_list)

        f = open(checkpoint_path + 'loss.txt', 'a')
        f.write("epoch %d:loss %f \n" % (i, loss_list[-1]))
        f.write("epoch %d:loss %f test_loss %f\n" % (i, loss_list[-1], test_loss_list[-1]))
        f.close()
        plot(i, loss_list, test_loss_list)
        if (i - 1) % 10 == 0:
            plot_imgs(str(i).zfill(3) + '_test', imgs)
            torch.save(trainer.model.state_dict(), checkpoint_path + str(i) + '_epoch' + '.pth')
        if test_loss < test_baseline:
            test_baseline = test_loss
            torch.save(trainer.model.state_dict(), checkpoint_path + "best_model_state.pth")

    ed_time = time.time()
    print('end training , it costs ', str(ed_time-st_time), ' sec in total...')




