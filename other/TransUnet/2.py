import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import cv2
from PIL import Image
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
# 导入evaluation.py中的评价函数
from evaluation import (
    calculate_dice,
    calculate_miou,
    get_figure_of_merit,
    get_map_2018kdasb_new
)
from skimage.measure import label
import gala.evaluate as ev
from sklearn.metrics import confusion_matrix  # calculate_miou依赖


class Custom2DDataset(Dataset):
    def __init__(self, base_dir, split="test", list_dir=None, transform=None):
        self.base_dir = base_dir
        self.split = split
        self.transform = transform
        self.image_dir = os.path.join(base_dir, "datasets", "segmentation", "net_test", "test", "images")
        self.label_dir = os.path.join(base_dir, "datasets", "segmentation", "net_test", "test", "labels")

        # 加载数据集列表
        if list_dir is not None:
            list_path = os.path.join(list_dir, f"{split}.txt")
            with open(list_path, 'r') as f:
                self.case_list = [line.strip() for line in f.readlines()]
        else:
            self.case_list = [
                f.split('.')[0] for f in os.listdir(self.image_dir)
                if f.endswith(('.png', '.jpg', '.jpeg'))
            ]

    def __len__(self):
        return len(self.case_list)

    def __getitem__(self, idx):
        case_name = self.case_list[idx]

        image_path = os.path.join(self.image_dir, f"{case_name}.png")
        label_path = os.path.join(self.label_dir, f"{case_name}.png")

        # 读取图片（H,W,C）
        image = cv2.imread(image_path)  # BGR格式
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转RGB
        label = cv2.imread(label_path, 0)  # 标注为单通道（类别ID）
        label = (label == 255).astype(np.uint8)
        # 基础预处理：归一化+转张量
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0  # (C,H,W)
        label = torch.from_numpy(label).long()  # (H,W)

        return {
            "image": image,
            "label": label,
            "case_name": case_name
        }


def test_single_2d_image(image, label, model, classes, patch_size, test_save_path, case):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image = image.to(device)

    # 推理
    with torch.no_grad():
        output = model(image)
        pred = torch.argmax(output, dim=1)
        pred = pred.squeeze(0).cpu().numpy()
        label = label.squeeze(0).cpu().numpy()

    # 转换为evaluation.py要求的255/0二值格式（类别1为前景255，背景0）
    pred_255 = (pred == 1).astype(np.uint8) * 255
    label_255 = (label == 1).astype(np.uint8) * 255

    # 计算evaluation.py中的核心指标
    dice = calculate_dice(pred_255, label_255)
    miou = calculate_miou(pred_255, label_255)
    f_score = get_figure_of_merit(pred_255, label_255)
    map_score = get_map_2018kdasb_new(pred_255, label_255)

    # 计算RI/VI指标（连通域分析）
    pred_labeled, num_pred = label(pred_255, background=255, return_num=True, connectivity=1)
    label_labeled, num_label = label(label_255, background=255, return_num=True, connectivity=1)
    merger_error, split_error = ev.split_vi(pred_labeled, label_labeled)
    vi = merger_error + split_error
    ri = ev.rand_index(pred_labeled, label_labeled)
    adjust_ri = ev.adj_rand_index(pred_labeled, label_labeled)

    # 保存预测结果（可选）
    if test_save_path is not None:
        os.makedirs(test_save_path, exist_ok=True)
        pred_img = Image.fromarray(pred_255)
        pred_img.save(os.path.join(test_save_path, f"{case}_pred.png"))

    # 构造指标返回格式（兼容原代码的metric_list累加逻辑）
    metric = [
        [dice, miou, f_score, map_score, ri, adjust_ri, vi, merger_error, split_error]
    ]
    return metric


# ===================== 命令行参数 =====================
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str,
                    default='F:/iron/datasets/segmentation/net_test/test')
parser.add_argument('--dataset', type=str,
                    default='Custom2D')
parser.add_argument('--num_classes', type=int,
                    default=2)
parser.add_argument('--list_dir', type=str,
                    default=None)

# 保留通用训练/推理参数
parser.add_argument('--max_iterations', type=int, default=200)
parser.add_argument('--max_epochs', type=int, default=30)
parser.add_argument('--batch_size', type=int, default=24)
parser.add_argument('--img_size', type=int, default=256)
parser.add_argument('--is_savenii', action="store_true")

parser.add_argument('--n_skip', type=int, default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str, default='ViT-B_16', help='select one vit model')

parser.add_argument('--test_save_dir', type=str, default='F:/iron/datasets/segmentation/result')
parser.add_argument('--deterministic', type=int, default=1)
parser.add_argument('--base_lr', type=float, default=0.01)
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--vit_patches_size', type=int, default=16, help='vit_patches_size, default is 16')
args = parser.parse_args()


def inference(args, model, test_save_path=None):
    # 加载2D测试集
    db_test = Custom2DDataset(
        base_dir=args.data_path,
        split="test",
        list_dir=args.list_dir
    )
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()


    metric_list = np.zeros((args.num_classes - 1, 9))

    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]

        metric_i = test_single_2d_image(
            image, label, model,
            classes=args.num_classes,
            patch_size=[args.img_size, args.img_size],
            test_save_path=test_save_path,
            case=case_name
        )
        metric_list += np.array(metric_i)

        # 打印单样本指标
        single_metric = np.mean(metric_i, axis=0)
        logging.info(
            'idx %d case %s | dice: %.4f, miou: %.4f, f_score: %.4f, map: %.4f, ri: %.4f, adjust_ri: %.4f, vi: %.4f' % (
                i_batch, case_name,
                single_metric[0], single_metric[1], single_metric[2], single_metric[3],
                single_metric[4], single_metric[5], single_metric[6]
            )
        )

    # 计算平均指标
    metric_list = metric_list / len(db_test)
    for i in range(1, args.num_classes):
        logging.info(
            'Mean class %d | dice: %.4f, miou: %.4f, f_score: %.4f, map: %.4f, ri: %.4f, adjust_ri: %.4f, vi: %.4f' % (
                i, metric_list[i - 1][0], metric_list[i - 1][1], metric_list[i - 1][2],
                metric_list[i - 1][3], metric_list[i - 1][4], metric_list[i - 1][5], metric_list[i - 1][6]
            )
        )

    mean_metric = np.mean(metric_list, axis=0)
    logging.info('Testing performance in best val model:')
    logging.info('mean_dice: %.4f, mean_miou: %.4f, mean_f_score: %.4f, mean_map: %.4f' % (
        mean_metric[0], mean_metric[1], mean_metric[2], mean_metric[3]
    ))
    logging.info('mean_ri: %.4f, mean_adjust_ri: %.4f, mean_vi: %.4f' % (
        mean_metric[4], mean_metric[5], mean_metric[6]
    ))
    return "Testing Finished!"


if __name__ == "__main__":
    # 固定随机种子
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_config = {
        'Custom2D': {
            'num_classes': args.num_classes,
            'data_path': args.data_path,
            'list_dir': args.list_dir,
        },
    }
    dataset_name = args.dataset
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.data_path = dataset_config[dataset_name]['data_path']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.is_pretrain = True

    # 模型快照路径
    args.exp = 'TU_' + dataset_name + str(args.img_size)
    snapshot_path = "../model/{}/{}".format(args.exp, 'TU')
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path += '_' + args.vit_name
    snapshot_path = snapshot_path + '_skip' + str(args.n_skip)
    snapshot_path = snapshot_path + '_vitpatch' + str(
        args.vit_patches_size) if args.vit_patches_size != 16 else snapshot_path
    snapshot_path = snapshot_path + '_epo' + str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    snapshot_path = snapshot_path + '_bs' + str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_' + str(args.img_size)
    snapshot_path = snapshot_path + '_s' + str(args.seed) if args.seed != 1234 else snapshot_path

    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    config_vit.patches.size = (args.vit_patches_size, args.vit_patches_size)
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (
            int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
    net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()

    snapshot = os.path.join(snapshot_path, 'best_model.pth')
    if not os.path.exists(snapshot):
        snapshot = snapshot.replace('best_model', 'epoch_' + str(args.max_epochs - 1))
    net.load_state_dict(torch.load(snapshot))
    snapshot_name = snapshot_path.split('/')[-1]

    log_folder = './test_log/test_log_' + args.exp
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(
        filename=log_folder + '/' + snapshot_name + ".txt",
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S'
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)

    if args.is_savenii:
        test_save_path = os.path.join(args.test_save_dir, args.exp, snapshot_name)
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None


    inference(args, net, test_save_path)