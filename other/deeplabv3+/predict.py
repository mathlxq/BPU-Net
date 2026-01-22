from torch.utils.data import dataset
from tqdm import tqdm
import network
import utils
import os
import random
import argparse
import numpy as np

from torch.utils import data
from torchvision import transforms as T

import torch
import torch.nn as nn
import cv2
from PIL import Image
from glob import glob
import evaluation
from skimage.measure import label


def get_argparser():
    parser = argparse.ArgumentParser()

    # 数据集适配
    parser.add_argument("--input", type=str, required=True,
                        help="path to a single image or image directory ")
    parser.add_argument("--gt_dir", type=str, required=True,
                        help="path to ground truth label directory ")
    parser.add_argument("--save_val_results_to", default=None,
                        help="save segmentation results to the specified dir")
    parser.add_argument("--eval_ri_vi_save_to", type=str, default="./eval_RI_VI_results.txt",
                        help="save RI/VI/ARI/mIoU/Dice results to txt")
    parser.add_argument("--eval_f_map_save_to", type=str, default="./eval_F_mAP_results.txt",
                        help="save F-score/mAP/mIoU/Dice results to txt")
    parser.add_argument("--eval_all_metrics_save_to", type=str, default="./eval_all_metrics_summary.txt",
                        help="save all metrics summary ")

    # Deeplab Options
    available_models = sorted(name for name in network.modeling.__dict__ if name.islower() and \
                              not (name.startswith("__") or name.startswith('_')) and callable(
        network.modeling.__dict__[name])
                              )
    parser.add_argument("--model", type=str, default='deeplabv3plus_mobilenet',
                        choices=available_models, help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--val_batch_size", type=int, default=4,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=513)
    parser.add_argument("--num_classes", type=int, default=2,
                        )
    parser.add_argument("--ckpt", default=None, type=str,
                        help="resume from checkpoint")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    return parser


def preprocess_for_evaluation(pred, gt):

    pred = np.where(pred == 1, 255, 0).astype(np.uint8)
    pred = pred[10:-10, 10:-10] if (pred.shape[0] > 20 and pred.shape[1] > 20) else pred
    gt = gt[10:-10, 10:-10] if (gt.shape[0] > 20 and gt.shape[1] > 20) else gt
    if pred.shape != gt.shape:
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
    return pred, gt


def calculate_single_image_metrics(pred, gt):

    dice = evaluation.calculate_dice(pred, gt)
    miou = evaluation.calculate_miou(pred, gt)


    pred_labeled, _ = label(pred, background=255, return_num=True, connectivity=1)
    gt_labeled, _ = label(gt, background=255, return_num=True, connectivity=1)
    merger_error, split_error = evaluation.ev.split_vi(pred_labeled, gt_labeled)
    vi = merger_error + split_error
    ari = evaluation.ev.adj_rand_index(pred_labeled, gt_labeled)
    ri = evaluation.ev.rand_index(pred_labeled, gt_labeled)

    f_score = evaluation.get_figure_of_merit(pred, gt)
    map_score = evaluation.get_map_2018kdasb_new(pred, gt)

    return {
        "dice": dice, "miou": miou,
        "ri": ri, "ari": ari,
        "merger_error": merger_error, "split_error": split_error, "vi": vi,
        "f_score": f_score, "map": map_score
    }


def main():
    opts = get_argparser().parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # 1. 加载图像和标注文件
    image_files = []
    if os.path.isdir(opts.input):
        for ext in ['png', 'jpeg', 'jpg', 'JPEG']:
            files = glob(os.path.join(opts.input, '**/*.%s' % ext), recursive=True)
            if len(files) > 0:
                image_files.extend(files)
    elif os.path.isfile(opts.input):
        image_files.append(opts.input)
    gt_files = {os.path.basename(f).split('.')[0]: os.path.join(opts.gt_dir, os.path.basename(f)) for f in image_files}
    if len(image_files) == 0:
        print("[Error] 未找到输入图像文件！")
        return

    # 2. 初始化模型
    model = network.modeling.__dict__[opts.model](
        num_classes=opts.num_classes,
        output_stride=opts.output_stride
    )
    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)

    # 加载预训练权重
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(device)
        print("Resume model from %s" % opts.ckpt)
        del checkpoint
    else:
        print("[Warning] 未加载预训练权重，预测结果无意义！")
        model = nn.DataParallel(model)
        model.to(device)


    if opts.crop_val:
        transform = T.Compose([
            T.Resize(opts.crop_size),
            T.CenterCrop(opts.crop_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    # 4. 创建结果保存目录
    if opts.save_val_results_to is not None:
        os.makedirs(opts.save_val_results_to, exist_ok=True)

    metrics_accumulator = {
        "dice": 0.0, "miou": 0.0,
        "ri": 0.0, "ari": 0.0,
        "merger_error": 0.0, "split_error": 0.0, "vi": 0.0,
        "f_score": 0.0, "map": 0.0
    }
    valid_count = 0


    with open(opts.eval_ri_vi_save_to, "w") as f:
        f.write("img_name,RI,mRI,ARI,mARI,merger_error,m_merger_error,split_error,m_split_error,VI,mVI,mIoU,mDice\n")
    with open(opts.eval_f_map_save_to, "w") as f:
        f.write("img_name,F-score,avF,mAP,avmAP,mIoU,avmIoU,Dice,avDice\n")


    with torch.no_grad():
        model = model.eval()
        for img_path in tqdm(image_files):

            img_name = os.path.basename(img_path).split('.')[0]
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device)


            pred = model(img_tensor).max(1)[1].cpu().numpy()[0]


            if opts.save_val_results_to:
                pred_img = Image.fromarray(pred.astype(np.uint8))
                pred_img.save(os.path.join(opts.save_val_results_to, f"{img_name}.png"))


            if img_name not in gt_files or not os.path.exists(gt_files[img_name]):

                continue
            valid_count += 1


            gt = cv2.imread(gt_files[img_name], 0)
            pred_processed, gt_processed = preprocess_for_evaluation(pred, gt)


            metrics = calculate_single_image_metrics(pred_processed, gt_processed)


            for k in metrics_accumulator.keys():
                metrics_accumulator[k] += metrics[k]


            current_avg = {k: metrics_accumulator[k] / valid_count for k in metrics_accumulator.keys()}


            with open(opts.eval_ri_vi_save_to, "a") as f:
                f.write(
                    f"{img_name},"
                    f"{metrics['ri']:.6f},{current_avg['ri']:.6f},"
                    f"{metrics['ari']:.6f},{current_avg['ari']:.6f},"
                    f"{metrics['merger_error']:.6f},{current_avg['merger_error']:.6f},"
                    f"{metrics['split_error']:.6f},{current_avg['split_error']:.6f},"
                    f"{metrics['vi']:.6f},{current_avg['vi']:.6f},"
                    f"{metrics['miou']:.6f},{current_avg['dice']:.6f}\n"
                )


            with open(opts.eval_f_map_save_to, "a") as f:
                f.write(
                    f"{img_name},"
                    f"{metrics['f_score']:.6f},{current_avg['f_score']:.6f},"
                    f"{metrics['map']:.6f},{current_avg['map']:.6f},"
                    f"{metrics['miou']:.6f},{current_avg['miou']:.6f},"
                    f"{metrics['dice']:.6f},{current_avg['dice']:.6f}\n"
                )


    if valid_count > 0:
        final_avg = {k: metrics_accumulator[k] / valid_count for k in metrics_accumulator.keys()}

        # 8.1 打印最终结果
        print("\n" + "=" * 80)
        print(f"基础指标：Dice={final_avg['dice']:.6f}, mIoU={final_avg['miou']:.6f}")
        print(f"实例指标：RI={final_avg['ri']:.6f}, ARI={final_avg['ari']:.6f}, VI={final_avg['vi']:.6f}")
        print(f"  - VI分解：merger_error={final_avg['merger_error']:.6f}, split_error={final_avg['split_error']:.6f}")
        print(f"边缘指标：F-score={final_avg['f_score']:.6f}, mAP={final_avg['map']:.6f}")
        print("=" * 80 + "\n")


        with open(opts.eval_all_metrics_save_to, "w") as f:


            f.write(f"Dice系数: {final_avg['dice']:.6f}\n")
            f.write(f"mIoU: {final_avg['miou']:.6f}\n")
            f.write(f"RI（兰德指数）: {final_avg['ri']:.6f}\n")
            f.write(f"ARI（调整兰德指数）: {final_avg['ari']:.6f}\n")
            f.write(f"VI（变化指数）: {final_avg['vi']:.6f}\n")
            f.write(f"  - 合并误差（merger_error）: {final_avg['merger_error']:.6f}\n")
            f.write(f"  - 分割误差（split_error）: {final_avg['split_error']:.6f}\n")
            f.write(f"mAP（2018kdasb）: {final_avg['map']:.6f}\n")
    else:
        print("[Error] ")


if __name__ == '__main__':
    main()