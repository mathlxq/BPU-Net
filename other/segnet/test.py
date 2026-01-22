import glob
import random
import json
import os
import six
import numpy as np
import cv2
from tqdm import tqdm
from time import time
from skimage.measure import label
import gala.evaluate as ev

# 导入segnet模型
from .models.segnet import segnet, vgg_segnet, resnet50_segnet, mobilenet_segnet
from .train import find_latest_checkpoint
from .data_utils.data_loader import get_image_array, get_segmentation_array, \
    DATA_LOADER_SEED, class_colors, get_pairs_from_paths
from .models.config import IMAGE_ORDERING


random.seed(DATA_LOADER_SEED)

def calculate_dice(pred, mask):
    pred_bin = (pred == 1).astype(np.uint8)
    mask_bin = (mask == 1).astype(np.uint8)

    intersection = np.sum(pred_bin & mask_bin)
    sum_pred = np.sum(pred_bin)
    sum_mask = np.sum(mask_bin)

    if sum_pred + sum_mask == 0:
        return 1.0
    return 2.0 * intersection / (sum_pred + sum_mask)


def calculate_miou(pred, mask):
    pred_bin = (pred == 1).astype(np.uint8)
    mask_bin = (mask == 1).astype(np.uint8)

    cm = np.zeros((2, 2), dtype=np.int64)
    cm[0, 0] = np.sum((mask_bin == 0) & (pred_bin == 0))  # TN
    cm[0, 1] = np.sum((mask_bin == 0) & (pred_bin == 1))  # FP
    cm[1, 0] = np.sum((mask_bin == 1) & (pred_bin == 0))  # FN
    cm[1, 1] = np.sum((mask_bin == 1) & (pred_bin == 1))  # TP

    ious = []
    for i in range(2):
        true_positives = cm[i, i]
        false_positives = cm[:, i].sum() - true_positives
        false_negatives = cm[i, :].sum() - true_positives

        denominator = true_positives + false_positives + false_negatives
        if denominator == 0:
            ious.append(1.0)  # 无像素时IoU为1
        else:
            ious.append(true_positives / denominator)

    return np.mean(ious)


def get_map_2018kdasb_new(pred, mask):
    pred_255 = (pred == 1).astype(np.uint8) * 255
    mask_255 = (mask == 1).astype(np.uint8) * 255

    thresholds = np.array([0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])
    tp = np.zeros(10)

    mask_255 = 255 - mask_255
    pred_255 = 255 - pred_255

    label_mask, num_mask = label(mask_255, background=0, return_num=True, connectivity=1)
    label_pred, num_pred = label(pred_255, background=0, return_num=True, connectivity=1)

    for i_pred in range(1, num_pred + 1):
        intersect_mask_labels = list(np.unique(label_mask[label_pred == i_pred]))
        if 0 in intersect_mask_labels:
            intersect_mask_labels.remove(0)

        if len(intersect_mask_labels) == 0:
            continue

        intersect_mask_label_area = np.zeros((len(intersect_mask_labels), 1))
        union_mask_label_area = np.zeros((len(intersect_mask_labels), 1))

        for index, i_mask in enumerate(intersect_mask_labels):
            intersect_mask_label_area[index, 0] = np.count_nonzero(label_pred[label_mask == i_mask] == i_pred)
            union_mask_label_area[index, 0] = np.count_nonzero((label_mask == i_mask) | (label_pred == i_pred))

        iou = intersect_mask_label_area / union_mask_label_area
        max_iou = np.max(iou, axis=0)
        tp[thresholds < max_iou] = tp[thresholds < max_iou] + 1

    fp = num_pred - tp
    fn = num_mask - tp
    map_score = np.average(tp / (tp + fp + fn + 1e-8))  # 避免除零
    return map_score


def calculate_vi_ari(pred, mask):
    pred_label = label((pred == 1).astype(np.uint8), background=0, connectivity=1)
    mask_label = label((mask == 1).astype(np.uint8), background=0, connectivity=1)

    merger_error, split_error = ev.split_vi(pred_label, mask_label)
    vi = merger_error + split_error

    ari = ev.adj_rand_index(pred_label, mask_label)

    return vi, ari



def model_from_checkpoint_path(checkpoints_path, n_classes=2, channels=1):

    assert os.path.isfile(checkpoints_path + "_config.json"), "Checkpoint not found."
    model_config = json.loads(open(checkpoints_path + "_config.json", "r").read())

    # 优先使用配置文件的参数
    n_classes = model_config.get('n_classes', n_classes)
    input_height = model_config.get('input_height', 416)
    input_width = model_config.get('input_width', 608)
    channels = model_config.get('channels', channels)
    model_name = model_config.get('model_class', 'vgg_segnet')

    # 加载对应SegNet模型
    if model_name == 'segnet':
        model = segnet(n_classes, input_height, input_width, channels=channels)
    elif model_name == 'vgg_segnet':
        model = vgg_segnet(n_classes, input_height, input_width, channels=channels)
    elif model_name == 'resnet50_segnet':
        model = resnet50_segnet(n_classes, input_height, input_width, channels=channels)
    elif model_name == 'mobilenet_segnet':
        model = mobilenet_segnet(n_classes, input_height, input_width, channels=channels)
    else:
        raise ValueError(f"不支持的模型类型: {model_name}")

    # 加载权重
    latest_weights = find_latest_checkpoint(checkpoints_path)
    assert latest_weights is not None, "Checkpoint weights not found."
    print(f"Loaded weights: {latest_weights}")
    status = model.load_weights(latest_weights)
    if status is not None:
        status.expect_partial()

    return model


def get_colored_segmentation_image(seg_arr, n_classes, colors=class_colors):
    output_height = seg_arr.shape[0]
    output_width = seg_arr.shape[1]
    seg_img = np.zeros((output_height, output_width, 3), dtype=np.uint8)

    colors = [(0, 0, 0), (255, 255, 255)] if n_classes == 2 else colors
    for c in range(n_classes):
        seg_arr_c = seg_arr[:, :] == c
        seg_img[:, :, 0] += (seg_arr_c * colors[c][0]).astype('uint8')
        seg_img[:, :, 1] += (seg_arr_c * colors[c][1]).astype('uint8')
        seg_img[:, :, 2] += (seg_arr_c * colors[c][2]).astype('uint8')

    return seg_img


def get_legends(class_names, colors=class_colors):
    n_classes = len(class_names)
    legend = np.zeros(((len(class_names) * 25) + 25, 125, 3), dtype="uint8") + 255

    if n_classes == 2 and class_names is None:
        class_names = ['Background', 'Grain']
        colors = [(0, 0, 0), (255, 255, 255)]

    class_names_colors = enumerate(zip(class_names[:n_classes], colors[:n_classes]))
    for (i, (class_name, color)) in class_names_colors:
        color = [int(c) for c in color]
        cv2.putText(legend, class_name, (5, (i * 25) + 17),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)
        cv2.rectangle(legend, (100, (i * 25)), (125, (i * 25) + 25),
                      tuple(color), -1)

    return legend

def overlay_seg_image(inp_img, seg_img):
    orininal_h = inp_img.shape[0]
    orininal_w = inp_img.shape[1]
    seg_img = cv2.resize(seg_img, (orininal_w, orininal_h), interpolation=cv2.INTER_NEAREST)
    fused_img = (inp_img / 2 + seg_img / 2).astype('uint8')
    return fused_img

def concat_lenends(seg_img, legend_img):
    new_h = np.maximum(seg_img.shape[0], legend_img.shape[0])
    new_w = seg_img.shape[1] + legend_img.shape[1]
    out_img = np.zeros((new_h, new_w, 3)).astype('uint8') + legend_img[0, 0, 0]
    out_img[:legend_img.shape[0], :legend_img.shape[1]] = np.copy(legend_img)
    out_img[:seg_img.shape[0], legend_img.shape[1]:] = np.copy(seg_img)
    return out_img

def visualize_segmentation(seg_arr, inp_img=None, n_classes=None,
                           colors=class_colors, class_names=None,
                           overlay_img=False, show_legends=False,
                           prediction_width=None, prediction_height=None):
    if n_classes is None:
        n_classes = np.max(seg_arr) + 1

    seg_img = get_colored_segmentation_image(seg_arr, n_classes, colors=colors)

    if inp_img is not None:
        original_h, original_w = inp_img.shape[:2]
        seg_img = cv2.resize(seg_img, (original_w, original_h), interpolation=cv2.INTER_NEAREST)

    if prediction_height and prediction_width:
        seg_img = cv2.resize(seg_img, (prediction_width, prediction_height), interpolation=cv2.INTER_NEAREST)
        if inp_img is not None:
            inp_img = cv2.resize(inp_img, (prediction_width, prediction_height))

    if overlay_img:
        assert inp_img is not None
        seg_img = overlay_seg_image(inp_img, seg_img)

    if show_legends:
        assert class_names is not None or n_classes == 2
        if n_classes == 2 and class_names is None:
            class_names = ['Background', 'Grain']
        legend_img = get_legends(class_names, colors=colors)
        seg_img = concat_lenends(seg_img, legend_img)

    return seg_img

def predict(model=None, inp=None, out_fname=None,
            checkpoints_path=None, overlay_img=False,
            class_names=None, show_legends=False, colors=class_colors,
            prediction_width=None, prediction_height=None,
            read_image_type=0):  # 灰度图默认read_image_type=0

    if model is None and checkpoints_path is not None:
        model = model_from_checkpoint_path(checkpoints_path)

    assert inp is not None
    assert (type(inp) is np.ndarray) or isinstance(inp, six.string_types)

    # 读取图片
    if isinstance(inp, six.string_types):
        inp = cv2.imread(inp, read_image_type)

    if len(inp.shape) == 2:
        inp = np.expand_dims(inp, axis=-1)

    output_width = model.output_width
    output_height = model.output_height
    input_width = model.input_width
    input_height = model.input_height
    n_classes = model.n_classes

    x = get_image_array(inp, input_width, input_height, ordering=IMAGE_ORDERING)
    pr = model.predict(np.array([x]), verbose=0)[0]
    pr = pr.reshape((output_height, output_width, n_classes)).argmax(axis=2)

    seg_img = visualize_segmentation(pr, inp, n_classes=n_classes,
                                     colors=colors, overlay_img=overlay_img,
                                     show_legends=show_legends,
                                     class_names=class_names,
                                     prediction_width=prediction_width,
                                     prediction_height=prediction_height)

    if out_fname is not None:
        cv2.imwrite(out_fname, seg_img)

    return pr



def evaluate_segnet_grain(
        checkpoints_path,
        inp_images_dir,
        annotations_dir,
        out_result_dir,
        n_classes=2,
        channels=1,
        read_image_type=0
):

    os.makedirs(out_result_dir, exist_ok=True)
    pred_dir = os.path.join(out_result_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)

    # 2. 加载模型
    model = model_from_checkpoint_path(checkpoints_path, n_classes, channels)

    # 3. 获取测试集路径对
    paths = get_pairs_from_paths(inp_images_dir, annotations_dir)
    inp_images = [p[0] for p in paths]
    annotations = [p[1] for p in paths]
    assert len(inp_images) > 0, "测试集为空！"

    # 4. 初始化指标汇总
    metrics = {
        'dice': [],
        'miou': [],
        'vi': [],
        'ari': [],
        'map': []
    }

    for img_path, ann_path in tqdm(zip(inp_images, annotations), total=len(inp_images)):

        img_name = os.path.basename(img_path)
        pred_out_path = os.path.join(pred_dir, img_name)
        pr = predict(model, img_path, pred_out_path, checkpoints_path, read_image_type=read_image_type)

        ann = cv2.imread(ann_path, read_image_type)

        ann = cv2.resize(ann, (model.output_width, model.output_height), interpolation=cv2.INTER_NEAREST)
        ann_bin = (ann > 127).astype(np.uint8)

        dice = calculate_dice(pr, ann_bin)
        miou = calculate_miou(pr, ann_bin)
        vi, ari = calculate_vi_ari(pr, ann_bin)
        map_score = get_map_2018kdasb_new(pr, ann_bin)

        metrics['dice'].append(dice)
        metrics['miou'].append(miou)
        metrics['vi'].append(vi)
        metrics['ari'].append(ari)
        metrics['map'].append(map_score)

    avg_metrics = {
        'avg_dice': np.mean(metrics['dice']),
        'avg_miou': np.mean(metrics['miou']),
        'avg_vi': np.mean(metrics['vi']),
        'avg_ari': np.mean(metrics['ari']),
        'avg_map': np.mean(metrics['map']),
        'std_dice': np.std(metrics['dice']),
        'std_miou': np.std(metrics['miou'])
    }

    result_file = os.path.join(out_result_dir, "evaluation_metrics.txt")
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write(f"平均Dice系数: {avg_metrics['avg_dice']:.4f} (±{avg_metrics['std_dice']:.4f})\n")
        f.write(f"平均mIoU: {avg_metrics['avg_miou']:.4f} (±{avg_metrics['std_miou']:.4f})\n")
        f.write(f"平均VI: {avg_metrics['avg_vi']:.4f}\n")
        f.write(f"平均ARI: {avg_metrics['avg_ari']:.4f}\n")
        f.write(f"平均mAP: {avg_metrics['avg_map']:.4f}\n")

        f.write("\n=== 单样本详细指标 ===\n")
        f.write("文件名,Dice,mIoU,VI,ARI,mAP\n")
        for i, (img_path, dice, miou, vi, ari, map_score) in enumerate(
                zip(inp_images, metrics['dice'], metrics['miou'], metrics['vi'], metrics['ari'], metrics['map'])
        ):
            f.write(f"{os.path.basename(img_path)},{dice:.4f},{miou:.4f},{vi:.4f},{ari:.4f},{map_score:.4f}\n")

    print("\n=== 评估完成 ===")
    print(f"平均Dice系数: {avg_metrics['avg_dice']:.4f}")
    print(f"平均mIoU: {avg_metrics['avg_miou']:.4f}")
    print(f"平均VI: {avg_metrics['avg_vi']:.4f}")
    print(f"平均ARI: {avg_metrics['avg_ari']:.4f}")
    print(f"平均mAP: {avg_metrics['avg_map']:.4f}")

    return avg_metrics



if __name__ == '__main__':
    # 配置参数（根据自己的数据集修改）
    CHECKPOINTS_PATH = "./checkpoints/segnet_grain"  # 模型检查点路径
    INP_IMAGES_DIR = "./datasets/test/images"  # 测试图片目录
    ANNOTATIONS_DIR = "./datasets/test/labels"  # 标注图片目录
    OUT_RESULT_DIR = "./results/segnet"  # 结果输出目录
    N_CLASSES = 2  # 二分类（前景+背景）
    CHANNELS = 1  # 晶粒图像为灰度图，通道数=1

    # 执行评估
    evaluate_segnet_grain(
        checkpoints_path=CHECKPOINTS_PATH,
        inp_images_dir=INP_IMAGES_DIR,
        annotations_dir=ANNOTATIONS_DIR,
        out_result_dir=OUT_RESULT_DIR,
        n_classes=N_CLASSES,
        channels=CHANNELS,
        read_image_type=0
    )