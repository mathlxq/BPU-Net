import glob
import random
import json
import os
import six

import cv2
import numpy as np
from tqdm import tqdm
from time import time
from skimage.measure import label
import gala.evaluate as ev

from .train import find_latest_checkpoint
from .data_utils.data_loader import get_image_array, get_segmentation_array, \
    DATA_LOADER_SEED, class_colors, get_pairs_from_paths
from .models.config import IMAGE_ORDERING


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

    cm = np.zeros((2, 2))
    cm[0, 0] = np.sum((pred_bin == 0) & (mask_bin == 0))  # TN
    cm[0, 1] = np.sum((pred_bin == 1) & (mask_bin == 0))  # FP
    cm[1, 0] = np.sum((pred_bin == 0) & (mask_bin == 1))  # FN
    cm[1, 1] = np.sum((pred_bin == 1) & (mask_bin == 1))  # TP

    ious = []
    for i in range(2):
        true_positives = cm[i, i]
        false_positives = cm[:, i].sum() - true_positives
        false_negatives = cm[i, :].sum() - true_positives

        denominator = true_positives + false_positives + false_negatives
        if denominator == 0:
            ious.append(1.0)
        else:
            ious.append(true_positives / denominator)

    return np.mean(ious)


def get_dis_from_mask_point(mask, index_x, index_y, neighbor_length=60):
    if mask[index_x, index_y] == 1:
        return 0
    region_start_row = 0
    region_start_col = 0
    region_end_row = mask.shape[0]
    region_end_col = mask.shape[1]
    if index_x - neighbor_length > 0:
        region_start_row = index_x - neighbor_length
    if index_x + neighbor_length < mask.shape[0]:
        region_end_row = index_x + neighbor_length
    if index_y - neighbor_length > 0:
        region_start_col = index_y - neighbor_length
    if index_y + neighbor_length < mask.shape[1]:
        region_end_col = index_y + neighbor_length
    x, y = np.where(mask[region_start_row: region_end_row, region_start_col: region_end_col] == 1)

    if len(x) == 0:
        min_distance = 30
    else:
        min_distance = np.amin(
            np.linalg.norm(np.array([x + region_start_row, y + region_start_col]) - np.array([[index_x], [index_y]]),
                           axis=0))

    return min_distance


def get_figure_of_merit(pred, mask, const_index=0.1):
    num_pred = np.count_nonzero(pred == 1)
    num_mask = np.count_nonzero(mask == 1)
    num_max = num_pred if num_pred > num_mask else num_mask

    if num_max == 0:
        return 0.0

    temp = 0.0
    for index_x in range(0, pred.shape[0]):
        for index_y in range(0, pred.shape[1]):
            if pred[index_x, index_y] == 1:
                distance = get_dis_from_mask_point(mask, index_x, index_y)
                temp = temp + 1 / (1 + const_index * pow(distance, 2))
    f_score = (1.0 / num_max) * temp
    return f_score


def get_map_2018kdasb_new(pred, mask, target_image=0):
    thresholds = np.array([0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])
    tp = np.zeros(10)
    if target_image == 0:
        pred[0, :] = 1
        pred[:, 0] = 1
        pred[-1, :] = 1
        pred[:, -1] = 1
        mask[0, :] = 1
        mask[:, 0] = 1
        mask[-1, :] = 1
        mask[:, -1] = 1
        mask = 1 - mask
        pred = 1 - pred

    label_mask, num_mask = label(mask, background=0, return_num=True, connectivity=1)
    label_pred, num_pred = label(pred, background=0, return_num=True, connectivity=1)

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
    map_score = np.average(tp / (tp + fp + fn + 1e-8))
    return map_score


random.seed(DATA_LOADER_SEED)


def model_from_checkpoint_path(checkpoints_path):
    from .models.all_models import model_from_name
    assert (os.path.isfile(checkpoints_path + "_config.json")), "Checkpoint not found."
    model_config = json.loads(open(checkpoints_path + "_config.json", "r").read())
    latest_weights = find_latest_checkpoint(checkpoints_path)
    assert (latest_weights is not None), "Checkpoint not found."
    model = model_from_name[model_config['model_class']](
        model_config['n_classes'], input_height=model_config['input_height'],
        input_width=model_config['input_width'])
    print("loaded weights ", latest_weights)
    status = model.load_weights(latest_weights)

    if status is not None:
        status.expect_partial()

    return model


def get_colored_segmentation_image(seg_arr, n_classes, colors=class_colors):
    output_height = seg_arr.shape[0]
    output_width = seg_arr.shape[1]

    seg_img = np.zeros((output_height, output_width, 3))

    for c in range(n_classes):
        seg_arr_c = seg_arr[:, :] == c
        seg_img[:, :, 0] += ((seg_arr_c) * (colors[c][0])).astype('uint8')
        seg_img[:, :, 1] += ((seg_arr_c) * (colors[c][1])).astype('uint8')
        seg_img[:, :, 2] += ((seg_arr_c) * (colors[c][2])).astype('uint8')

    return seg_img


def get_legends(class_names, colors=class_colors):
    n_classes = len(class_names)
    legend = np.zeros(((len(class_names) * 25) + 25, 125, 3), dtype="uint8") + 255

    class_names_colors = enumerate(zip(class_names[:n_classes], colors[:n_classes]))

    for (i, (class_name, color)) in class_names_colors:
        color = [int(c) for c in color]
        cv2.putText(legend, class_name, (5, (i * 25) + 17), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)
        cv2.rectangle(legend, (100, (i * 25)), (125, (i * 25) + 25), tuple(color), -1)

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
        original_h = inp_img.shape[0]
        original_w = inp_img.shape[1]
        seg_img = cv2.resize(seg_img, (original_w, original_h), interpolation=cv2.INTER_NEAREST)

    if (prediction_height is not None) and (prediction_width is not None):
        seg_img = cv2.resize(seg_img, (prediction_width, prediction_height), interpolation=cv2.INTER_NEAREST)
        if inp_img is not None:
            inp_img = cv2.resize(inp_img, (prediction_width, prediction_height))

    if overlay_img:
        assert inp_img is not None
        seg_img = overlay_seg_image(inp_img, seg_img)

    if show_legends:
        assert class_names is not None
        legend_img = get_legends(class_names, colors=colors)
        seg_img = concat_lenends(seg_img, legend_img)

    return seg_img


def predict(model=None, inp=None, out_fname=None,
            checkpoints_path=None, overlay_img=False,
            class_names=None, show_legends=False, colors=class_colors,
            prediction_width=None, prediction_height=None,
            read_image_type=1):
    if model is None and (checkpoints_path is not None):
        model = model_from_checkpoint_path(checkpoints_path)

    assert (inp is not None)
    assert ((type(inp) is np.ndarray) or isinstance(inp, six.string_types)), \
        "Input should be the CV image or the input file name"

    if isinstance(inp, six.string_types):
        inp = cv2.imread(inp, read_image_type)

    assert (len(inp.shape) in [1, 3, 4]), "Image should be h,w or h,w,3 or h,w,4"

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


def predict_multiple(model=None, inps=None, inp_dir=None, out_dir=None,
                     checkpoints_path=None, overlay_img=False,
                     class_names=None, show_legends=False, colors=class_colors,
                     prediction_width=None, prediction_height=None, read_image_type=1,
                     ann_dir=None, eval_save_path=None):

    if model is None and (checkpoints_path is not None):
        model = model_from_checkpoint_path(checkpoints_path)

    if inps is None and (inp_dir is not None):
        inps = glob.glob(os.path.join(inp_dir, "*.jpg")) + glob.glob(
            os.path.join(inp_dir, "*.png")) + glob.glob(os.path.join(inp_dir, "*.jpeg"))
        inps = sorted(inps)

    assert type(inps) is list
    all_prs = []


    total_vi = 0.0
    total_ari = 0.0
    total_map = 0.0
    total_dice = 0.0
    total_miou = 0.0
    count = 0

    ann_paths = []
    if ann_dir is not None:
        for inp_path in inps:
            fname = os.path.basename(inp_path).split('.')[0]
            ann_path = glob.glob(os.path.join(ann_dir, f"{fname}.*"))
            if ann_path:
                ann_paths.append(ann_path[0])
            else:
                ann_paths.append(None)

    if eval_save_path is not None:
        os.makedirs(os.path.dirname(eval_save_path), exist_ok=True)
        with open(eval_save_path, 'w', encoding='utf-8') as f:
            f.write("filename,VI,ARI,mAP,Dice,mIoU\n")

    if not out_dir is None:
        os.makedirs(out_dir, exist_ok=True)

    for i, (inp, ann_path) in enumerate(tqdm(zip(inps, ann_paths))):
        if out_dir is None:
            out_fname = None
        else:
            if isinstance(inp, six.string_types):
                out_fname = os.path.join(out_dir, os.path.basename(inp))
            else:
                out_fname = os.path.join(out_dir, str(i) + ".jpg")

        pr = predict(model, inp, out_fname,
                     overlay_img=overlay_img, class_names=class_names,
                     show_legends=show_legends, colors=colors,
                     prediction_width=prediction_width,
                     prediction_height=prediction_height, read_image_type=read_image_type)
        all_prs.append(pr)

        # 计算评价指标
        if ann_path is not None and os.path.exists(ann_path):
            count += 1
            gt = cv2.imread(ann_path, 0)
            gt = cv2.resize(gt, (pr.shape[1], pr.shape[0]), interpolation=cv2.INTER_NEAREST)
            gt_bin = (gt > 127).astype(np.uint8)
            pr_bin = pr.astype(np.uint8)

            pr_labeled, _ = label(pr_bin, background=0, return_num=True, connectivity=1)
            gt_labeled, _ = label(gt_bin, background=0, return_num=True, connectivity=1)
            merger_error, split_error = ev.split_vi(pr_labeled, gt_labeled)
            vi = merger_error + split_error
            ari = ev.adj_rand_index(pr_labeled, gt_labeled)

            # mAP
            map_score = get_map_2018kdasb_new(pr_bin, gt_bin)

            #  Dice
            dice = calculate_dice(pr_bin, gt_bin)

            #  mIoU
            miou = calculate_miou(pr_bin, gt_bin)

            # 累加指标
            total_vi += vi
            total_ari += ari
            total_map += map_score
            total_dice += dice
            total_miou += miou
            if eval_save_path is not None:
                fname = os.path.basename(inp)
                with open(eval_save_path, 'a', encoding='utf-8') as f:
                    f.write(f"{fname},{vi:.4f},{ari:.4f},{map_score:.4f},{dice:.4f},{miou:.4f}\n")

    if count > 0:
        avg_vi = total_vi / count
        avg_ari = total_ari / count
        avg_map = total_map / count
        avg_dice = total_dice / count
        avg_miou = total_miou / count
        print(f"平均VI: {avg_vi:.4f}")
        print(f"平均ARI: {avg_ari:.4f}")
        print(f"平均mAP: {avg_map:.4f}")
        print(f"平均Dice: {avg_dice:.4f}")
        print(f"平均mIoU: {avg_miou:.4f}")

        if eval_save_path is not None:
            with open(eval_save_path, 'a', encoding='utf-8') as f:
                f.write(f"\n平均值,,,{avg_vi:.4f},{avg_ari:.4f},{avg_map:.4f},{avg_dice:.4f},{avg_miou:.4f}\n")

    return all_prs


def set_video(inp, video_name):
    cap = cv2.VideoCapture(inp)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    size = (video_width, video_height)
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    video = cv2.VideoWriter(video_name, fourcc, fps, size)
    return cap, video, fps


def predict_video(model=None, inp=None, output=None,
                  checkpoints_path=None, display=False, overlay_img=True,
                  class_names=None, show_legends=False, colors=class_colors,
                  prediction_width=None, prediction_height=None):
    if model is None and (checkpoints_path is not None):
        model = model_from_checkpoint_path(checkpoints_path)
    n_classes = model.n_classes

    cap, video, fps = set_video(inp, output)
    while (cap.isOpened()):
        prev_time = time()
        ret, frame = cap.read()
        if frame is not None:
            pr = predict(model=model, inp=frame)
            fused_img = visualize_segmentation(
                pr, frame, n_classes=n_classes,
                colors=colors,
                overlay_img=overlay_img,
                show_legends=show_legends,
                class_names=class_names,
                prediction_width=prediction_width,
                prediction_height=prediction_height
            )
        else:
            break
        print("FPS: {:.2f}".format(1 / (time() - prev_time)))
        if output is not None:
            video.write(fused_img)
        if display:
            cv2.imshow('Frame masked', fused_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    if output is not None:
        video.release()
    cv2.destroyAllWindows()


def evaluate(model=None, inp_images=None, annotations=None,
             inp_images_dir=None, annotations_dir=None, checkpoints_path=None, read_image_type=1):
    if model is None:
        assert (checkpoints_path is not None), "Please provide the model or the checkpoints_path"
        model = model_from_checkpoint_path(checkpoints_path)

        paths = get_pairs_from_paths(inp_images_dir, annotations_dir)
        paths = list(zip(*paths))
        inp_images = list(paths[0])
        annotations = list(paths[1])

    assert type(inp_images) is list
    assert type(annotations) is list

    tp = np.zeros(model.n_classes)
    fp = np.zeros(model.n_classes)
    fn = np.zeros(model.n_classes)
    n_pixels = np.zeros(model.n_classes)

    for inp, ann in tqdm(zip(inp_images, annotations)):
        pr = predict(model, inp, read_image_type=read_image_type)
        gt = get_segmentation_array(ann, model.n_classes,
                                    model.output_width, model.output_height,
                                    no_reshape=True, read_image_type=read_image_type)
        gt = gt.argmax(-1)
        pr = pr.flatten()
        gt = gt.flatten()

        for cl_i in range(model.n_classes):
            tp[cl_i] += np.sum((pr == cl_i) * (gt == cl_i))
            fp[cl_i] += np.sum((pr == cl_i) * ((gt != cl_i)))
            fn[cl_i] += np.sum((pr != cl_i) * ((gt == cl_i)))
            n_pixels[cl_i] += np.sum(gt == cl_i)

    cl_wise_score = tp / (tp + fp + fn + 1e-8)
    n_pixels_norm = n_pixels / (np.sum(n_pixels) + 1e-8)
    frequency_weighted_IU = np.sum(cl_wise_score * n_pixels_norm)
    mean_IU = np.mean(cl_wise_score)

    return {
        "frequency_weighted_IU": frequency_weighted_IU,
        "mean_IU": mean_IU,
        "class_wise_IU": cl_wise_score
    }


if __name__ == "__main__":
    checkpoint_path = "./checkpoints/fcn"
    input_dir = "./datasets/test/images"
    ann_dir = "./datasets//test/labels"
    output_dir = "./predict_results/result"  
    eval_save_path = "./eval_results/grain_eval.csv"

       predict_multiple(
        checkpoints_path=checkpoint_path,
        inp_dir=input_dir,
        out_dir=output_dir,
        ann_dir=ann_dir,
        eval_save_path=eval_save_path,
        read_image_type=0
    )