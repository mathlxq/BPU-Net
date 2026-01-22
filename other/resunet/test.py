import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import CustomObjectScope
from data_generator import parse_image, parse_mask, DataGen
from metrics import dice_coef, dice_loss
import evaluation
from skimage.measure import label


def mask_to_3d(mask):
    mask = np.squeeze(mask)
    mask = np.clip(mask, 0, 255).astype(np.uint8)
    mask_3d = np.stack([mask, mask, mask], axis=-1)
    return mask_3d


if __name__ == "__main__":
    model_path = "C:/Users/Lenovo/Desktop/ResUNet/resunet.h5"
    save_path = "C:/Users/Lenovo/Desktop/ResUNet/grain_segmentation_result"
    test_path = "C:/Users/Lenovo/Desktop/ResUNet/datasets/test/"
    image_size = 256
    batch_size = 1
    threshold = 0.5
    eval_result_save_path = os.path.join(save_path, "evaluation_metrics.txt")

    os.makedirs(save_path, exist_ok=True)


    test_image_paths = glob(os.path.join(test_path, "images", "*"))
    test_mask_paths = glob(os.path.join(test_path, "labels", "*"))
    test_image_paths.sort()
    test_mask_paths.sort()


    with CustomObjectScope({'dice_loss': dice_loss, 'dice_coef': dice_coef}):
        try:
            model = load_model(model_path)
        except Exception as e:
            raise Exception(f"模型加载失败：{e}\n")

    dice_scores = []
    miou_scores = []
    f_scores = []
    map_scores = []
    ri_scores = []
    adjust_ri_scores = []
    merger_error_scores = []
    split_error_scores = []
    vi_scores = []


    for i in tqdm(range(len(test_image_paths)), desc="计算指标"):
        image = parse_image(test_image_paths[i], image_size)
        true_mask = parse_mask(test_mask_paths[i], image_size)


        pred_mask = model.predict(np.expand_dims(image, axis=0), verbose=0)[0]
        pred_mask = (pred_mask > threshold) * 255.0

        true_mask = np.squeeze(true_mask) * 255.0
        pred_mask = np.squeeze(pred_mask).astype(np.uint8)
        true_mask = true_mask.astype(np.uint8)


        dice = evaluation.calculate_dice(pred_mask, true_mask)

        miou = evaluation.calculate_miou(pred_mask, true_mask)
        map_score = evaluation.get_map_2018kdasb_new(pred_mask, true_mask)

        pred_mask_crop = pred_mask[10:-10, 10:-10]
        true_mask_crop = true_mask[10:-10, 10:-10]

        pred_label, _ = label(pred_mask_crop, background=255, return_num=True, connectivity=1)
        true_label, _ = label(true_mask_crop, background=255, return_num=True, connectivity=1)
        merger_error, split_error = evaluation.ev.split_vi(pred_label, true_label)
        vi = merger_error + split_error
        ri = evaluation.ev.rand_index(pred_label, true_label)
        adjust_ri = evaluation.ev.adj_rand_index(pred_label, true_label)

        dice_scores.append(dice)
        miou_scores.append(miou)
        map_scores.append(map_score)
        ri_scores.append(ri)
        adjust_ri_scores.append(adjust_ri)
        merger_error_scores.append(merger_error)
        split_error_scores.append(split_error)
        vi_scores.append(vi)

    avg_dice = np.mean(dice_scores)
    avg_miou = np.mean(miou_scores)
    avg_f = np.mean(f_scores)
    avg_map = np.mean(map_scores)
    avg_ri = np.mean(ri_scores)
    avg_adjust_ri = np.mean(adjust_ri_scores)
    avg_merger_error = np.mean(merger_error_scores)
    avg_split_error = np.mean(split_error_scores)
    avg_vi = np.mean(vi_scores)


    print(f"  - 平均Dice系数：{avg_dice:.4f}")
    print(f"  - 平均mIoU：{avg_miou:.4f}")
    print(f"  - 平均mAP（2018kdasb）：{avg_map:.4f}")
    print("-" * 50)
    print(f"实例级指标（RI/VI）：")
    print(f"  - 平均RI（Rand Index）：{avg_ri:.4f}")
    print(f"  - 平均Adjust RI（调整Rand Index）：{avg_adjust_ri:.4f}")
    print(f"  - 平均Merger Error（合并误差）：{avg_merger_error:.4f}")
    print(f"  - 平均Split Error（分割误差）：{avg_split_error:.4f}")
    print(f"  - 平均VI（Variation of Information）：{avg_vi:.4f}")
    print("=" * 50)


    with open(eval_result_save_path, "w", encoding="utf-8") as f:
        f.write("evaluation.py 所有评价指标结果\n")
        f.write("=" * 50 + "\n")
        f.write(f"平均Dice系数：{avg_dice:.4f}\n")
        f.write(f"平均mIoU：{avg_miou:.4f}\n")
        f.write(f"平均mAP：{avg_map:.4f}\n")
        f.write(f"平均RI：{avg_ri:.4f}\n")
        f.write(f"平均Adjust RI：{avg_adjust_ri:.4f}\n")
        f.write(f"平均Merger Error：{avg_merger_error:.4f}\n")
        f.write(f"平均Split Error：{avg_split_error:.4f}\n")
        f.write(f"平均VI：{avg_vi:.4f}\n")

    for i, img_path in tqdm(enumerate(test_image_paths), total=len(test_image_paths)):
        image = parse_image(test_image_paths[i], image_size)
        mask = parse_mask(test_mask_paths[i], image_size)

        predict_mask = model.predict(np.expand_dims(image, axis=0), verbose=0)[0]
        predict_mask = (predict_mask > threshold) * 255.0

        sep_line = np.ones((image_size, 10, 3)) * 255.0

        mask_3d = mask_to_3d(mask * 255)
        predict_mask_3d = mask_to_3d(predict_mask)
        image_vis = image * 255.0

        all_images = [image_vis, sep_line, mask_3d, sep_line, predict_mask_3d]
        result_img = np.concatenate(all_images, axis=1).astype(np.uint8)

        img_name = os.path.basename(img_path).split('.')[0]
        cv2.imwrite(f"{save_path}/{img_name}_segmentation.png", result_img)

    mask_save_dir = os.path.join(save_path, "predict_masks")
    os.makedirs(mask_save_dir, exist_ok=True)
    for i, img_path in enumerate(test_image_paths):
        image = parse_image(test_image_paths[i], image_size)
        predict_mask = model.predict(np.expand_dims(image, axis=0), verbose=0)[0]
        predict_mask = (predict_mask > threshold) * 255.0
        predict_mask = np.squeeze(predict_mask).astype(np.uint8)
        img_name = os.path.basename(img_path).split('.')[0]
        cv2.imwrite(f"{mask_save_dir}/{img_name}_mask.png", predict_mask)
