import numpy as np
import gala.evaluate as ev
from skimage.measure import label
import time, os, cv2

cwd = os.getcwd()


def calculate_dice(pred, mask):
   
    pred_bin = (pred == 255).astype(np.uint8)
    mask_bin = (mask == 255).astype(np.uint8)

    intersection = np.sum(pred_bin & mask_bin)
    sum_pred = np.sum(pred_bin)
    sum_mask = np.sum(mask_bin)

    if sum_pred + sum_mask == 0:
        return 1.0  # 均为背景时Dice为1
    return 2.0 * intersection / (sum_pred + sum_mask)


# 新增：计算mIoU
def calculate_miou(pred, mask):
  
    pred_bin = (pred == 255).astype(np.uint8)
    mask_bin = (mask == 255).astype(np.uint8)

    
    cm = confusion_matrix(
        mask_bin.flatten(),
        pred_bin.flatten(),
        labels=[0, 1]
    )

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


def get_figure_of_merit(pred, mask, const_index=0.1):
    num_pred = np.count_nonzero(pred[pred == 255])
    num_mask = np.count_nonzero(mask[mask == 255])
    num_max = num_pred if num_pred > num_mask else num_mask

    temp = 0.0
    for index_x in range(0, pred.shape[0]):
        for index_y in range(0, pred.shape[1]):
            if pred[index_x, index_y] == 255:
                distance = get_dis_from_mask_point(mask, index_x, index_y)
                temp = temp + 1 / (1 + const_index * pow(distance, 2))
    f_score = (1.0 / num_max) * temp
    return f_score


def get_dis_from_mask_point(mask, index_x, index_y, neighbor_length=60):
    """
    计算检测到的边缘点与离它最近边缘点的距离
    """
    if mask[index_x, index_y] == 255:
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
    x, y = np.where(mask[region_start_row: region_end_row, region_start_col: region_end_col] == 255)

    if len(x) == 0:
        min_distance = 30
    else:
        min_distance = np.amin(
            np.linalg.norm(np.array([x + region_start_row, y + region_start_col]) - np.array([[index_x], [index_y]]),
                           axis=0))

    return min_distance


def get_map_2018kdasb_new(pred, mask, target_image=0):
    """
    计算边缘检测值
    """
    thresholds = np.array([0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])
    tp = np.zeros(10)
    if target_image == 0:
        # Eliminate the impact of poor performance on the border of image
        pred[0, :] = 255
        pred[:, 0] = 255
        pred[-1, :] = 255
        pred[:, -1] = 255
        mask[0, :] = 255
        mask[:, 0] = 255
        mask[-1, :] = 255
        mask[:, -1] = 255
        # 预处理，为保证按照每个实例进行识别，需要label,所以统一前背景
        mask = 255 - mask
        pred = 255 - pred

    label_mask, num_mask = label(mask, background=0, return_num=True, connectivity=1)
    label_pred, num_pred = label(pred, background=0, return_num=True, connectivity=1)

    for i_pred in range(1, num_pred + 1):
        intersect_mask_labels = list(
            np.unique(label_mask[label_pred == i_pred]))  
        if 0 in intersect_mask_labels:
            intersect_mask_labels.remove(0)

        if len(
                intersect_mask_labels) == 0:  
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
    map_score = np.average(tp / (tp + fp + fn))
    return map_score


def eval_RI_VI(results_dir, outAd, gt_dir=os.path.join(cwd, 'datasets', 'segmentation', 'net_test', 'test', 'labels')):
    results_path = sorted(os.listdir(results_dir))
    print(results_path)

    mRI = 0
    mad_RI = 0
    m_merger_error = 0
    m_split_error = 0
    mVI = 0
    mIoU_sum = 0 
    dice_sum = 0 
    count = 0
    out = open(outAd, "a")  # record the result
    out.write(
        "RI, mRI, adjust_RI, m_adjust_RI, merger_error, m_merger_error, split_error, m_split_error, VI, mVI, IoU, mIoU, Dice, mDice" + '\n')
    out.close()
    for r in results_path:
        name = r.split(".")[0]
        gt_path = os.path.join(gt_dir, name + ".png")
        print(gt_path)
        if os.path.exists(gt_path):
            count += 1

            gt = cv2.imread(gt_path, 0)

            result = cv2.imread(os.path.join(results_dir, r), 0)
            result = result[10:-10, 10:-10]
            gt = gt[10:-10, 10:-10]

            result, num_result = label(result, background=255, return_num=True, connectivity=1)
            gt, num_gt = label(gt, background=255, return_num=True, connectivity=1)

            # false merges(缺失), false splits（划痕）
            merger_error, split_error = ev.split_vi(result, gt)
            VI = merger_error + split_error
            RI = ev.rand_index(result, gt)
            adjust_RI = ev.adj_rand_index(result, gt)

            iou = calculate_miou(result, gt)
            dice = calculate_dice(result, gt)
            mIoU_sum += iou
            dice_sum += dice
            
            m_merger_error += merger_error
            m_split_error += split_error
            mRI += RI
            mVI += VI
            mad_RI += adjust_RI

            out = open(outAd, "a")  # # record the result
            # "RI, mRI, adjust_RI, m_adjust_RI, merger_error, m_merger_error, split_error, m_split_error, VI, mVI"
            line = str(RI) + "," + str(mRI / count) + "," + str(adjust_RI) + "," + str(mad_RI / count) + "," + str(
                merger_error) + "," + str(m_merger_error / count) + "," + str(split_error) + "," + str(
                m_split_error / count) + "," + str(VI) + "," + str(mVI / count) + "," +str(iou)+ "," +str(mIoU_sum/count)+"," 
                +str(dice)+"," +str(dice_sum/count)+"\n"
            out.write(line)
            out.close()

    
    
    print("average RI : ", mRI / count, "average adRI : ", mad_RI / count, "average VI : ", mVI / count,
          "average merger_error : ", m_merger_error / count, "average split_error : ", m_split_error / count,
          "average mIoU: ",mIoU_sum / count, "average Dice:", dice_sum / count)


def eval_F_mapKaggle(results_dir, outAd,
                     gt_dir=os.path.join(cwd, 'datasets', 'segmentation', 'net_test', 'test', 'labels')):
   
    results_path = sorted(os.listdir(results_dir))
    F = 0
    mAP = 0
    mIoU_sum = 0  
    dice_sum = 0 
    count = 0

    out = open(outAd, "a")
    out.write("F,avF,mAP,avmAP,IoU,mIoU,Dice,mDice" + '\n')  # 修改文件头
    out.close()

    for r in results_path:
        name = r.split(".")[0]
        gt_path = os.path.join(gt_dir, name + ".png")
        if os.path.exists(gt_path):
            count += 1

            gt = cv2.imread(gt_path, 0)
            result = cv2.imread(os.path.join(results_dir, r), 0)
            result = result[10:-10, 10:-10]
            gt = gt[10:-10, 10:-10]

          
            F_test = get_figure_of_merit(result, gt)
            mAP_test = get_map_2018kdasb_new(result, gt)
        
            iou = calculate_miou(result, gt)
            dice = calculate_dice(result, gt)

           
            F += F_test
            mAP += mAP_test
            mIoU_sum += iou
            dice_sum += dice

            
            out = open(outAd, "a")
            line = f"{F_test},{F / count},{mAP_test},{mAP / count},{iou},{mIoU_sum / count},{dice},{dice_sum / count}\n"
            out.write(line)
            out.close()

    F = F / count
    mAP = mAP / count
    mIoU = mIoU_sum / count
    dice = dice_sum / count
    print("count {count}, average F: {F}, average mAP: {mAP}")
    print(f"average mIoU: {mIoU}, average Dice: {dice}")  # 新增打印
    print("count ", count, " average F : ", F, "average mAP : ", mAP,"average mIoU: ",mIoU, "average Dice:", dice)

if __name__ == '__main__':
    RI_save_dir = "./evaluation/big_RI_VI/"
    Map_save_dir = "./evaluation/big_F_mAP/"
    print("BPU_Net_model " + "#####"*20)
    eval_RI_VI("./result_total/BP_Unet_model/", RI_save_dir+"BPU_Net_model.txt")
    eval_F_mapKaggle("./result_total/BP_Unet_model/", Map_save_dir+"BPU_Net_model.txt")

