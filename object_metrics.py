import numpy as np
from PIL import Image
from os.path import join
from post_process import create_separate_masks, calculate_object_confusion_matrix
import os
import time
from tqdm import tqdm

### Hyper-parameters: iou_threshold, object_threshold
### Get the prediction and the label array
start_time_all = time.time()
num_classes = 4
print('Num classes', num_classes)

# img = "img/valid/07_08_52-533203_jpg.rf.fd06edefa74b4de400beac2607e6ddb6.jpg"
# img_label = "img/valid_mask/07_08_52-533203_jpg.rf.fd06edefa74b4de400beac2607e6ddb6.png"
# name_classes = ["background", "copper", "meatball", "trash"]
# deeplab = DeeplabV3()
# pr_img, blend_image = deeplab.get_miou_png(Image.open(img))
# pred = np.array(pr_img)

# # print(pred)
#
# # ------------------------------------------------#
# #   读取一张对应的标签，转化成numpy数组
# # ------------------------------------------------#
# label = np.array(Image.open(img_label))
# # print(label)
#
# ground_truth_masks = create_separate_masks(pred, object_threshold=1000)
# predicted_masks = create_separate_masks(label, object_threshold=1000)
# num_gt = ground_truth_masks.shape[-1]
# num_pred = predicted_masks.shape[-1]
# print(ground_truth_masks.shape, predicted_masks.shape)
#
# classes = [1, 2, 3]
# tp, fp, fn, conf_matrix, precision_per_class, recall_per_classconf_matrix, precision_per_class, recall_per_class = calculate_object_confusion_matrix(ground_truth_masks, predicted_masks, classes, iou_threshold=0.5)
# print("Confusion Matrix:")
# print(conf_matrix)
# print("Precision per class:", precision_per_class)
# print("Recall per class:", recall_per_class)


# -----------------------------------------------------------------#
#   在所有验证集图像上求所有类别的object-recall precision值
# -----------------------------------------------------------------#

# Iron_path = 'Iron_dataset'
# image_ids = open(os.path.join(Iron_path, "img/ImageSets/val.txt"), 'r').read().splitlines()
# gt_dir = os.path.join(Iron_path, "img/ImageSets")
# # miou_out_path = "miou_out"
# # pred_dir = os.path.join(miou_out_path, 'detection-results')
# #
# gt_imgs = [join(gt_dir, x + ".png") for x in image_ids]  # benchmark mask(label)
# pred_imgs = [join(pred_dir, x + ".png") for x in image_ids]  # predicted mask


name_classes    = ["background", "copper","meatball","trash"]
# Initialize cumulative counters for each class
cumulative_tp = {}
cumulative_fp = {}
cumulative_fn = {}

# List of all classes (e.g., [1, 2, 3] for three classes)
classes = [1, 2, 3]
for cls in classes:
    cumulative_tp[cls] = 0
    cumulative_fp[cls] = 0
    cumulative_fn[cls] = 0


Iron_path = 'img'
image_ids = open(os.path.join(Iron_path, "ImageSets/val.txt"), 'r').read().splitlines()
gt_images = [join("img/valid", x + ".jpg") for x in image_ids]  # (images)
gt_labels = [join("img/valid_mask", x + ".png") for x in image_ids]  # benchmark mask(label)
pre_labels = [join("img/prediction_masks", x + ".png") for x in image_ids]  # predicted mask


Iron_path = 'Iron_dataset'
image_ids = open(os.path.join(Iron_path, "Iron_material/ImageSets/Segmentation/val.txt"), 'r').read().splitlines()
gt_images = [join("Iron_dataset/Iron_material/JPEGImages", x + ".jpg") for x in image_ids]  # (images)
gt_labels = [join("Iron_dataset/Iron_material/SegmentationClass", x + ".png") for x in image_ids]  # benchmark mask(label)
pre_labels = [join("Iron_dataset/Iron_material/prediction-results", x + ".png") for x in image_ids]  # predicted mask


for ind in tqdm(range(len(image_ids))):

    # ------------------------------------------------#
    #   读取每一个（图片-标签）对
    # ------------------------------------------------#
    label = np.array(Image.open(gt_labels[ind]))
    pred = np.array(Image.open(pre_labels[ind]))

    # 如果图像分割结果与标签的大小不一样，这张图片就不计算
    if len(label.flatten()) != len(pred.flatten()):
        print(
            'Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(
                len(label.flatten()), len(pred.flatten()), gt_images[ind],
                pred))
        continue


    # print('-'* 30)
    # print(f'{ind+1}th ground truth image')
    ground_truth_masks = create_separate_masks(label, object_threshold=0)
    # print(f'{ind + 1}th prediction image')
    predicted_masks = create_separate_masks(pred, object_threshold=0)
    tp, fp, fn, conf_matrix, precision_per_class, recall_per_class = calculate_object_confusion_matrix(ground_truth_masks,
                                                                                           predicted_masks, classes,
                                                                                           iou_threshold=0.5)
    # Store the result
    # Open the file in append mode
    with open("object_metric_result.txt", "a") as file:
        # Write the results to the file
        file.write(f"Image pair {ind + 1}: TP={tp}, FP={fp}, FN={fn}\n")

    # Update cumulative counts
    for cls in classes:
        cumulative_tp[cls] += tp.get(cls, 0)
        cumulative_fp[cls] += fp.get(cls, 0)
        cumulative_fn[cls] += fn.get(cls, 0)
# ------------------------------------------------#
#   计算所有验证集图片的逐类别precision, recall值
# ------------------------------------------------#
print('cumulative_tp', cumulative_tp)
print('cumulative_fp', cumulative_fp)
print('cumulative_fn', cumulative_fn)
Precision_object = {cls: cumulative_tp[cls] / (cumulative_tp[cls] + cumulative_fp[cls]) if (cumulative_tp[cls] + cumulative_fp[cls]) > 0 else 0 for cls in classes}
Recall_object = {cls: cumulative_tp[cls] / (cumulative_tp[cls] + cumulative_fn[cls]) if (cumulative_tp[cls] + cumulative_fn[cls]) > 0 else 0 for cls in classes}


# ------------------------------------------------#
#   逐类别输出一下precision, recall值
# ------------------------------------------------#
if name_classes is not None:
    for ind_class in classes:
        print('===>' + name_classes[ind_class] + '; Recall' + str(round(Recall_object[ind_class] * 100, 2)) + '; Precision-' + str(
            round(Precision_object[ind_class] * 100, 2)))

end_time_all = time.time()
execution_time = end_time_all - start_time_all
print("All execution time:", execution_time)