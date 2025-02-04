from PIL import Image
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import label as scipy_label  # Import with an alias to avoid conflict
from sklearn.metrics import confusion_matrix
import time

name_classes    = ["background", "copper","meatball","trash"]
# Extract the predicted segmentation masks
img = "img/valid/07_05_38-081171_jpg.rf.6cde9f06af395589815fa8fb5db64b65.jpg"

pred = np.array(Image.open("img/prediction_masks/07_05_38-081171_jpg.rf.6cde9f06af395589815fa8fb5db64b65.png"))
label = np.array(Image.open("img/valid_mask/07_05_38-081171_jpg.rf.6cde9f06af395589815fa8fb5db64b65.png"))


# def create_separate_masks(matrix, object_threshold=1000):
#     '''
#     This is the function to count the number of objects of each class in the predicted segmentation result
#     and create a 3D masks.
#     '''
#     def separate_connected_objects(matrix):
#         # Helper function to check if indices are within bounds
#         def isValidIndex(row_idx, col_idx):
#             return 0 <= row_idx < row_num and 0 <= col_idx < col_num
#
#         # Iterative DFS to avoid recursion limit issues
#         def iterative_dfs(start_r, start_c, class_id, object_mask):
#             stack = [(start_r, start_c)]
#             visited.add((start_r, start_c))
#             area = 0
#
#             while stack:
#                 r, c = stack.pop()
#                 area += 1  # Each visited cell adds to the area
#                 object_mask[r][c] = class_id  # Mark this cell in the object's mask
#                 # Check four possible directions (up, down, left, right)
#                 for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
#                     new_r, new_c = r + dr, c + dc
#                     if isValidIndex(new_r, new_c) and (new_r, new_c) not in visited:
#                         if matrix[new_r][new_c] == class_id:
#                             visited.add((new_r, new_c))
#                             stack.append((new_r, new_c))
#             return area
#
#         row_num = len(matrix)
#         col_num = len(matrix[0])
#         visited = set()
#         classes_id = [1, 2, 3]  # Define the valid class IDs
#         masks = []
#
#         # Loop through each element in the matrix
#         for r in range(row_num):
#             for c in range(col_num):
#                 # If the cell belongs to a class and hasn't been visited
#                 if matrix[r][c] in classes_id and (r, c) not in visited:
#                     class_id = matrix[r][c]
#                     # Create a JPEGImages mask for this object
#                     object_mask = np.zeros((row_num, col_num), dtype=int)
#                     # Perform DFS and calculate the area of the connected component
#                     area = iterative_dfs(r, c, class_id, object_mask)
#                     if area <= object_threshold:
#                         # print("skip this ", area)
#                         continue  # Skip small objects
#                     else:
#                         # print(area, 'class_id:', class_id)
#                         masks.append(object_mask)  # Save the mask for this object
#
#         # Stack the masks to create a 3D matrix
#         if masks:
#             masks_3d = np.stack(masks, axis=-1)
#         else:
#             masks_3d = np.zeros((row_num, col_num, 0))  # In case there are no objects
#
#         return masks_3d  # Return the 3D matrix of masks
#
#     # Call the inner function and return its result
#     return separate_connected_objects(matrix)
def create_separate_masks(matrix, object_threshold=1000):
    classes_id = [1, 2, 3]
    masks = []

    for class_id in classes_id:
        class_mask = (matrix == class_id).astype(int)

        labeled_array, num_features = scipy_label(class_mask)

        for component in range(1, num_features + 1):
            object_mask = (labeled_array == component).astype(int) * class_id
            area = np.sum(object_mask > 0)

            if area > object_threshold:
                masks.append(object_mask)

    if masks:
        masks_3d = np.stack(masks, axis=-1)
    else:
        row_num, col_num = matrix.shape
        masks_3d = np.zeros((row_num, col_num, 0), dtype=int)

    return masks_3d

# matrix = [[0, 1, 0, 3, 3, 3],
#           [0, 1, 0, 0, 1, 1],
#           [1, 1, 0, 0, 3, 0],
#           [1, 0, 0, 2, 0, 0],
#           [0, 0, 0, 2, 2, 2],
#           [0, 0, 2, 2, 2, 2]]
#
# label =  [[0, 1, 0, 3, 3, 3],
#           [0, 1, 0, 0, 1, 1],
#           [1, 1, 1, 0, 3, 0],
#           [1, 0, 0, 2, 0, 0],
#           [0, 0, 0, 2, 2, 0],
#           [0, 0, 2, 2, 2, 2]]


#------------------------------------------------------------------------------------------------#


def calculate_object_confusion_matrix(ground_truth_masks, predicted_masks, classes, iou_threshold):
    """
    计算按物体分离后的混淆矩阵，使用匈牙利算法进行匹配

    Parameters:
    ground_truth_masks: list of np.ndarray
        真实标签的物体 masks (每个mask是一个独立的物体)
    predicted_masks: list of np.ndarray
        预测标签的物体 masks (每个mask是一个独立的物体)
    classes: list
        类别列表 (例如 [0, 1, 2, 3] 对应类别)

    Returns:
    """

    def iou(mask1, mask2):
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        return intersection / union if union != 0 else 0

    num_gt = ground_truth_masks.shape[-1]
    num_pred = predicted_masks.shape[-1]
    # print('number of objects in gt and pred', num_gt, num_pred)

    # -------------------------------------------------------------------------------#
    # # 计算每个真实物体和预测物体之间的 IoU
    # 创建 IoU 成本矩阵，大小为 [num_gt, num_pred]
    # cost_matrix = np.zeros((num_gt, num_pred))
    # for i in range(num_pred):
    #     pred_mask = predicted_masks[:, :, i]
    #     pred_class = pred
    #     gt_mask = ground_truth_masks[:, :, i]  # Extract 6x6 mask for the i-th ground truth object
    #     for j in range(num_pred):
    #         pred_mask = predicted_masks[:, :, j]
    #         cost_matrix[i, j] = -iou(gt_mask, pred_mask)
    # print('cost matrix', cost_matrix)
    # # 使用匈牙利算法来找到最优匹配
    # row_ind, col_ind = linear_sum_assignment(cost_matrix)
    # print(row_ind, col_ind)
    # # 遍历匹配的物体，计算混淆矩阵
    # for i, j in zip(row_ind, col_ind):
    #     if -cost_matrix[i, j] > thresold_IoU:  # threshold_IoU match
    #     # ith object in ground truth matches jth object in prediction
    #     # 计算混淆矩阵
    # -------------------------------------------------------------------------------#
    # Step 1: Iterate over predicted masks to find best ground truth match

    ## Track TP, FP, FN counts for each class
    classes_id = [1, 2, 3]  # Define the valid class IDs
    tp = {cls: 0 for cls in classes_id}
    fp = {cls: 0 for cls in classes_id}
    fn = {cls: 0 for cls in classes_id}
    cost_matrix = np.zeros((num_gt, num_pred))
    matched_gt = set()  # Keep track of which ground truth masks have been matched
    y_true = []
    y_pred = []


    # Step 1: Iterate over predicted masks to find best ground truth match
    for i in range(num_pred):
        pred_mask = predicted_masks[:, :, i]
        pred_class = np.unique(pred_mask[pred_mask != 0]).item()

        best_iou = 0
        best_match = -1

        for j in range(num_gt):
            gt_mask = ground_truth_masks[:, :, j]
            gt_class = np.unique(gt_mask[gt_mask != 0]).item()

            # Compute IoU between predicted and ground truth mask
            iou_val = iou(pred_mask, gt_mask)

            if iou_val > best_iou and gt_class == pred_class:
                best_iou = iou_val
                best_match = j

        # Step 2: Classify the prediction based on IoU threshold
        if best_iou >= iou_threshold:
            # True positive: correct prediction
            tp[pred_class] += 1
            matched_gt.add(best_match)
            y_true.append(pred_class)
            y_pred.append(pred_class)  # Correctly classified
        else:
            # False positive: no matching ground truth mask or IoU < threshold
            fp[pred_class] += 1
            y_true.append(0)  # Class 0: No match (background/incorrect)
            y_pred.append(pred_class)  # Incorrectly predicted class

    # Step 3: Check for ground truth masks that were not matched to any prediction (False Negative)
    for i in range(num_gt):
        if i not in matched_gt:
            gt_mask = ground_truth_masks[:, :, i]
            gt_class = np.unique(gt_mask[gt_mask != 0]).item()
            fn[gt_class] += 1  # pixel-wise
            y_true.append(gt_class)  # Ground truth class
            y_pred.append(0)  # No predicted match (class 0 for background)

    # Step 4: Compute confusion matrix, precision, and recall per class
    if len(set(y_true).intersection(classes)) == 0:
        # print("Warning: No ground truth labels match the specified classes. Skipping confusion matrix calculation.")
        conf_matrix = None
    else:
        conf_matrix = confusion_matrix(y_true, y_pred, labels=classes)

    precision_per_class = {cls: tp[cls] / (tp[cls] + fp[cls]) if (tp[cls] + fp[cls]) > 0 else 0 for cls in classes}
    recall_per_class = {cls: tp[cls] / (tp[cls] + fn[cls]) if (tp[cls] + fn[cls]) > 0 else 0 for cls in classes}

    return tp, fp, fn, conf_matrix, precision_per_class, recall_per_class



# masks_3d = create_separate_masks(matrix, threshold=1)

# 示例调用
ground_truth_masks = create_separate_masks(label, object_threshold=1000)
print(ground_truth_masks.shape)
# print(ground_truth_masks[:, :, 0])
predicted_masks = create_separate_masks(pred, object_threshold=1000)
print(predicted_masks.shape)
# print(predicted_masks[:, :, 0])

num_gt = ground_truth_masks.shape[-1]
num_pred = predicted_masks.shape[-1]
# print('ground truth objects:', num_gt, 'prediction objects:', num_pred)

classes = [1, 2, 3]
tp, fp, fn, conf_matrix, precision_per_class, recall_per_class = calculate_object_confusion_matrix(ground_truth_masks, predicted_masks, classes, iou_threshold=0.5)
print("tp, fp, fn:", tp, fp, fn)
print("Confusion Matrix:")
print(conf_matrix)
print("Precision per class:", precision_per_class)
print("Recall per class:", recall_per_class)

