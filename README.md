# Enodo_robotics_CV_task

## Dataset and Model info

The cropped v6 dataset is downloaded through google drive
`https://drive.google.com/file/d/1-Crh7wct3OWZ_4lDo8Yl-_gdAYS-et7_/view?usp=drive_link`

The model data is downloaded through google drive
`https://drive.google.com/file/d/1NvL2df2DvKJsuoxPRGUke1HvNVwrRTCI/view?usp=sharing`


The trained model data is downloaded through google drive
`https://drive.google.com/file/d/1BVYEyqhSSj-hl5oggokWKwP9A37b_7_f/view?usp=sharing`

## Testing result
### First round of training (the winner: last_epoch_weights_400swinv2.pth)

| **Model**                      | **Epochs** | **mIoU** | **mPA** | **Accuracy** | **Copper IoU** | **Meatball IoU** | **Trash IoU** |
|--------------------------------|------------|----------|---------|--------------|----------------|------------------|---------------|
| MobileNet (even weight)        | 50         | 48.69%   | 59.87%  | 99.17%       | 35.82%         | 27.51%           | 32.18%        |
| MobileNet (even weight)        | 200        | 60.91%   | 69.73%  | 99.49%       | 49.44%         | 44.79%           | 49.88%        |
| MobileNet (cls weight modified)| 200        | 61.27%   | 71.74%  | 99.45%       | 51.86%         | 45.18%           | 48.57%        |
| Swinv2T                        | 200        | 51.5%    | 73.02%  | 98.99%       | 40.22%         | 32.67%           | 34.05%        |
| Swinv2T                        | 400        | 62.9%    | 76.75%  | 99.43%       | 49.64%         | 51.64%           | 50.87%        |


### Object-level Recall and Precision for Each Class

| **Class**  | **Recall (%)** | **Precision (%)** |
|------------|----------------|-------------------|
| Copper     | 51.82          | 48.97             |
| Meatball   | 53.21          | 55.24             |
| Trash      | 44.44          | 50.00             |

### Comparison of Performance Metrics for Different Test Conditions

| **Test Condition**                                | **mIoU (%)** | **mPA (%)** | **Accuracy (%)** | **Copper IoU (%)** | **Meatball IoU (%)** | **Trash IoU (%)** |
|--------------------------------------------------|--------------|-------------|------------------|--------------------|----------------------|-------------------|
| Only Testing Dataset (No Cropping)               | 39.77        | 45.4        | 98.45            | 18.62              | 28.06                | 13.82             |
| Only Testing Dataset (After Cropping in 512*512) | 22.77        | 35.91       | 81.55            | 4.78               | 3.2                  | 1.51              |
| Testing Dataset + Validation Dataset             | 47.14        | 68.38       | 84.91            | 43.55              | 39.67                | 20.37             |


