import numpy as np
from tqdm import tqdm
import cv2

import torch
from torch.utils.data import  DataLoader
import torch.optim as optim

from yolov3 import YOLO
from dataset import YOLODataset
from utils import non_max_suppression, load_checkpoint
from transform import test_transforms


"""
test different functions: class accuracy, mean_average_precision, non_maximum_suppression, plot bounding boxes
"""

anchors = np.array([[116, 90], [156, 198], [373, 326], [30, 61], [62, 45], [59, 119],  [10, 13], [16, 30], [33, 23]],
                      np.float32) / 416
img_size = 416
di = torch.tensor([int(img_size/32), int(img_size/16), int(img_size/8)]).unsqueeze(1)
device = "cpu"

scaled_anchors = (
        torch.tensor(anchors)*torch.repeat_interleave(di, torch.tensor([3, 3, 3]), dim=0).repeat(1,2)

    ).to(device)


train_img_dir= "PASCAL_VOC/images"
val_img_dir= "PASCAL_VOC/images"

train_label_dir= "PASCAL_VOC/labels"
val_label_dir= "PASCAL_VOC/labels"

train_annotation_file= "PASCAL_VOC/train.csv"
val_annotation_file= "PASCAL_VOC/1examples.csv"

val_dataset = YOLODataset(
        val_img_dir,
        val_label_dir,
        val_annotation_file,
        scaled_anchors,
        20,
        0.45,
        416,
        test_transforms
    )

val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, drop_last=False)


model = YOLO(20).to(device)

optimizer = optim.Adam(
            model.parameters(), lr=1e-5, weight_decay=1e-4
        )
load_checkpoint(
            "checkpoint.pth.tar", model, optimizer, 1e-5
)


'''
class accuracy
'''
# for batch, (x, y) in enumerate(val_loader):
#     with torch.no_grad():
#         x = x.to(device)
#         out = model(x)
#     # target
#         # forward
#     for i in range(3):
#         y[i] = y[i].to(config.DEVICE)
#         obj = y[i][..., 4] == 1 # in paper this is Iobj_i
#         noobj = y[i][..., 4] == 0  # in paper this is Iobj_i
#         correct_class += torch.sum(
#             torch.argmax(out[i][..., 5:][obj], dim=-1) == y[i][..., 5][obj]
#         )
#         print(y[i][obj].shape)
#         print(out[i][obj].shape)
#         tot_class_preds += torch.sum(obj)
#         # print(torch.argmax(y[i][..., 5:][obj], dim=-1))
#         print( y[i][..., 5][obj])
#         print("@@@@@@@@@@@@@@@@@@@")
#         print(torch.argmax(out[i][..., 5:][obj], dim=-1))
#
#         obj_preds = torch.sigmoid(out[i][..., 4]) > 0.2
#
#         correct_obj += torch.sum(obj_preds[obj] == y[i][..., 4][obj])
#         tot_obj += torch.sum(obj)
#         correct_noobj += torch.sum(obj_preds[noobj] == y[i][..., 4][noobj])
#         tot_noobj += torch.sum(noobj)
#
# print(f"Class accuracy is: {(correct_class/(tot_class_preds+1e-16))*100:2f}%")
# print(f"No obj accuracy is: {(correct_noobj/(tot_noobj+1e-16))*100:2f}%")
# print(f"Obj accuracy is: {(correct_obj/(tot_obj+1e-16))*100:2f}%")

'''
mean_average_precision
'''

# map = mean_average_precision(model, val_loader, scaled_anchors, 20, 0.4, 0.2)# for idx, (x, y) in enumerate(tqdm(test_loadecheck_class_accuracyr)):


'''
non_maximum_suppression
'''
#
pred_bboxes = []

PASCAL_CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor"
]


for idx, (x, y) in enumerate(tqdm(val_loader)):
    x = x.float()
    x = x.to(device)
    with torch.no_grad():
        out = model(x)
        for i in range(3):
            pred_bboxes.append(out[i])

    boxes = non_max_suppression(pred_bboxes, scaled_anchors, conf_thres=0.4, iou_thres=0.2)
    print(len(boxes))

    print(PASCAL_CLASSES[torch.argmax(torch.sigmoid(boxes[0][..., 5:]))])


    '''
    plot
    '''

    # xywh = boxes[0][0:4]
    # y = xywh.clone() if isinstance(xywh, torch.Tensor) else np.copy(xywh)
    # y[0] = xywh[0] - xywh[2] / 2  # top left x
    # y[1] = xywh[1] - xywh[3] / 2  # top left y
    # y[2] = xywh[0] + xywh[2] / 2  # bottom right x
    # y[3] = xywh[1] + xywh[3] / 2  # bottom right y
    #
    # S = [32, 16, 8]
    # #
    #
    # image = x.reshape(x.shape[1], x.shape[2], x.shape[3]).permute(1, 2, 0)
    #
    # image = image.cpu().float().numpy()
    # image_ = image.copy()
    #
    # i = int(boxes[0][5].item())
    #
    # cv2.rectangle(image_, (int(y[0].item()*S[i]), int(y[1].item()*S[i])), (int(y[2].item()*S[i]), int(y[3].item()*S[i])), (0, 0, 255), 2)
    #
    # label = PASCAL_CLASSES[torch.argmax(torch.sigmoid(boxes[0][..., 5:]))]
    # tl = 3  # line thickness
    # tf = max(tl - 1, 1)  # font thickness
    # t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
    # cv2.putText(image_, label, (int(y[0].item()*S[i]+5), int(y[1].item()*S[i]+5)), 0, tl / 3, [220, 220, 220], thickness=tf, lineType=cv2.LINE_AA)
    # cv2.imshow("fff", image_)
    # cv2.waitKey(0)


