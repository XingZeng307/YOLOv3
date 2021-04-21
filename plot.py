import numpy as np
import cv2

import torch
import torch.optim as optim

from yolov3 import YOLO
from utils import non_max_suppression, load_checkpoint
from transform import test_transforms


device = "cpu"
anchors = np.array([[116, 90], [156, 198], [373, 326], [30, 61], [62, 45], [59, 119],  [10, 13], [16, 30], [33, 23]],
                      np.float32) / 416
img_size = 416
di =  torch.tensor([int(img_size/32), int(img_size/16), int(img_size/8)]).unsqueeze(1)


scaled_anchors = (
        torch.tensor(anchors)*torch.repeat_interleave(di, torch.tensor([3, 3, 3]), dim=0).repeat(1,2)

    ).to(device)


def plot_one_image(img_path, class_names, weights_file, iou_thre, con_thre):
    image = cv2.imread(img_path)
    bboxes = []
    augmentations = test_transforms(image=image, bboxes=bboxes)
    image = augmentations["image"]
    img = image
    image = image.reshape(1, image.shape[0],image.shape[1], image.shape[2])
    model = YOLO(len(class_names))
    optimizer = optim.Adam(
        model.parameters(), lr=1e-5, weight_decay=1e-4
    )
    load_checkpoint(
                weights_file, model, optimizer, 1e-5
    )

    pred_bboxes = []
    with torch.no_grad():
        out = model(image)
        for i in range(3):
            scale = torch.zeros((out[i].shape[0], out[i].shape[1], out[i].shape[2],out[i].shape[3], 1))
            # here scale used to cache the scale where the box is in
            pred_bboxes.append(torch.cat((out[i], scale), -1))


    boxes = non_max_suppression(pred_bboxes, scaled_anchors, con_thre, iou_thre)

    # print(names[torch.argmax(torch.sigmoid(boxes[0][..., 5:]))])
    x = boxes[0][0:4]

    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[0] = x[0] - x[2] / 2  # top left x
    y[1] = x[1] - x[3] / 2  # top left y
    y[2] = x[0] + x[2] / 2  # bottom right x
    y[3] = x[1] + x[3] / 2  # bottom right y

    S = [32, 16, 8]

    image = img.permute(1, 2, 0)

    image = image.cpu().float().numpy()
    i = int(boxes[0][5].item())
    cv2.rectangle(image, (int(y[0].item()*S[i]), int(y[1].item()*S[i])), (int(y[2].item()*S[i]), int(y[3].item()*S[i])), (0, 0, 255), 2)

    label = class_names[torch.argmax(torch.sigmoid(boxes[0][..., 5:]))]
    tl = 3  # line thickness
    tf = max(tl - 1, 1)  # font thickness
    t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
    cv2.putText(image, label, (int(y[2].item()*S[i]+5), int(y[3].item()*S[i]+5)), 0, tl / 3, [220, 220, 220], thickness=tf, lineType=cv2.LINE_AA)
    cv2.imshow("fff", image)
    cv2.waitKey(0)


if __name__ == "__main__":
    img_path = "2f957019bebbc64d0bbddba0809b9cb4.JPEG"
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
    weights_file = "checkpoint.pth.tar"
    iou_thre = 0.3
    con_thre = 0.7
    plot_one_image(img_path, PASCAL_CLASSES, weights_file, iou_thre, con_thre)