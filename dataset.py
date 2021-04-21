import os
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
from utils import cal_anchors_iou


"""
preprocess  the class and bounding boxes annotations into same format as prediction in order to calculate loss.
(batch, grid, grid, anchor, (centroid x, centroid y, width, height, objectness, class))
(Assign one anchor to each target in each_scale)


y_pred: Prediction tensor from the model output, in the shape of 
(batch, grid, grid, boxes, 5 + num_classes)
e.g. torch.Size([2, 13, 13, 3, 5+num_classes], [2, 26, 26, 3, 5+num_classes], [2, 52, 52, 3, 5+num_classes])


y_label: annotations are [class, x, y, w, h] ---> 
[[batch, grid, grid, achors, 5+1] for gird in [13, 26, 52]], 
here 5+1 represents:  [sigmoid(tx), sigmoid(ty), tw, th, obj_confidence, class_label]

"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class YOLODataset(Dataset):
    def __init__(self, img_dir, label_dir, annotation_file, anchors, num_classes, iou_threshold, img_size, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_labels = pd.read_csv(annotation_file)
        self.transform = transform
        self.target_transform = target_transform
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.anchors = anchors
        self.img_size = img_size
        self.S = [int(img_size/32), int(img_size/16), int(img_size/8)]
        self.iou_threshold = iou_threshold

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        # initialize the target in desired format in three scales
        target = [torch.zeros((S, S, 3, 6)) for S in self.S]  # e.g. 13, 26, 52
        # target = [torch.zeros((S, S, 3, 5+self.num_classes)) for S in self.S]  # e.g. 13, 26, 52
        # img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 1])
        # label_path = os.path.join(self.label_dir, self.img_labels.iloc[idx, 2])
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        label_path = os.path.join(self.label_dir, self.img_labels.iloc[idx, 1])
        # label = np.loadtxt(fname=label_path, delimiter=" ", ndmin=2).tolist()
        # [x y w h class]
        bboxes = np.roll(np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), 4, axis=1).tolist()
        image = np.array(Image.open(img_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=image, bboxes=bboxes)
            image = augmentations["image"]
            bboxes = augmentations["bboxes"]
        # p = transforms.Compose([transforms.Resize((self.img_size, self.img_size))])
        # image = p(image)

        # find best anchor for each ground truth box in each scale
        # here we also decide positive & negative samples,  YOLOv3 predicts an objectness score for each bounding box.
        # This should be 1 if the anchor overlaps a ground truth object by more than any other anchors.
        # If the bounding box prior is not the best but does overlap a ground truth object by more than some threshold
        # we ignore the prediction
        for box in bboxes:
            class_label = box[4]
            box = box[0:4]
            anchor_iou_sorted = cal_anchors_iou(box) # [{'anchor_idx': 6, 'iou': 0.79}, {'anchor_idx': 5, 'iou': 0.52}...]
            # make sure each scale has one anchor
            scales_has_anchor = [False]*3

            for each_anchor_iou in anchor_iou_sorted:
                # anchor_scale0 (0,1,2) --> large, anchor_scale1 (3,4,5) --> medium, anchor_scale2 (6,7,8)--> small object
                # corresponding feature map is [13, 26, 52]
                anchor_idx = each_anchor_iou['anchor_idx']
                # there are three scales for anchors, check the current anchor in which scale
                anchor_scale = anchor_idx//3
                # if anchor in scale 2 --> anchor index is in [6,7,8]--> converted it into [0,1,2]
                anchor_index_on_scale = anchor_idx % 3
                # calculate the bbox size in current scale (in dataset, it's normalized)
                bx, by = box[0] * self.S[anchor_scale], box[1] * self.S[anchor_scale]
                bw, bh = box[2] * self.S[anchor_scale], box[3] * self.S[anchor_scale]
                # calculate offset/ coordinates in current scale, in paper, it's called Cx, Cy
                cx = int(by)
                cy = int(bx)
                # anchor width and height, which already normalized by image size
                (pw, ph) = self.anchors[anchor_idx]
                # calculate the box width and height offset relative to anchor box by inverting the equations in the paper
                tw, th = torch.log(bw/pw + 1e-16), torch.log(bh/ph + 1e-16)
                # calculate the box x and y offset relative to grid cell, by inverting the equations in the paper
                # but here for simplicity, not invert sigmoid function
                sig_tx, sig_ty = bx-int(bx), by-int(by)
                # class
                tc = int(class_label)
                box_coordinates = torch.tensor(
                    [sig_tx, sig_ty, tw, th]
                )
                # only one anchor for one target in each scale
                anchor_taken = target[anchor_scale][cx, cy, anchor_index_on_scale, 4]
                if not scales_has_anchor[anchor_scale] and not anchor_taken:
                    target[anchor_scale][cx, cy, anchor_index_on_scale, 0:4] = box_coordinates
                    # objectness score should be 1 if anchor overlaps a ground truth object by more than any other anchors
                    target[anchor_scale][cx, cy, anchor_index_on_scale, 4] = 1
                    # convert class into one-hot encoding class
                    # target[anchor_scale][cx, cy, anchor_index_on_scale, 5:] = torch.nn.functional.one_hot(torch.tensor(tc), self.num_classes)
                    target[anchor_scale][cx, cy, anchor_index_on_scale, 5] = tc
                    scales_has_anchor[anchor_scale] = True

                elif not anchor_taken and each_anchor_iou['iou'] > self.iou_threshold:
                    target[anchor_scale][cx, cy, anchor_index_on_scale, 4] = -1  # igore
        return image, target


if __name__ == "__main__":
    # S = [52, 26, 13]
    pass
    anchors = np.array([[116, 90], [156, 198], [373, 326], [30, 61], [62, 45], [59, 119],  [10, 13], [16, 30], [33, 23]],
                      np.float32) / 416
    img_size = 256
    di = torch.tensor([int(img_size/32), int(img_size/16), int(img_size/8)]).unsqueeze(1)
    scaled_anchors = (
            torch.tensor(anchors)*torch.repeat_interleave(di, torch.tensor([3, 3, 3]), dim=0).repeat(1, 2)
        ).to(device)

    dataset = YOLODataset(
        "PASCAL_VOC/images/",
        "PASCAL_VOC/labels/",
        "PASCAL_VOC/test.csv",
        scaled_anchors,
        20,
        0.4,
        416,
    )
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    x, y = next(iter(dataloader))
    print(y[2].shape)
    #
    #


