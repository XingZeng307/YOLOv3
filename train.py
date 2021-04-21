import numpy as np
import argparse
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from yolov3 import YOLO
from dataset import YOLODataset
from loss import YOLOloss
from utils import check_class_accuracy, mean_average_precision, load_weights_darknet53, save_checkpoint, load_checkpoint
from transform import train_transforms, test_transforms

'''
define some variables
'''
# here change the number of classes of own dataset
num_classes = 20
# anchors the original yolov3 used, I did normalization for later calculation
anchors = np.array([[116, 90], [156, 198], [373, 326], [30, 61], [62, 45], [59, 119],  [10, 13], [16, 30], [33, 23]],
                      np.float32) / 416


def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLO detection"
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--img-size", type=int, default=416)
    parser.add_argument("--iou-thre", type=float, default=0.4)
    parser.add_argument("--conf-thre", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epoches", type=int, default=100)

    parser.add_argument(
        "--train-img-dir", type=str, default="train/images"
    )
    parser.add_argument(
        "--val-img-dir", type=str, default="val/images"
    )
    parser.add_argument(
        "--train-label-dir", type=str, default="train/labels"
    )
    parser.add_argument(
        "--val-label-dir", type=str, default="val/labels"
    )
    parser.add_argument(
        "--train-annotation-file", type=str, default="train.csv"
    )
    parser.add_argument(
        "--val-annotation-file", type=str, default="val.csv"
    )

    parser.add_argument(
        "--load-weights",
        action="store_true",
        help="Whether to load pre_trained model"
    )
    return parser.parse_args()


# train for one batch
def train_loop(dataloader, model, YOLOloss, optimizer, device, scaled_anchors):
    loop = tqdm(dataloader, leave=True)
    losses = []
    for batch, (x, y) in enumerate(loop):
        x = x.float()
        x = x.to(device)

        # target (three scales)
        y0, y1, y2 = (
            y[0].to(device),
            y[1].to(device),
            y[2].to(device),
        )
        # forward
        pred = model(x)

        # calculate loss
        loss = YOLOloss(pred[0], y0, scaled_anchors[0:3]) + YOLOloss(pred[1], y1, scaled_anchors[3:6]) + YOLOloss(pred[2], y2, scaled_anchors[6:9])
        losses.append(loss.item())

        optimizer.zero_grad()
        # backward propagation
        loss.backward()
        # update parameters
        optimizer.step()

        # display loss
        mean_loss = sum(losses)/len(losses)
        loop.set_postfix(loss=mean_loss)


def main(args):
    # scale normalized anchors by their corresponding scales (e.g img_size = 416, feature maps will 32, 16, 8,
    # scaled_anchors =  normalized_anchors X (8, 16, 32))
    """
    scaled anchors
    tensor([[ 3.6250,  2.8125],
        [ 4.8750,  6.1875],
        [11.6562, 10.1875],
        [ 1.8750,  3.8125],
        [ 3.8750,  2.8125],
        [ 3.6875,  7.4375],
        [ 1.2500,  1.6250],
        [ 2.0000,  3.7500],
        [ 4.1250,  2.8750]])
    """
    di = torch.tensor([int(args.img_size/32), int(args.img_size/16), int(args.img_size/8)]).unsqueeze(1)
    scaled_anchors = (
            torch.tensor(anchors)*torch.repeat_interleave(di, torch.tensor([3, 3, 3]), dim=0).repeat(1, 2)
        ).to(args.device)

    train_dataset = YOLODataset(
        args.train_img_dir,
        args.train_label_dir,
        args.train_annotation_file,
        scaled_anchors,
        num_classes,
        args.iou_thre,
        args.img_size,
        transform=train_transforms
    )

    val_dataset = YOLODataset(
        args.val_img_dir,
        args.val_label_dir,
        args.val_annotation_file,
        scaled_anchors,
        num_classes,
        args.iou_thre,
        args.img_size,
        transform=test_transforms
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

    model = YOLO(num_classes).to(args.device)

    # if it is first time to train the model, use darknet53 pretrained weights
    # load_weights_darknet53("darknet53.conv.74", model)

    optimizer = optim.Adam(
            model.parameters(), lr=1e-5, weight_decay=1e-4
        )

    load_checkpoint(
            "checkpoint.pth.tar", model, optimizer, 1e-5
    )
    for epoch in range(args.epoches):

        train_loop(train_loader, model, YOLOloss, optimizer, args.device, scaled_anchors)
        print(f"Epoch {epoch+1}\n-------------------------------")
        save_checkpoint(model, optimizer, filename=f"checkpoint.pth.tar")
        model.eval()
        results = check_class_accuracy(model, val_loader, args.conf_thre)
        mAP = mean_average_precision(model, val_loader, scaled_anchors, num_classes, args.iou_thre)
        results.append(mAP)
        with open("result.txt", 'a') as f:
            f.write(str(results).strip("[").strip("]")+'\n')

        model.train()

    print("Done!")


if __name__ == "__main__":
    args = parse_args()
    main(args)


