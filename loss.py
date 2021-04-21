import torch
from torch import nn
from utils import rel_to_abs_box, intersection_over_union
'''
pred three scales
[(batch, grid, grid, boxes, 5 + num_classes),(batch, grid, grid, boxes, 5 + num_classes),(batch, grid, grid, boxes, 5 + num_classes)]

target three scales
[(batch, grid, grid, anchors, 5 + 1),((batch, grid, grid, anchors, 5 + 1),((batch, grid, grid, anchors, 5 + 1)]

'''


def YOLOloss(predictions, targets, anchors):

    """
    calculate loss for one scale
    :param predictions: pred shape for one scale (batch, scale, grid, grid, 3, 5 + num_classes)
    :param targets: target shape for one scale (batch, scale, grid, grid, 3, 5 + 1)
    :param anchors: scaled anchor size based on current scale
    :return: box_loss + object_loss + noobject_loss + class_loss
    """

    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()
    obj = targets[..., 4] == 1
    noobj = targets[..., 4] == 0
    entropy = nn.CrossEntropyLoss()

    # bounding box loss
    # use sig(tx), sig(ty) to calculate x,y center loss
    predictions[..., 0:2] = torch.sigmoid(predictions[..., 0:2])
    # sig(tx), sig(ty), tw, th --> calculate box loss
    box_loss = mse(predictions[..., 0:4][obj], targets[..., 0:4][obj])

    # object loss
    # absolute value --> calculate ious
    pred_abs_boxes = rel_to_abs_box(predictions[..., 0:4], anchors)
    target_abs_boxes = rel_to_abs_box(targets[..., 0:4], anchors)
    ious = intersection_over_union(pred_abs_boxes[obj], target_abs_boxes[obj])
    # ious = torch.flatten(ious)

    # YOLOv3 predicts an objectness score for each bounding box using logistic regression, so here use sigmoid function
    # and based on the iou bt prediction boxes and target boxes to calculate obj loss
    object_loss = mse(torch.sigmoid(predictions[..., 4:5][obj]), ious * targets[..., 4:5][obj])


    # noobj loss
    noobject_loss = bce(predictions[..., 4:5][noobj], targets[..., 4:5][noobj])

    # class_loss = bce(predictions[..., 5:][obj], targets[..., 5:][obj])
    class_loss = entropy(
            (predictions[..., 5:][obj]), (targets[..., 5][obj].long()),
        )

    # here the weights can be changed
    return 5* box_loss + 5*object_loss + 0.5*noobject_loss + 5*class_loss


