import numpy as np
from tqdm import tqdm
from yolov3 import *

anchors = np.array([[116, 90], [156, 198], [373, 326], [30, 61], [62, 45], [59, 119],  [10, 13], [16, 30], [33, 23]],
                      np.float32) / 416

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# used for target dataset preprocessing
def cal_anchors_iou(box):
    """
    use absolute value to calcualte ious!!!
    calculate ious between the target box and all the anchors, in order to assign one best iou anchor to target
    :param box: one target box [bx, by, bw, bh] normalized coordinates (/image_siz)
    :return: sorted anchor_index and corresponding iou given target box
    """
    num_anchors = len(anchors)

    # iou for one anchor
    def calculate_iou(anchor_idx):
        anchor = anchors[anchor_idx]
        # we assume anchor has same center with target box!!!
        intersection = min(box[2], anchor[0]) * min(box[3], anchor[1])
        box_area = box[2] * box[3]
        anchor_area = anchor[0] * anchor[1]
        union = box_area + anchor_area - intersection
        iou = intersection/union
        return {"anchor_idx": anchor_idx, "iou": iou}
    if len(box) == 4:
        # ious for all anchors
        ious = list(map(calculate_iou, range(num_anchors)))
        anchor_iou_sorted = sorted(ious, key=lambda item: item["iou"], reverse=True)
        return anchor_iou_sorted

    else:
        return []


def rel_to_abs_box(rel_boxs, scaled_anchors):
    """
    convert relative box coordinates to  absolute box coordinates
    :param rel_boxs: [batch, grid, grid, 3, 4] here 4 -> (sigmoid(tx), sigmoid(ty), tw, th)
    :param anchors: scaled anchors based on current grid size
    :return: [batch, grid, grid, 3, 4] here 4 -> (bx, by, bw, bh)
    """
    rel_xy = rel_boxs[..., 0:2].to(device)
    rel_wh = rel_boxs[..., 2:4].to(device)
    grid_size = rel_boxs.shape[1]
    # create grid
    xy = torch.meshgrid(torch.range(0, grid_size-1), torch.range(0, grid_size-1))
    # coordinates --> (cx)
    xy = torch.stack(xy, dim=-1)
    # add one more dimension for calculation
    xy = torch.unsqueeze(xy, dim=2)
    xy = xy.to(device)
    # bx = sigmoid(tx) + cx
    # by = sigmoid(by) + cy
    abs_xy = rel_xy + xy
    # abs value based on anchors --> bw bh
    # scaled_anchors = scaled_anchors.reshape(1, 1, 1, 3, 2)
    abs_wh = torch.exp(rel_wh)*scaled_anchors
    abs_boxes = torch.cat((abs_xy, abs_wh), dim=-1)
    return abs_boxes


def intersection_over_union(pred_boxes, target_boxes):
    """
    calculates iou given predictions and labels boxes
    using asolulate value to calculate iou
    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (..., 4)
        boxes_labels (tensor): Correct labels of Bounding Boxes (..., 4)
    Returns:
        tensor: Intersection over union for all examples
    """

    # box top left coordinates (x1, y1)
    # box down right coordinates (x2, y2)
    box1_x1 = pred_boxes[..., 0:1] - pred_boxes[..., 2:3] / 2
    box1_y1 = pred_boxes[..., 1:2] - pred_boxes[..., 3:4] / 2
    box1_x2 = pred_boxes[..., 0:1] + pred_boxes[..., 2:3] / 2
    box1_y2 = pred_boxes[..., 1:2] + pred_boxes[..., 3:4] / 2
    box2_x1 = target_boxes[..., 0:1] - target_boxes[..., 2:3] / 2
    box2_y1 = target_boxes[..., 1:2] - target_boxes[..., 3:4] / 2
    box2_x2 = target_boxes[..., 0:1] + target_boxes[..., 2:3] / 2
    box2_y2 = target_boxes[..., 1:2] + target_boxes[..., 3:4] / 2

    # intersection box top left coordinates (x1, y1)
    # intersection box down right coordinates (x2, y2)
    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # min value = 0
    # intersection = ((x2 - x1) * (y2 - y1)).clamp(0)
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area-intersection)


# for evaluation
def mean_average_precision(model, loader, scaled_anchors, num_classes, iou_thres, conf_thres=0):
    """
    :param model:  model for evaluation
    :param loader: val dataloader for loading all predicted and targets bounding box on val dataset
    :param iou_thres: threshold for deciding Positive sample or Negative sample
    :param conf_thres: optional, filter confidence score larger than conf_thres
    :return: mean_average_precision on sepcific iou
    """

    # first, stack three scales predictions or targets boxes in one tensor
    pred_bboxes = []
    target_bboxes = []
    for idx, (x, y) in enumerate(tqdm(loader)):
        x = x.float()
        x = x.to(device)
        with torch.no_grad():
            out = model(x)
            out[0], out[1], out[2] = (
                out[0].to(device),
                out[1].to(device),
                out[2].to(device),
            )
            # print(torch.sigmoid(out[0][..., 0:4]))
            for i in range(3):
                out[i][..., 0:2] = torch.sigmoid(out[i][..., 0:2])
                out[i][..., 0:4] = rel_to_abs_box(out[i][..., 0:4], scaled_anchors[i*3:i*3+3])
                y[i][..., 0:4] = rel_to_abs_box(y[i][..., 0:4], scaled_anchors[i*3:i*3+3])
                out[i] = out[i].reshape((-1, out[i].shape[-1]))
                y[i] = y[i].reshape((-1, y[i].shape[-1]))
                # only focused on target samples
                pred_bboxes.append(out[i][y[i][..., 4] == 1])
                target_bboxes.append(y[i][y[i][..., 4] == 1])

    all_pred_boxes = torch.cat(pred_bboxes, dim=0).to(device)

    all_target_boxes = torch.cat(target_bboxes, dim=0).to(device)

    # second, do some processing on prediction data and target data
    # convert one-hot-encoding to index
    all_pred_boxes[..., 5] = torch.argmax(all_pred_boxes[..., 5:], dim=1)
    # same for target
    # all_target_boxes[..., 5] = torch.argmax(all_target_boxes[..., 5:], dim=1)

    # logistic regression for boxes and confidence
    all_pred_boxes[..., 4] = torch.sigmoid(all_pred_boxes[..., 4])

    # columns meaning in prediction tensor
    # 0-4 bbox, 4 confidence, 5 class_index, 6 correct_class, 7 ious

    # set the 6th columns in prediction tensor --> true of false for classes match with target
    classes_match = all_pred_boxes[..., 5] == all_target_boxes[..., 5]

    all_pred_boxes[..., 6] = classes_match

    # set the 7th columns in prediction tensor --> ious given target
    ious = intersection_over_union(all_pred_boxes[..., 0:4], all_target_boxes[..., 0:4])
    all_pred_boxes[..., 7] = torch.flatten(ious)

    average_precisions = []

    # third, calculate precision and recall for each class
    for i in range(num_classes):
        # for class i
        target_class_i = all_target_boxes[all_target_boxes[..., 5] == i]
        if target_class_i.shape[0] > 0:
            pred_class_i = all_pred_boxes[..., 0:8]
            pred_class_i = pred_class_i[pred_class_i[..., 5] == i]
            TP_FN = pred_class_i.shape[0]
            #  all targets in class i, including True Positive and False Negative
            # optional, filter boxes by given conf_thres
            pred_class_i = pred_class_i[pred_class_i[..., 4] > conf_thres]

            # assign TP or FN to pred box given iou threshold


            '''
            non_maximum_suppression
            can be used, I didn't use nms here. both works
            '''
            # # for each scale
            #
            # # sort boxes based on confidence score
            # pred_class_i = torch.stack(sorted(pred_class_i, key=lambda box: box[..., 4], reverse=True))
            # bboxes_after_nms = []
            #
            #
            # # compare iou of one box with the rest of boxes
            # def process_batch(bboxes):
            #     # add the first box (highest confidence score)into nms boxes list
            #     bboxes_after_nms.append(bboxes[0, ...].reshape(-1, bboxes[0, ...].shape[-1]))
            #     # expand the dimension of first box in ordrer to do iou calculation
            #     box1 = bboxes[0, 0:4].expand(* bboxes[1:, 0:4].shape)
            #
            #     # calculate ious of the first box with the rest of boxes
            #     ious = intersection_over_union(box1, bboxes[1:, 0:4])
            #     ious = torch.flatten(ious < iou_thres)
            #     # select boxes with iou < iou threshold
            #     rest = bboxes[1:, ...][ious]
            #         # print(rest.shape)
            #     return rest
            #
            # # deal with all rest boxes until no boxes left
            # while pred_class_i.shape[0]:
            #     pred_class_i = process_batch(pred_class_i)
            #
            # # bboxes_after_nms = bboxes_after_nms.reshape(-1, bboxes_after_nms.shape[-1])
            # pred_class_i = torch.cat(bboxes_after_nms, dim=0).to(device)

            pred_TP_FP_class = pred_class_i[..., 7] > iou_thres

            # sorted by confidence scores
            # pred_class_i = torch.stack(sorted(pred_class_i, key=lambda box: box[..., 4], reverse=True))

            # calculate TP and FP
            TP_cumsum = torch.cumsum(pred_TP_FP_class, dim=0)
            FP_cumsum = torch.cumsum(pred_TP_FP_class == 0, dim=0)

            '''
            precision = TP/ TP+FP
            recall = TP/ TP+FN
            '''
            precisions = TP_cumsum / (TP_cumsum + FP_cumsum )
            recalls = TP_cumsum / TP_FN
            precisions = torch.cat((torch.tensor([1]).to(device), precisions))
            recalls = torch.cat((torch.tensor([0]).to(device), recalls))

            # torch.trapz for numerical integration
            average_precisions.append(torch.trapz(precisions, recalls))
    model.train()
    mAP = sum(average_precisions) / len(average_precisions)
    print("mAP 0.5::" +str(mAP.item()))
    return mAP.item()


# for evaluation
def check_class_accuracy(model, loader, conf_thres):
    """
    used to print out and check class accuracy and object accuracy
    :param model: given model
    :param loader: validation data loader
    :param conf_thres: confidence score threshold of object
    """
    tot_class_preds, correct_class = 0, 0
    tot_noobj, correct_noobj = 0, 0
    tot_obj, correct_obj = 0, 0

    for idx, (x, y) in enumerate(tqdm(loader)):
        x = x.float()
        x = x.to(device)
        with torch.no_grad():
            out = model(x)

        # three scales
        for i in range(3):
            y[i] = y[i].to(device)
            obj = y[i][..., 4] == 1
            noobj = y[i][..., 4] == 0
            # class accuracy
            correct_class += torch.sum(
                # torch.argmax(torch.sigmoid(out[i][..., 5:])[obj], dim=-1) == torch.argmax(y[i][..., 5:][obj], dim=-1)
                torch.argmax(torch.sigmoid(out[i][..., 5:])[obj], dim=-1) == y[i][..., 5][obj]
            )
            tot_class_preds += torch.sum(obj)

            # object accuracy
            obj_preds = torch.sigmoid(out[i][..., 4]) > conf_thres
            correct_obj += torch.sum(
                obj_preds[obj] == y[i][..., 4][obj]
            )
            tot_obj += torch.sum(obj)

            # no_object accuracy
            correct_noobj += torch.sum(obj_preds[noobj] == y[i][..., 4][noobj])
            tot_noobj += torch.sum(noobj)
    cls_accu = correct_class/(tot_class_preds+1e-16)
    noobj_accu = correct_noobj/(tot_noobj+1e-16)
    obj_accu = correct_obj/(tot_obj+1e-16)

    print(f"Class accuracy is: {(cls_accu)*100:2f}%")
    print(f"No obj accuracy is: {(noobj_accu)*100:2f}%")
    print(f"Obj accuracy is: {(obj_accu)*100:2f}%")
    return [cls_accu.item(), noobj_accu.item(), obj_accu.item()]


# used for detection
def non_max_suppression(predictions, scaled_anchors, conf_thres=0.25, iou_thres=0.45):
    """
    remove all overlapping boxes given iou_thres --> one object is detected by only one box
    :param prediction:  [batch, grid, grid, 3, 5 + num_classes] for grid in three scales
    :param conf_thres: filter predictions given confidence threshold
    :param iou_thres: remove all overlapping boxes given iou threshold
    :return: list of boxes after nms [box1, box2....] here box1 --> (tx, ty, tw, th, classes)
    """
    # stack three scales predictions into bboxes
    bboxes = []
    # for each scale
    for i, each_scale_pred in enumerate(predictions):
        # convert relative coordinates of boxes into absolute coordinates
        each_scale_pred[..., 0:2] = torch.sigmoid(each_scale_pred[..., 0:2])
        each_scale_pred[..., 0:4] = rel_to_abs_box(each_scale_pred[..., 0:4], scaled_anchors[i*3:i*3+3])
        each_scale_pred[..., 4] = torch.sigmoid(each_scale_pred[..., 4])
        # filter predictions by confidence threshold
        each_scale_boxes = each_scale_pred[each_scale_pred[..., 4] > conf_thres]
        # reshape into [number_of_boxes, 5 + num_classes]
        each_scale_boxes = each_scale_boxes.reshape((-1, each_scale_boxes.shape[-1]))
        bboxes.append(each_scale_boxes)

    bboxes = torch.cat(bboxes, dim=0).to(device)
    # sort boxes based on confidence score
    bboxes = torch.stack(sorted(bboxes, key=lambda box: box[..., 4], reverse=True))

    bboxes_after_nms = []

    # compare iou of one box with the rest of boxes
    def process_batch(bboxes):
        # add the first box (highest confidence score)into nms boxes list
        bboxes_after_nms.append(bboxes[0, ...])
        # expand the dimension of first box in ordrer to do iou calculation
        box1 = bboxes[0, 0:4].expand(* bboxes[1:, 0:4].shape)

        # calculate ious of the first box with the rest of boxes
        ious = intersection_over_union(box1, bboxes[1:, 0:4])
        ious = torch.flatten(ious < iou_thres)
        # select boxes with iou < iou threshold
        rest = bboxes[1:, ...][ious]
        return rest

    # deal with all rest boxes until no boxes left
    while bboxes.shape[0]:
        bboxes = process_batch(bboxes)

    return bboxes_after_nms


# used for load darknet53 pretrained model
def copy_weights(bn, conv, ptr, weights, use_bn=True):
    if use_bn:
        num_bn_biases = bn.bias.numel()

        #Load the weights
        bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
        ptr += num_bn_biases

        bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
        ptr  += num_bn_biases

        bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
        ptr  += num_bn_biases

        bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
        ptr  += num_bn_biases

        #Cast the loaded weights into dims of model weights.
        bn_biases = bn_biases.view_as(bn.bias.data)
        bn_weights = bn_weights.view_as(bn.weight.data)
        bn_running_mean = bn_running_mean.view_as(bn.running_mean)
        bn_running_var = bn_running_var.view_as(bn.running_var)

        #Copy the data to model
        bn.bias.data.copy_(bn_biases)
        bn.weight.data.copy_(bn_weights)
        bn.running_mean.copy_(bn_running_mean)
        bn.running_var.copy_(bn_running_var)
    else:
        #Number of biases
        num_biases = conv.bias.numel()

        #Load the weights
        conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
        ptr = ptr + num_biases

        #reshape the loaded weights according to the dims of the model weights
        conv_biases = conv_biases.view_as(conv.bias.data)

        #Finally copy the data
        conv.bias.data.copy_(conv_biases)

    #Let us load the weights for the Convolutional layers
    num_weights = conv.weight.numel()
    conv_weights = torch.from_numpy(weights[ptr:ptr+num_weights])
    ptr = ptr + num_weights

    conv_weights = conv_weights.view_as(conv.weight.data)
    conv.weight.data.copy_(conv_weights)
    return ptr


def load_weights_darknet53(weightfile, yolov3):
    fp = open(weightfile, "rb")
    #The first 5 values are header information
    # 1. Major version number
    # 2. Minor Version Number
    # 3. Subversion number
    # 4. IMages seen
    header = np.fromfile(fp, dtype = np.int32, count = 5)
    weights = np.fromfile(fp, dtype = np.float32)
    ptr = 0
    layers = yolov3.layers
    i = 0
    for layer in layers:
        if isinstance(layer, ConvBlock):
            i += 1
            if i<7:
                conv = layer.conv
                bn = layer.bn
                ptr = copy_weights(bn, conv, ptr, weights)
        elif isinstance(layer, ResBlock):
            for res_layers in layer.layers:
                for block in res_layers:
                    conv = block.conv
                    bn = block.bn
                    ptr = copy_weights(bn, conv, ptr, weights)

    print("load finished")
    fp.close()


def load_weights_yolov3(weightfile, yolov3):
    fp = open(weightfile, "rb")
    #The first 5 values are header information
    # 1. Major version number
    # 2. Minor Version Number
    # 3. Subversion number
    # 4, 5. IMages seen
    header = np.fromfile(fp, dtype = np.int32, count = 5)
    weights = np.fromfile(fp, dtype = np.float32)
    print(len(weights))
    ptr = 0

    layers = yolov3.layers
    i = 0
    for layer in layers:
        if isinstance(layer, ConvBlock):
            i += 1
            if i<7:
                conv = layer.conv
                bn = layer.bn
                ptr = copy_weights(bn, conv, ptr, weights)
        elif isinstance(layer, ResBlock):
            for res_layers in layer.layers:
                for block in res_layers:
                    conv = block.conv
                    bn = block.bn
                    ptr = copy_weights(bn, conv, ptr, weights)

    predict_conv_list1 = layers[11]
    predict_conv_list1_1 = layers[12]
    smooth_conv1 = layers[13]
    print(smooth_conv1)

    predict_conv_list2 = layers[15]
    predict_conv_list2_1 = layers[16]
    smooth_conv2 = layers[17]

    predict_conv_list3 = layers[19]
    predict_conv_list3_1 = layers[20]
    for block in predict_conv_list1.yolo_layer:
        conv = block.conv
        bn = block.bn
        ptr = copy_weights(bn, conv, ptr, weights)


    i = 0
    for block in predict_conv_list1_1.detection_layer:
        i += 1
        if i==1:
            conv = block.conv
            bn = block.bn
            ptr = copy_weights(bn, conv, ptr, weights)
        if i==2:
            bn = 0
            conv = block.conv
            ptr = copy_weights(bn, conv, ptr, weights, use_bn=False)

    bn = smooth_conv1.bn
    conv = smooth_conv1.conv
    ptr = copy_weights(bn, conv, ptr, weights)

    for block in predict_conv_list2.yolo_layer:
        conv = block.conv
        bn = block.bn
        ptr = copy_weights(bn, conv, ptr, weights)

    i = 0
    for block in predict_conv_list2_1.detection_layer:
        i += 1
        if i==1:
            conv = block.conv
            bn = block.bn
            ptr = copy_weights(bn, conv, ptr, weights)
        if i==2:
            bn = 0
            conv = block.conv
            ptr = copy_weights(bn, conv, ptr, weights, use_bn=False)


    bn = smooth_conv2.bn
    conv = smooth_conv2.conv
    ptr = copy_weights(bn, conv, ptr, weights)

    for block in predict_conv_list3.yolo_layer:
        conv = block.conv
        bn = block.bn
        ptr = copy_weights(bn, conv, ptr, weights)

    i = 0
    for block in predict_conv_list3_1.detection_layer:
        i += 1
        if i==1:
            conv = block.conv
            bn = block.bn
            ptr = copy_weights(bn, conv, ptr, weights)
        if i==2:
            bn = 0
            conv = block.conv
            ptr = copy_weights(bn, conv, ptr, weights, use_bn=False)


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


