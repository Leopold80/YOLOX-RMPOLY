import math

import numpy as np
from loguru import logger

import torch
import torch.nn as nn
import torch.nn.functional as F

from yolox.utils import bboxes_iou, minimum_outer_rect, order_corners

# from .losses import PolyIOULoss
from .network_blocks import BaseConv, DWConv

from .yolo_head import YOLOXHead
from functools import wraps


class YOLOXPolyHead(YOLOXHead):
    def __init__(self, num_classes, width=1.0, strides=None, in_channels=None, act="silu", depthwise=False):
        if strides is None:
            strides = [8, 16, 32]
        if in_channels is None:
            in_channels = [256, 512, 1024]
        super().__init__(num_classes, width, strides, in_channels, act, depthwise)

        # 修改reg_preds层，使其输出通道数变为8，即四个点
        for i in range(len(in_channels)):
            # in_ch out_ch kernelsize stride
            self.reg_preds[i] = nn.Conv2d(int(256 * width), 8, (1, 1), (1, 1), 0)

        self.use_l1 = True
        self.l1_loss = nn.SmoothL1Loss(reduction='none')

    def forward(self, xin, labels=None, imgs=None):
        self.use_l1 = True

        outputs = []
        origin_preds = []
        x_shifts = []
        y_shifts = []

        expanded_strides = []

        for k, (cls_conv, reg_conv, stride_this_level, x) in enumerate(
                zip(self.cls_convs, self.reg_convs, self.strides, xin)
        ):
            x = self.stems[k](x)
            cls_x = x
            reg_x = x

            cls_feat = cls_conv(cls_x)
            cls_output = self.cls_preds[k](cls_feat)

            reg_feat = reg_conv(reg_x)
            reg_output = self.reg_preds[k](reg_feat)
            obj_output = self.obj_preds[k](reg_feat)

            if self.training:
                output = torch.cat([reg_output, obj_output, cls_output], 1)
                output, grid = self.get_output_and_grid(
                    output, k, stride_this_level, xin[0].type()
                )
                x_shifts.append(grid[:, :, 0])
                y_shifts.append(grid[:, :, 1])
                expanded_strides.append(
                    torch.zeros(1, grid.shape[1])
                        .fill_(stride_this_level)
                        .type_as(xin[0])
                )
                if self.use_l1:
                    batch_size = reg_output.shape[0]
                    hsize, wsize = reg_output.shape[-2:]
                    reg_output = reg_output.view(
                        batch_size, self.n_anchors, 8, hsize, wsize
                    )
                    reg_output = reg_output.permute(0, 1, 3, 4, 2).reshape(
                        batch_size, -1, 8
                    )
                    origin_preds.append(reg_output.clone())

            else:
                output = torch.cat(
                    [reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1
                )

            outputs.append(output)

        if self.training:
            return self.get_losses(
                imgs,
                x_shifts,
                y_shifts,
                expanded_strides,
                labels,
                torch.cat(outputs, 1),
                origin_preds,
                dtype=xin[0].dtype,
            )
        else:
            self.hw = [x.shape[-2:] for x in outputs]
            # [batch, n_anchors_all, 85]
            outputs = torch.cat(
                [x.flatten(start_dim=2) for x in outputs], dim=2
            ).permute(0, 2, 1)
            if self.decode_in_inference:
                return self.decode_outputs(outputs, dtype=xin[0].type())
            else:
                return outputs

    def get_output_and_grid(self, output, k, stride, dtype):
        grid = self.grids[k]

        batch_size = output.shape[0]
        # n_ch = 5 + self.num_classes
        n_ch = 8 + 1 + self.num_classes
        hsize, wsize = output.shape[-2:]
        if grid.shape[2:4] != output.shape[2:4]:
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, 1, hsize, wsize, 2).type(dtype)
            # grid = torch.stack((xv, yv, xv, yv, xv, yv, xv, yv), 2).view(1, 1, hsize, wsize, 8).type(dtype)
            self.grids[k] = grid

        output = output.view(batch_size, self.n_anchors, n_ch, hsize, wsize)
        output = output.permute(0, 1, 3, 4, 2).reshape(
            batch_size, self.n_anchors * hsize * wsize, -1
        )
        grid = grid.view(1, -1, 2)
        # grid = grid.view(1, -1, 8)
        # output[..., :2] = (output[..., :2] + grid) * stride
        output[..., :8] = (output[..., :8] + grid.repeat(1, 1, 4)) * stride
        # output[..., 2:4] = torch.exp(output[..., 2:4]) * stride
        return output, grid

    def get_losses(
            self,
            imgs,
            x_shifts,
            y_shifts,
            expanded_strides,
            labels,
            outputs,
            origin_preds,
            dtype,
    ):
        self.use_l1 = True
        # x_shifts_poly = [x_shift.clone() for x_shift in x_shifts]
        # y_shifts_poly = [y_shift.clone() for y_shift in y_shifts]
        labels_poly = labels.clone()
        outputs_poly = outputs.clone()

        # 将x_shifts和y_shifts更改为原有的形式
        # x_shifts = [i[..., 0] for i in x_shifts_poly]
        # y_shifts = [i[..., 0] for i in y_shifts_poly]

        # 将labels和outputs做最小外接矩形
        labels = labels_poly.new_zeros(labels_poly.shape[0], labels_poly.shape[1], 5)
        gtboxes = minimum_outer_rect(labels_poly[..., 1:].view(-1, 8))
        labels[..., 0] = labels_poly[..., 0]
        labels[..., 1:] = gtboxes.view(labels_poly.shape[0], labels_poly.shape[1], 4)

        outputs = outputs_poly.new_zeros(outputs_poly.shape[0], outputs_poly.shape[1], 4 + 1 + self.num_classes)
        bboxes = minimum_outer_rect(outputs_poly[..., 0:8].view(-1, 8))
        outputs[..., 0:4] = bboxes.view(outputs_poly.shape[0], outputs_poly.shape[1], 4)
        outputs[..., 4:] = outputs_poly[..., 8:]

        bbox_preds = outputs[:, :, :4]  # [batch, n_anchors_all, 4]
        obj_preds = outputs[:, :, 4].unsqueeze(-1)  # [batch, n_anchors_all, 1]
        cls_preds = outputs[:, :, 5:]  # [batch, n_anchors_all, n_cls]

        # calculate targets
        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects

        total_num_anchors = outputs.shape[1]
        x_shifts = torch.cat(x_shifts, 1)  # [1, n_anchors_all]
        y_shifts = torch.cat(y_shifts, 1)  # [1, n_anchors_all]
        expanded_strides = torch.cat(expanded_strides, 1)
        if self.use_l1:
            origin_preds = torch.cat(origin_preds, 1)

        cls_targets = []
        reg_targets = []
        l1_targets = []
        obj_targets = []
        fg_masks = []

        num_fg = 0.0
        num_gts = 0.0

        for batch_idx in range(outputs.shape[0]):
            num_gt = int(nlabel[batch_idx])
            num_gts += num_gt
            if num_gt == 0:
                cls_target = outputs.new_zeros((0, self.num_classes))
                reg_target = outputs.new_zeros((0, 4))
                l1_target = outputs.new_zeros((0, 8))
                obj_target = outputs.new_zeros((total_num_anchors, 1))
                fg_mask = outputs.new_zeros(total_num_anchors).bool()
            else:
                gt_bboxes_per_image = labels[batch_idx, :num_gt, 1:5]
                gt_classes = labels[batch_idx, :num_gt, 0]
                bboxes_preds_per_image = bbox_preds[batch_idx]

                try:
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(  # noqa
                        batch_idx,
                        num_gt,
                        total_num_anchors,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        bbox_preds,
                        obj_preds,
                        labels,
                        imgs,
                    )
                except RuntimeError:
                    logger.error(
                        "OOM RuntimeError is raised due to the huge memory cost during label assignment. \
                           CPU mode is applied in this batch. If you want to avoid this issue, \
                           try to reduce the batch size or image size."
                    )
                    torch.cuda.empty_cache()
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(  # noqa
                        batch_idx,
                        num_gt,
                        total_num_anchors,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        bbox_preds,
                        obj_preds,
                        labels,
                        imgs,
                        "cpu",
                    )

                torch.cuda.empty_cache()
                num_fg += num_fg_img

                cls_target = F.one_hot(
                    gt_matched_classes.to(torch.int64), self.num_classes
                ) * pred_ious_this_matching.unsqueeze(-1)
                obj_target = fg_mask.unsqueeze(-1)
                reg_target = gt_bboxes_per_image[matched_gt_inds]
                if self.use_l1:
                    poly_gt_bboxes_per_image = labels_poly[batch_idx, :num_gt, 1:9]
                    # l1_target = self.get_l1_target(
                    #     outputs.new_zeros((num_fg_img, 8)),
                    #     gt_bboxes_per_image[matched_gt_inds],
                    #     expanded_strides[0][fg_mask],
                    #     x_shifts=x_shifts[0][fg_mask],
                    #     y_shifts=y_shifts[0][fg_mask],
                    # )
                    l1_target = self.get_l1_target(
                        outputs.new_zeros((num_fg_img, 8)),
                        poly_gt_bboxes_per_image[matched_gt_inds],
                        expanded_strides[0][fg_mask],
                        x_shifts=x_shifts[0][fg_mask],
                        y_shifts=y_shifts[0][fg_mask],
                    )

            cls_targets.append(cls_target)
            reg_targets.append(reg_target)
            obj_targets.append(obj_target.to(dtype))
            fg_masks.append(fg_mask)
            if self.use_l1:
                l1_targets.append(l1_target)

        cls_targets = torch.cat(cls_targets, 0)
        reg_targets = torch.cat(reg_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        fg_masks = torch.cat(fg_masks, 0)
        if self.use_l1:
            l1_targets = torch.cat(l1_targets, 0)

        num_fg = max(num_fg, 1)
        loss_iou = (
                       self.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets)
                   ).sum() / num_fg
        loss_obj = (
                       self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets)
                   ).sum() / num_fg
        loss_cls = (
                       self.bcewithlog_loss(
                           cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets
                       )
                   ).sum() / num_fg
        if self.use_l1:
            l1_targets = order_corners(l1_targets).view(-1, 8)
            loss_l1 = (
                          self.l1_loss(origin_preds.view(-1, 8)[fg_masks], l1_targets)
                      ).sum() / num_fg
            loss_l1 *= 10.0
        else:
            loss_l1 = 0.0

        reg_weight = 5.0
        # 如果定位效果不好的话，试试先关掉iouloss，训练一段时间后再试着开启
        # TODO 要不要试着把l1loss改成正则化的形式？
        loss = reg_weight * loss_iou + loss_obj + loss_cls + loss_l1
        # loss = loss_obj + loss_cls + loss_l1

        return (
            loss,
            reg_weight * loss_iou,
            loss_obj,
            loss_cls,
            loss_l1,
            num_fg / max(num_gts, 1),
        )

    def get_l1_target(self, l1_target, gt, stride, x_shifts, y_shifts, eps=1e-8):
        # l1_target[:, 0] = gt[:, 0] / stride - x_shifts
        # l1_target[:, 1] = gt[:, 1] / stride - y_shifts
        # l1_target[:, 2] = torch.log(gt[:, 2] / stride + eps)
        # l1_target[:, 3] = torch.log(gt[:, 3] / stride + eps)
        l1_target[:, 0:8:2] = gt[:, 0:8:2] / stride.unsqueeze(1).repeat(1, 4) - x_shifts.unsqueeze(1).repeat(1, 4)
        l1_target[:, 1:8:2] = gt[:, 1:8:2] / stride.unsqueeze(1).repeat(1, 4) - y_shifts.unsqueeze(1).repeat(1, 4)
        return l1_target

    def decode_outputs(self, outputs, dtype):
        grids = []
        strides = []
        for (hsize, wsize), stride in zip(self.hw, self.strides):
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv, xv, yv, xv, yv, xv, yv), 2).view(1, -1, 8)
            grids.append(grid)
            shape = grid.shape[:2]
            strides.append(torch.full((*shape, 1), stride))

        grids = torch.cat(grids, dim=1).type(dtype)
        strides = torch.cat(strides, dim=1).type(dtype)

        # outputs[..., :2] = (outputs[..., :2] + grids) * strides
        # outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
        outputs[..., :8] = (outputs[..., :8] + grids) * strides
        return outputs
