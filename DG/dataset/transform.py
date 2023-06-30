import random
import math
import torch
from torch import nn

from torchvision.ops import misc as misc_nn_ops
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.roi_heads import paste_masks_in_image


class PCBEVTransform(nn.Module):

    def __init__(self, min_size=200, max_size=200):
        super(PCBEVTransform, self).__init__()
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size
        #self.image_mean = [0.485, 0.456, 0.406]
        #self.image_std = [0.229, 0.224, 0.225]

    def forward(self, images, targets=None):
        images = [img for img in images]
        for i in range(len(images)):
            image = images[i]
            target = targets[i] if targets is not None else targets
            if image.dim() != 3:
                raise ValueError("images is expected to be a list of 3d tensors "
                                 "of shape [C, H, W], got {}".format(image.shape))
            #image = self.normalize(image)
            #image, target = self.resize(image, target)
            images[i] = image
            if targets is not None:
                targets[i] = target

        image_sizes = [img.shape[-2:] for img in images]
        images = self.batch_images(images, size_divisible=1)
#         image_list = ImageList(images, image_sizes)
#         return image_list, targets
        return images, targets

    def normalize(self, image):
        dtype, device = image.dtype, image.device
        mean = torch.as_tensor(self.image_mean, dtype=dtype, device=device)
        std = torch.as_tensor(self.image_std, dtype=dtype, device=device)
        return (image - mean[:, None, None]) / std[:, None, None]

    def resize(self, image, target):
        h, w = image.shape[-2:]
        min_size = float(min(image.shape[-2:]))
        max_size = float(max(image.shape[-2:]))
        if self.training:
            size = random.choice(self.min_size)
        else:
            # FIXME assume for now that testing uses the largest scale
            size = self.min_size[-1]
        scale_factor = size / min_size
        if max_size * scale_factor > self.max_size:
            scale_factor = self.max_size / max_size
#         image = torch.nn.functional.interpolate(
#             image[None], scale_factor=scale_factor, mode='bilinear', align_corners=False)[0]
#         image = torch.nn.functional.interpolate(
#             image[None], scale_factor=scale_factor, mode='nearest', recompute_scale_factor=True)[0]
        image = torch.nn.functional.interpolate(
            image[None], size=(600, 960),mode='nearest')[0]

        if target is None:
            return image, target

#         bbox = target["boxes"]
#         bbox = resize_boxes(bbox, (h, w), image.shape[-2:])
#         target["boxes"] = bbox

#         if "masks" in target:
#             mask = target["masks"]
#             mask = misc_nn_ops.interpolate(mask[None].float(), scale_factor=scale_factor)[0].byte()
#             target["masks"] = mask

        return image, target

    def batch_images(self, images, size_divisible=32):
        # concatenate
        max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))

        stride = size_divisible
        max_size = list(max_size)
        max_size[1] = int(math.ceil(float(max_size[1]) / stride) * stride)
        max_size[2] = int(math.ceil(float(max_size[2]) / stride) * stride)
        max_size = tuple(max_size)

        batch_shape = (len(images),) + max_size
        batched_imgs = images[0].new(*batch_shape).zero_()
        for img, pad_img in zip(images, batched_imgs):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)

        return batched_imgs

    def postprocess(self, result, image_shapes, original_image_sizes):
        if self.training:
            return result
        for i, (pred, im_s, o_im_s) in enumerate(zip(result, image_shapes, original_image_sizes)):
            boxes = pred["boxes"]
            boxes = resize_boxes(boxes, im_s, o_im_s)
            result[i]["boxes"] = boxes
            if "masks" in pred:
                masks = pred["masks"]
                masks = paste_masks_in_image(masks, boxes, o_im_s)
                result[i]["masks"] = masks
        return result



class POINTPILLARTransform(nn.Module):

    def __init__(self, min_size=200, max_size=200):
        super(POINTPILLARTransform, self).__init__()
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size
        #self.image_mean = [0.485, 0.456, 0.406]
        #self.image_std = [0.229, 0.224, 0.225]

    def forward(self, images, targets=None):
        images = [img for img in images]
        for i in range(len(images)):
            image = images[i]
            target = targets[i] if targets is not None else targets
            if image.dim() != 3:
                raise ValueError("images is expected to be a list of 3d tensors "
                                 "of shape [C, H, W], got {}".format(image.shape))
            #image = self.normalize(image)
            #image, target = self.resize(image, target)
            images[i] = image
            if targets is not None:
                targets[i] = target

        image_sizes = [img.shape[-2:] for img in images]
        images = self.batch_images(images, size_divisible=1)
#         image_list = ImageList(images, image_sizes)
#         return image_list, targets
        return images, targets

    def normalize(self, image):
        dtype, device = image.dtype, image.device
        mean = torch.as_tensor(self.image_mean, dtype=dtype, device=device)
        std = torch.as_tensor(self.image_std, dtype=dtype, device=device)
        return (image - mean[:, None, None]) / std[:, None, None]

    def resize(self, image, target):
        h, w = image.shape[-2:]
        min_size = float(min(image.shape[-2:]))
        max_size = float(max(image.shape[-2:]))
        if self.training:
            size = random.choice(self.min_size)
        else:
            # FIXME assume for now that testing uses the largest scale
            size = self.min_size[-1]
        scale_factor = size / min_size
        if max_size * scale_factor > self.max_size:
            scale_factor = self.max_size / max_size
#         image = torch.nn.functional.interpolate(
#             image[None], scale_factor=scale_factor, mode='bilinear', align_corners=False)[0]
#         image = torch.nn.functional.interpolate(
#             image[None], scale_factor=scale_factor, mode='nearest', recompute_scale_factor=True)[0]
        image = torch.nn.functional.interpolate(
            image[None], size=(600, 960),mode='nearest')[0]

        if target is None:
            return image, target

#         bbox = target["boxes"]
#         bbox = resize_boxes(bbox, (h, w), image.shape[-2:])
#         target["boxes"] = bbox

#         if "masks" in target:
#             mask = target["masks"]
#             mask = misc_nn_ops.interpolate(mask[None].float(), scale_factor=scale_factor)[0].byte()
#             target["masks"] = mask

        return image, target

    def batch_images(self, images, size_divisible=32):
        # concatenate
        max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))

        stride = size_divisible
        max_size = list(max_size)
        max_size[1] = int(math.ceil(float(max_size[1]) / stride) * stride)
        max_size[2] = int(math.ceil(float(max_size[2]) / stride) * stride)
        max_size = tuple(max_size)

        batch_shape = (len(images),) + max_size
        batched_imgs = images[0].new(*batch_shape).zero_()
        for img, pad_img in zip(images, batched_imgs):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)

        return batched_imgs

    def postprocess(self, result, image_shapes, original_image_sizes):
        if self.training:
            return result
        for i, (pred, im_s, o_im_s) in enumerate(zip(result, image_shapes, original_image_sizes)):
            boxes = pred["boxes"]
            boxes = resize_boxes(boxes, im_s, o_im_s)
            result[i]["boxes"] = boxes
            if "masks" in pred:
                masks = pred["masks"]
                masks = paste_masks_in_image(masks, boxes, o_im_s)
                result[i]["masks"] = masks
        return result

def resize_boxes(boxes, original_size, new_size):
    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(new_size, original_size))
    ratio_height, ratio_width = ratios
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    xmin = xmin * ratio_width
    xmax = xmax * ratio_width
    ymin = ymin * ratio_height
    ymax = ymax * ratio_height
    return torch.stack((xmin, ymin, xmax, ymax), dim=1)

import random
import torch

from torchvision.transforms import functional as F


# def _flip_coco_person_keypoints(kps, width):
#     flip_inds = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
#     flipped_data = kps[:, flip_inds]
#     flipped_data[..., 0] = width - flipped_data[..., 0]
#     # Maintain COCO convention that if visibility == 0, then x, y = 0
#     inds = flipped_data[..., 2] == 0
#     flipped_data[inds] = 0
#     return flipped_data


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
#         if random.random() < self.prob:
#             height, width = image.shape[-2:]
#             image = image.flip(-1)
# #             bbox = target["boxes"]
# #             bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
# #             target["boxes"] = bbox
#             if "Drivable_Area" in target:
#                 target["Drivable_Area"] = target["Drivable_Area"].flip(-1)
#             if "Port_Lane" in target:
#                 target["Port_Lane"] = target["Port_Lane"].flip(-1)
            
# #             if "keypoints" in target:
# #                 keypoints = target["keypoints"]
# #                 keypoints = _flip_coco_person_keypoints(keypoints, width)
# #                 target["keypoints"] = keypoints
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target
