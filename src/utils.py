"""
@author: Junguang Jiang, Baixu Chen
@contact: JiangJunguang1123@outlook.com, cbx_99_hasta@outlook.com
"""
import sys
import cv2
import os.path as osp
import time
from PIL import Image
import numpy as np

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from timm.data.auto_augment import auto_augment_transform, rand_augment_transform
from timm.loss import SoftTargetCrossEntropy
from timm.models import swin_small_patch4_window7_224
from torchtoolbox.transform import Cutout

import albumentations as A
from albumentations import DualTransform
import ttach as tta

sys.path.append('../../..')
import tllib.vision.datasets as datasets
import tllib.vision.models as models
from tllib.vision.transforms import ResizeImage
from tllib.utils.metric import accuracy, ConfusionMatrix
from tllib.utils.meter import AverageMeter, ProgressMeter
from tllib.vision.datasets.imagelist import MultipleDomainsDataset


class IBN(nn.Module):
    def __init__(self, planes):
        super(IBN, self).__init__()
        half1 = int(planes / 2)
        self.half = half1
        half2 = planes - half1
        self.IN = nn.InstanceNorm2d(half1, affine=True)
        self.BN = nn.BatchNorm2d(half2)

    def forward(self, x):
        split = torch.split(x, self.half, 1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        out = torch.cat((out1, out2), 1)
        return out


# map index to class, used for Food dataset only.
idx_to_class = {0: 0, 1: 1, 2: 10, 3: 11, 4: 12, 5: 13, 6: 14, 7: 15, 8: 16, 
                9: 17, 10: 18, 11: 19, 12: 2, 13: 20, 14: 21, 15: 22, 16: 23, 
                17: 24, 18: 25, 19: 26, 20: 27, 21: 28, 22: 29, 23: 3, 24: 30, 
                25: 31, 26: 4, 27: 5, 28: 6, 29: 7, 30: 8, 31: 9}

def get_model_names():
    return sorted(
        name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name])
    ) + timm.list_models()


def get_model(model_name, pretrain=True):
    if model_name in models.__dict__:
        # load models from tllib.vision.models
        backbone = models.__dict__[model_name](pretrained=pretrain)
    else:
        # load models from pytorch-image-models
        # backbone = timm.create_model(model_name, pretrained=pretrain)

        # *** convnext *** #
        backbone = timm.create_model(
            'convnext_base', 
            pretrained=True, 
            drop_path_rate=0.0,
            num_classes=32,
            # layer_scale_init_value=1e-6,
            head_init_scale=1.0,
        )

        # *** IBN *** #
        # from ibn import resnet50_ibn_a, resnet50_ibn_b
        # backbone = resnet50_ibn_a(
        #     pretrained=True,
        # )
        # backbone = resnet50_ibn_b(
        #     pretrained=True,
        # )

        # *** efficient-b7 *** #
        # backbone = timm.create_model(
        #     'efficientnet_b7', 
        #     pretrained=True, 
        #     num_classes=32,
        # )

        # IBN_a
        # backbone.stem[1] = IBN(planes=128)

        # *** swin-transformer *** #
        # backbone = timm.create_model(
        #     'swin_base_patch4_window7_224', 
        #     pretrained=True,
        #     num_classes=32,
        # )

        try:
            backbone.out_features = backbone.get_classifier().in_features
            backbone.reset_classifier(0, '')
        except:
            backbone.out_features = backbone.head.in_features
            backbone.head = nn.Identity()
            # backbone.classifier.out_features = backbone.classifier.in_features
    return backbone


def get_dataset_names():
    return sorted(
        name for name in datasets.__dict__
        if not name.startswith("__") and callable(datasets.__dict__[name])
    ) + ['Digits', 'Food']


def get_dataset(dataset_name, root, source, target, train_source_transform, val_transform, train_target_transform=None):
    if train_target_transform is None:
        train_target_transform = train_source_transform
    if dataset_name == "Digits":
        train_source_dataset = datasets.__dict__[source[0]](osp.join(root, source[0]), download=True,
                                                            transform=train_source_transform)
        train_target_dataset = datasets.__dict__[target[0]](osp.join(root, target[0]), download=True,
                                                            transform=train_target_transform)
        val_dataset = test_dataset = datasets.__dict__[target[0]](osp.join(root, target[0]), split='test',
                                                                  download=True, transform=val_transform)
        class_names = datasets.MNIST.get_classes()
        num_classes = len(class_names)
        
    elif dataset_name == "Food":
        train_source_dataset = datasets.__dict__[source[0]](osp.join(root, source[0]),
                                                            transform=train_source_transform)
        train_target_dataset = datasets.__dict__[target[0]](osp.join(root, target[0]),
                                                            transform=train_target_transform)
        val_dataset = test_dataset = datasets.__dict__[target[0]](osp.join(root, target[0]), split='test',
                                                                transform=val_transform)
        class_names = datasets.UPMC32.get_classes()
        num_classes = len(class_names)
        
    elif dataset_name in datasets.__dict__:
        # load datasets from tllib.vision.datasets
        dataset = datasets.__dict__[dataset_name]

        def concat_dataset(tasks, start_idx, **kwargs):
            # return ConcatDataset([dataset(task=task, **kwargs) for task in tasks])
            return MultipleDomainsDataset([dataset(task=task, **kwargs) for task in tasks], tasks,
                                          domain_ids=list(range(start_idx, start_idx + len(tasks))))

        train_source_dataset = concat_dataset(root=root, tasks=source, download=True, transform=train_source_transform,
                                              start_idx=0)
        train_target_dataset = concat_dataset(root=root, tasks=target, download=True, transform=train_target_transform,
                                              start_idx=len(source))
        val_dataset = concat_dataset(root=root, tasks=target, download=True, transform=val_transform,
                                     start_idx=len(source))
        if dataset_name == 'DomainNet':
            test_dataset = concat_dataset(root=root, tasks=target, split='test', download=True, transform=val_transform,
                                          start_idx=len(source))
        else:
            test_dataset = val_dataset
        class_names = train_source_dataset.datasets[0].classes
        num_classes = len(class_names)
    else:
        raise NotImplementedError(dataset_name)
    return train_source_dataset, train_target_dataset, val_dataset, test_dataset, num_classes, class_names


def validate(val_loader, model, args, device, save_path=None, epoch=0) -> float:
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    if args.per_class_eval:
        confmat = ConfusionMatrix(len(args.class_names))
    else:
        confmat = None

    # Add TTA
    transforms = tta.Compose(
        [
            tta.HorizontalFlip(),
            tta.Rotate90(angles=[0, 90]),
            tta.Resize(sizes=[(768,768),(512,512),(256,256)]),
            # tta.Scale(scales=[1, 2]),
            # tta.Multiply(factors=[0.9, 1.1]),
        ]
    )
    tta_model = tta.ClassificationTTAWrapper(
        model, transforms, merge_mode='mean'
    )
    TTA = False
    if TTA and epoch > 48:
        print(f"TTA mode: {TTA}")

    preds, hashs = [], []
    prob_list = []
    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(val_loader):
            images, target = data[:2]
            images = images.to(device)
            target = target.to(device)

            # compute output
            if TTA and epoch > 48:
                # *** TTA *** #
                for transformer in transforms:  # custom transforms or e.g. tta.aliases.d4_transform()
                    # augment image
                    augmented_image = transformer.augment_image(images)
                    output = tta_model(augmented_image)
                torch.cuda.empty_cache()
            else:
                output = model(images)
            pred = output.argmax(axis=1)
            loss = F.cross_entropy(output, target)

            # measure accuracy and record loss
            acc1, = accuracy(output, target, topk=(1,))
            if confmat:
                confmat.update(target, output.argmax(1))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # save the prediction result
            if len(data) == 3:
                preds.append(pred)
                hashs += data[2]

                # probability
                prob = F.softmax(output, dim=1).detach().cpu().numpy()
                prob_list.extend(prob.max(1))

            if i % args.print_freq == 0:
                progress.display(i)

        print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
        if confmat:
            print(confmat.format(args.class_names))

    if save_path is not None or args.phase == 'test':
        # save the prediction to the target path
        import pandas as pd
        from os import path
        preds = torch.hstack(preds)
        preds = torch.tensor([idx_to_class[p.item()] for p in preds]).to(device) # transform index to corresponding class
        if args.phase == 'test':
            save_path = './'
        df = pd.DataFrame(list(zip(hashs, preds.cpu().numpy())), columns=['Hash', 'label'])
        df.to_csv(path.join(save_path, f'submission_{epoch}.csv'), index=False)
        df2 = pd.DataFrame(list(zip(hashs, np.array(prob_list))), columns=['Hash', 'prob'])
        df2.to_csv(path.join(save_path, f'submission_{epoch}_prob.csv'), index=False)
    
    return top1.avg


def isotropically_resize_image(img, size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC):
    h, w = img.shape[:2]
    if max(w, h) == size:
        return img
    if w > h:
        scale = size / w
        h = h * scale
        w = size
    else:
        scale = size / h
        w = w * scale
        h = size
    interpolation = interpolation_up if scale > 1 else interpolation_down
    resized = cv2.resize(img, (int(w), int(h)), interpolation=interpolation)
    return resized


class IsotropicResize(DualTransform):
    def __init__(self, max_side, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC,
                 always_apply=False, p=1):
        super(IsotropicResize, self).__init__(always_apply, p)
        self.max_side = max_side
        self.interpolation_down = interpolation_down
        self.interpolation_up = interpolation_up

    def apply(self, img, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC, **params):
        return isotropically_resize_image(img, size=self.max_side, interpolation_down=interpolation_down,
                                          interpolation_up=interpolation_up)

    def apply_to_mask(self, img, **params):
        return self.apply(img, interpolation_down=cv2.INTER_NEAREST, interpolation_up=cv2.INTER_NEAREST, **params)

    def get_transform_init_args_names(self):
        return ("max_side", "interpolation_down", "interpolation_up")


def get_aug(img_arr):
    # trans = A.Compose([
    #         A.OneOf([
    #             A.RandomGamma(gamma_limit=(60, 120), p=0.9),
    #             A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.9),
    #             A.CLAHE(clip_limit=4.0, tile_grid_size=(4, 4), p=0.9),
    #             A.GaussianBlur(),
    #         ]),
    #         A.HorizontalFlip(p=0.5),
    #         A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=20,
    #                             interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, p=1),
    #         A.ImageCompression(quality_lower=60, quality_upper=90, p=0.5),
    #         A.OneOf([
    #                 A.CoarseDropout(),
    #                 A.GridDistortion(),
    #                 A.GridDropout(),
    #                 A.OpticalDistortion()
    #                 ]),
    # ])
    size = img_arr.shape[0]
    trans = A.Compose([
        A.ImageCompression(quality_lower=60, quality_upper=100, p=0.5),
        A.Downscale(scale_min=0.7, scale_max=0.9,
                    interpolation=cv2.INTER_LINEAR, p=0.3),
        A.GaussNoise(p=0.1),
        A.GaussianBlur(blur_limit=3, p=0.05),
        A.HorizontalFlip(),
        A.OneOf([
                IsotropicResize(
                    max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
                IsotropicResize(
                    max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_LINEAR),
                IsotropicResize(
                    max_side=size, interpolation_down=cv2.INTER_LINEAR, interpolation_up=cv2.INTER_LINEAR),
                ], p=1),
        A.PadIfNeeded(min_height=size, min_width=size,
                      border_mode=cv2.BORDER_CONSTANT),
        A.OneOf([A.RandomBrightnessContrast(), A.FancyPCA(),
                A.HueSaturationValue()], p=0.7),
        A.ToGray(p=0.2),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2,
                           rotate_limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.5),

        # A.OneOf([
        #         A.CoarseDropout(),
        #         A.GridDistortion(),
        #         A.GridDropout(),
        #         A.OpticalDistortion()
        #         ]),
    ]
    )
    trans_img = trans(image=img_arr)['image']
    return trans_img


def get_train_transform(resizing='default', scale=(0.8, 1.0), ratio=(3. / 4., 4. / 3.), random_horizontal_flip=True,
                        random_color_jitter=False, resize_size=512, norm_mean=(0.485, 0.456, 0.406),
                        norm_std=(0.229, 0.224, 0.225), auto_augment=None):
    """
    resizing mode:
        - default: resize the image to 256 and take a random resized crop of size 224;
        - cen.crop: resize the image to 256 and take the center crop of size 224;
        - res: resize the image to 224;
    """
    transformed_img_size = 224
    if resizing == 'default':
        transform = T.Compose([
            ResizeImage(256),
            T.RandomResizedCrop(224, scale=scale, ratio=ratio)
        ])
    elif resizing == 'cen.crop':
        transform = T.Compose([
            ResizeImage(256),
            T.CenterCrop(224)
        ])
    elif resizing == 'ran.crop':
        transform = T.Compose([
            ResizeImage(256),
            T.RandomCrop(224)
        ])
    elif resizing == 'res.':
        transform = ResizeImage(resize_size)
        transformed_img_size = resize_size
    else:
        raise NotImplementedError(resizing)
    transforms = [transform]
    if random_horizontal_flip:
        transforms.append(T.RandomHorizontalFlip())
    if auto_augment:
        aa_params = dict(
            translate_const=int(transformed_img_size * 0.45),
            img_mean=tuple([min(255, round(255 * x)) for x in norm_mean]),
            interpolation=Image.BILINEAR
        )
        if auto_augment.startswith('rand'):
            transforms.append(rand_augment_transform(auto_augment, aa_params))
        else:
            transforms.append(auto_augment_transform(auto_augment, aa_params))
    elif random_color_jitter:
        transforms.append(T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5))
    transforms.extend([
        Cutout(),
        T.ToTensor(),
        T.Normalize(mean=norm_mean, std=norm_std)
    ])
    return T.Compose(transforms)

    # trans = T.Compose([
    #     T.CenterCrop((512,512)),
    #     Cutout(p=0.5),
    #     T.RandomHorizontalFlip(p=0.7),
    #     T.RandomRotation(90),
    #     # T.ColorJitter(0.2,0.1,0.1,0.1),
    #     T.ToTensor(),
    #     T.Normalize(mean=norm_mean, std=norm_std),

    #     # ResizeImage(size=(224, 224)),
    #     # T.RandomHorizontalFlip(p=0.7),
    #     # T.RandomRotation(degrees=[-90.0, 90.0], expand=False, fill=0),
    #     # T.Pad(padding=40, fill=0, padding_mode='constant'),
    #     # T.ColorJitter(),
    #     # Cutout(),
    #     # T.ToTensor(),
    #     # T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    # ])
    # return trans


def get_val_transform(resizing='default', resize_size=224,
                      norm_mean=(0.485, 0.456, 0.406), norm_std=(0.229, 0.224, 0.225)):
    """
    resizing mode:
        - default: resize the image to 256 and take the center crop of size 224;
        – res.: resize the image to 224
    """
    if resizing == 'default':
        transform = T.Compose([
            ResizeImage(512),
            # T.CenterCrop((512,512)),
        ])
    elif resizing == 'res.':
        transform = ResizeImage(resize_size)
    else:
        raise NotImplementedError(resizing)
    return T.Compose([
        transform,
        T.ToTensor(),
        T.Normalize(mean=norm_mean, std=norm_std)
    ])
    # if resizing == 'default':
    #     transform = T.Compose([
    #         ResizeImage(256),
    #         T.CenterCrop(224),
    #     ])
    # elif resizing == 'res.':
    #     transform = ResizeImage(resize_size)
    # else:
    #     raise NotImplementedError(resizing)
    # return T.Compose([
    #     transform,
    #     T.ToTensor(),
    #     T.Normalize(mean=norm_mean, std=norm_std)
    # ])


def empirical_risk_minimization(train_source_iter, model, optimizer, lr_scheduler, epoch, args, device):
    batch_time = AverageMeter('Time', ':3.1f')
    data_time = AverageMeter('Data', ':3.1f')
    losses = AverageMeter('Loss', ':3.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')

    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses, cls_accs],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i in range(args.iters_per_epoch):
        x_s, labels_s = next(train_source_iter)[:2]
        x_s = x_s.to(device)
        labels_s = labels_s.to(device)

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        y_s, f_s = model(x_s)

        cls_loss = F.cross_entropy(y_s, labels_s)
        loss = cls_loss

        cls_acc = accuracy(y_s, labels_s)[0]

        losses.update(loss.item(), x_s.size(0))
        cls_accs.update(cls_acc.item(), x_s.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
