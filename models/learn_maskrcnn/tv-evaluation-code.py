# Sample code from the TorchVision 0.3 Object Detection Finetuning Tutorial
# http://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

import os
import numpy as np
import torch
from PIL import Image

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import ToTensor, ToPILImage

from engine import train_one_epoch, evaluate
import utils
import transforms as T


class PennFudanDataset(object):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)

        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(int(num_objs)):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def convert_tensor_to_RGB(network_output):
    converted_tensor = torch.squeeze(network_output)

    return converted_tensor


def main(img_dir):
    # use our dataset and defined transformations
    dataset = PennFudanDataset(img_dir,
                               get_transform(train=True))
    dataset_test = PennFudanDataset(img_dir,
                                    get_transform(train=False))

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-100])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-100:])

    # define data loader
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    # This is the same path you stored your model
    path = "/home/aousn/git-repo/OIMNetPlus_new/OIMNetPlus/models/learn_maskrcnn/model/trainedModel2.pth"
    model = torch.load(path)
    model.eval()

    print("###### Running the model ######")
    model.eval()
    model.cuda()
    image = next(iter(data_loader_test))

    # Here we create a list, because the model expects a list of Tensors
    lista = []
    # It is important to send the image to CUDA, otherwise it will try to execute in the CPU
    x = image[0][0].cuda()
    lista.append(x)
    output = model(lista)

    print("### Converting output to RGB ###")
    # for number in range(0, 100):
    #     output = convert_tensor_to_RGB(output[0].get('masks'))
    #     output_cpu = output.cpu()
    #     ToPILImage()(output_cpu).save('images/test_' + str(number) + '.png', mode='png')

    output = convert_tensor_to_RGB(output[0].get('masks'))
    output_cpu = output.cpu()
    ToPILImage()(output_cpu).save('images/test_' + str(0) + '.png', mode='png')

    # Here, we pass the output to CPU in order to properly save the image
    # output_cpu = output.cpu()

    # Just a number to order your images
    # number = 3
    # for number in range(0, 100):
    #     ToPILImage()(output_cpu).save('images/test_' + str(number) + '.png', mode='png')
    # Saving the images
    # ToPILImage()(output_cpu).save('images/test_' + str(number) + '.png', mode='png')

    print("#### All Done! :) ####")


if __name__ == "__main__":
    img_dir = "/home/aousn/dataset/PRW_MaskSeg/"
    main(img_dir)
