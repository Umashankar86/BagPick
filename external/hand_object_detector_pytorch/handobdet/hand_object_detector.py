from typing import List,Literal,Tuple
from numpy.typing import NDArray
from torch import Tensor

import os
import torch
import numpy as np
import cv2
from torchvision.ops import nms

from handobdet.lib.model.utils.config import cfg, cfg_from_file
from handobdet.lib.model.rpn.bbox_transform import clip_boxes
from handobdet.lib.model.rpn.bbox_transform import bbox_transform_inv
from handobdet.lib.model.utils.net_utils import vis_detections_filtered_objects_PIL
from handobdet.lib.model.utils.blob import im_list_to_blob
from handobdet.lib.model.faster_rcnn.resnet import resnet
from pathlib import Path



class HandObjectDetector:
    """
    Class for detecting hands and objects in images.
    Input: 
        input_format: input format of the image (RGB or BGR)
        thresh_obj: threshold for object detection
        thresh_hands: threshold for hand detection
    """
    def __init__(self,input_format: Literal["RGB","BGR"] = "RGB", tresh_obj: float = 0.5, thresh_hands: float = 0.5) -> None:
        module_path = os.path.abspath(__file__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_format = input_format
        self.thresh_hand = thresh_hands
        self.thresh_obj = tresh_obj
        
        # load config using some nasty path stuff
        resource_path = Path(__file__).parent / "cfgs/res101.yml"
        cfg_from_file(resource_path)

        cfg.USE_GPU_NMS = self.device == "cuda"
        np.random.seed(cfg.RNG_SEED)
        # load model using some nasty path stuff
        model_dir = "models"+ "/" + "res101" + "_handobj_100K" + "/" + "pascal_voc/faster_rcnn_1_8_132028.pth"
        load_name = Path(__file__).parent / model_dir
        self.pascal_classes = np.asarray(['__background__', 'targetobject', 'hand']) 
        self.fasterRCNN = resnet(self.pascal_classes, 101, pretrained=False, class_agnostic=False)
        self.fasterRCNN.create_architecture()
        if self.device == "cuda": 
            checkpoint = torch.load(load_name)
        else:
            checkpoint = torch.load(load_name, map_location=(lambda storage, loc: storage))
        self.fasterRCNN.load_state_dict(checkpoint['model'])
        if 'pooling_mode' in checkpoint.keys():
            cfg.POOLING_MODE = checkpoint['pooling_mode']

        self.fasterRCNN.to(self.device)
        self.fasterRCNN.eval()


    def detect(self,input: NDArray,viz=False) -> Tuple[Tensor,Tensor]:
        """
        Detects hands and objects in the image.
        Detections are represented as a array of length 10, 
        where the first 4 entries are the bounding box coordinates, 
        the 5th entry is the confidence score, 
        the 6th entry is the contact state, 
        the 7th to 9th entry are the offset vector, 
        and the 10th entry is the hand side.
        For object detections, only the first 5 entries have meaningfull information.
        
        Input:
            input: image to detect on as 3D tensor
            viz: if True, the image with the detections is shown
        returns: hand detections, object detections
        """
        im_data = torch.FloatTensor(1).to(self.device)
        im_info = torch.FloatTensor(1).to(self.device)
        num_boxes = torch.LongTensor(1).to(self.device) 
        gt_boxes = torch.FloatTensor(1).to(self.device)
        box_info = torch.FloatTensor(1)
        cfg.CUDA = (self.device == "cuda")
       
        im_data = im_data.to(self.device)
        
       
        im = input
        if self.input_format == "RGB":
            im = im [:,:,::-1]
        blobs, im_scales = _get_image_blob(im)
        assert len(im_scales) == 1, "Only single-image batch implemented"
        im_blob = blobs
        im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

        im_data_pt = torch.from_numpy(im_blob)
        im_data_pt = im_data_pt.permute(0, 3, 1, 2)
        im_info_pt = torch.from_numpy(im_info_np)

        with torch.no_grad():
            im_data.resize_(im_data_pt.size()).copy_(im_data_pt)
            im_info.resize_(im_info_pt.size()).copy_(im_info_pt)
            gt_boxes.resize_(1, 1, 5).zero_()
            num_boxes.resize_(1).zero_()
            box_info.resize_(1, 1, 5).zero_() 
        rois, cls_prob, bbox_pred, \
        rpn_loss_cls, rpn_loss_box, \
        RCNN_loss_cls, RCNN_loss_bbox, \
        rois_label, loss_list = self.fasterRCNN(im_data, im_info, gt_boxes, num_boxes, box_info) 
        scores = cls_prob.data
        boxes = rois.data[:, :, 1:5]

        # extact predicted params
        contact_vector = loss_list[0][0] # hand contact state info
        offset_vector = loss_list[1][0].detach() # offset vector (factored into a unit vector and a magnitude)
        lr_vector = loss_list[2][0].detach() # hand side info (left/right)

        # get hand contact 
        _, contact_indices = torch.max(contact_vector, 2)
        contact_indices = contact_indices.squeeze(0).unsqueeze(-1).float()

        # get hand side 
        lr = torch.sigmoid(lr_vector) > 0.5
        lr = lr.squeeze(0).float()

        if cfg.TEST.BBOX_REG:
            # Apply bounding-box regression deltas
            box_deltas = bbox_pred.data
            if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
            # Optionally normalize targets by a precomputed mean and stdev
                box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).to(self.device) \
                                + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).to(self.device)
                box_deltas = box_deltas.view(1, -1, 4 * len(self.pascal_classes))
            pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
            pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
        else:
            # Simply repeat the boxes, once for each class
            pred_boxes = np.tile(boxes, (1, scores.shape[1]))

        pred_boxes /= im_scales[0]
        scores = scores.squeeze()
        pred_boxes = pred_boxes.squeeze()
        
        
        obj_dets, hand_dets = None, None
        for j in range(1, len(self.pascal_classes)):
            # inds = torch.nonzero(scores[:,j] > thresh).view(-1)
            if self.pascal_classes[j] == 'hand':
                inds = torch.nonzero(scores[:,j]>self.thresh_hand).view(-1)
            elif self.pascal_classes[j] == 'targetobject':
                inds = torch.nonzero(scores[:,j]>self.thresh_obj).view(-1)

            # if there is det
            if inds.numel() > 0:
                cls_scores = scores[:,j][inds]
                _, order = torch.sort(cls_scores, 0, True)
                cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]
                cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1), contact_indices[inds], offset_vector.squeeze(0)[inds], lr[inds]), 1)
                cls_dets = cls_dets[order]
                keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
                cls_dets = cls_dets[keep.view(-1).long()]
                if self.pascal_classes[j] == 'targetobject':
                    obj_dets = cls_dets.cpu().numpy()
                if self.pascal_classes[j] == 'hand':
                    hand_dets = cls_dets.cpu().numpy()
        if viz:
            import matplotlib.pyplot as plt
            im2show = np.copy(im)
            im2show = vis_detections_filtered_objects_PIL(im2show, obj_dets, hand_dets, self.thresh_hand, self.thresh_obj)
            im2show = np.array(im2show)[:,:,0:3]
            plt.imshow(im2show)
            plt.show()
        return hand_dets,hand_dets
    

def _get_image_blob(im: NDArray) -> Tuple[NDArray,NDArray]:
    """Converts an image into a network input.
    Arguments:
      im (ndarray): a color image in BGR order
    Returns:
      blob (ndarray): a data blob holding an image pyramid
      im_scale_factors (list): list of image scales (relative to im) used
        in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)
    
       
   
def draw_detections(im, hand_dets: NDArray, obj_dets: NDArray,input_format: Literal['RGB','BGR'] = 'RGB', thresh_hand = 0.5, thresh_obj=0.5) -> NDArray:
    """
    Draws the detections on the image.
    Input:
        im: image to draw on
        hand_dets: hand detections
        obj_dets: object detections
        input_format: input format of the image
        thresh_hand: threshold for hand detection
        thresh_obj: threshold for object detection
    returns: image with detections as numpy array in RGB format
    """
    if input_format == "RGB":
        im = im [:,:,::-1]
    
    drawn= vis_detections_filtered_objects_PIL(im, obj_dets, hand_dets, thresh_hand, thresh_obj)
    return np.array(drawn)[:,:,0:3]
    