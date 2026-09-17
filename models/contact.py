"""Optional contact API, concrete CVPR 2020 adapter, and backpack association.

Only this module knows the external model. Disabled mode imports no contact
dependencies. All boxes returned by providers must be in original-frame pixels.
"""
from dataclasses import dataclass
import importlib
import logging
import math
import sys
from typing import Protocol
from core.geometry import Box, Point, box_distance, center, distance, iou, point_box_distance
from utils.config import resolve_path

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class HandQuery:
    person_id: int
    hand: str
    wrist: Point


@dataclass(frozen=True)
class RawContact:
    hand_region: Box
    detected: bool
    confidence: float
    object_region: Box | None = None
    contact_state: str = "unknown"


@dataclass(frozen=True)
class ContactResult:
    person_id: int
    hand: str
    detected: bool
    confidence: float
    object_region: Box | None = None
    backpack_associated: bool = False
    association: str = "none"


class ContactProvider(Protocol):
    def predict(self, frame) -> list[RawContact]:
        """Return one result per observed hand, including non-contact hands."""


class ContactService:
    """Failure isolation and normalized per-person/per-hand evidence."""
    def __init__(self, cfg, provider=None, status="disabled"):
        self.cfg, self.provider, self.status = cfg, provider, status

    def detect(self, frame, hands, backpack_box):
        if self.provider is None or not hands or backpack_box is None:
            return []
        try:
            raw = list(self.provider.predict(frame))
            for item in raw:
                if not isinstance(item, RawContact) or not math.isfinite(item.confidence) or not 0 <= item.confidence <= 1:
                    raise ValueError("Provider must return RawContact with finite confidence in [0,1]")
                for box in (item.hand_region, item.object_region):
                    if box is not None and (len(box) != 4 or not all(math.isfinite(v) for v in box)
                                             or box[0] >= box[2] or box[1] >= box[3]):
                        raise ValueError("Provider returned an invalid original-frame box")
            return associate_contacts(raw, hands, backpack_box, self.cfg)
        except Exception as exc:
            LOG.warning("Contact inference failed: %s. Continuing without contact detector.", exc)
            self.provider, self.status = None, "unavailable"
            return []


def create_contact_detector(cfg):
    if not cfg["enable_contact_detector"]:
        return ContactService(cfg)
    try:
        if cfg["provider"] == "cvpr2020":
            provider = CVPR2020ContactDetector(cfg)
        elif cfg["provider"] == "custom":
            module, name = cfg["custom_factory"].split(":", 1)
            provider = getattr(importlib.import_module(module), name)(**cfg["options"])
        else:
            raise ValueError(f"Unknown contact provider: {cfg['provider']}")
        if not callable(getattr(provider, "predict", None)):
            raise TypeError("Contact provider must implement predict(frame)")
        LOG.info("Pretrained contact detector loaded: %s", cfg["provider"])
        return ContactService(cfg, provider, "enabled")
    except Exception as exc:
        LOG.warning("Contact model could not load: %s. Continuing without contact detector.", exc)
        return ContactService(cfg, status="unavailable")


def associate_contacts(raw, hands, backpack_box, cfg):
    # One model hand can support exactly one tracked person's wrist.
    pairs = sorted((point_box_distance(hand.wrist, item.hand_region),
                    distance(hand.wrist, center(item.hand_region)), qi, ri)
                   for qi, hand in enumerate(hands) for ri, item in enumerate(raw))
    used_queries, used_raw, results = set(), set(), []
    for gap, _, qi, ri in pairs:
        if gap > cfg["hand_wrist_max_distance_pixels"] or qi in used_queries or ri in used_raw:
            continue
        used_queries.add(qi)
        used_raw.add(ri)
        hand, item = hands[qi], raw[ri]
        valid = item.detected and item.confidence >= cfg["confidence_threshold"]
        associated, method = False, "none"
        if valid and backpack_box is not None:
            if item.object_region is not None:
                object_iou = iou(item.object_region, backpack_box)
                if object_iou > 0 and object_iou >= cfg["min_object_overlap"]:
                    associated, method = True, "object overlap"
                elif box_distance(item.object_region, backpack_box) <= cfg["max_object_distance_pixels"] and distance(center(item.object_region), center(backpack_box)) <= cfg["max_object_distance_pixels"]:
                    associated, method = True, "object proximity"
            elif point_box_distance(hand.wrist, backpack_box) <= cfg["max_object_distance_pixels"]:
                # This is spatial association of POSITIVE learned contact, not a contact classifier.
                associated, method = True, "wrist proximity (no object region)"
        results.append(ContactResult(hand.person_id, hand.hand, item.detected, item.confidence,
                                     item.object_region, associated, method))
    return results


class CVPR2020ContactDetector:
    """Shan et al. ResNet101/Faster-RCNN pretrained on 100DOH.

    Uses the pinned larsrpe PyTorch port's network architecture. Inference is
    implemented here to support configurable weights/device and correct object
    outputs (the port's convenience wrapper returns the hand array twice).
    """
    def __init__(self, options):
        weights = resolve_path(options["weights_path"])
        repo = resolve_path(options["repository_path"])
        if not weights.is_file():
            raise FileNotFoundError(f"{weights}; run python -m scripts.setup_contact")
        if not (repo / "handobdet").is_dir():
            raise FileNotFoundError(f"Contact repository missing: {repo}; run python -m scripts.setup_contact")
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        import numpy as np
        import torch
        from handobdet.lib.model.utils.config import cfg, cfg_from_file
        from handobdet.lib.model.faster_rcnn.resnet import resnet
        cfg_from_file(repo / "handobdet/cfgs/res101.yml")
        self.torch, self.options, self.cfg = torch, options, cfg
        self.device = torch.device(options["device"])
        cfg.CUDA = self.device.type == "cuda"
        cfg.USE_GPU_NMS = cfg.CUDA
        classes = np.asarray(["__background__", "targetobject", "hand"])
        self.network = resnet(classes, 101, pretrained=False, class_agnostic=False)
        self.network.create_architecture()
        checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
        self.network.load_state_dict(checkpoint["model"], strict=True)
        cfg.POOLING_MODE = checkpoint.get("pooling_mode", "align")
        self.network.to(self.device).eval()

    def predict(self, frame):
        import numpy as np
        import cv2
        from torchvision.ops import nms
        from handobdet.lib.model.rpn.bbox_transform import bbox_transform_inv, clip_boxes
        torch, cfg = self.torch, self.cfg
        height, width = frame.shape[:2]
        scale = min(float(cfg.TEST.SCALES[0]) / min(height, width), float(cfg.TEST.MAX_SIZE) / max(height, width))
        normalized = frame.astype(np.float32)
        normalized -= cfg.PIXEL_MEANS
        resized = cv2.resize(normalized, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        image = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).to(self.device)
        info = torch.tensor([[resized.shape[0], resized.shape[1], scale]], device=self.device)
        with torch.inference_mode():
            outputs = self.network(image, info, torch.zeros((1, 1, 5), device=self.device),
                                   torch.zeros(1, dtype=torch.long, device=self.device),
                                   torch.zeros((1, 1, 5), device=self.device))
            rois, probabilities, deltas = outputs[:3]
            extensions = outputs[-1]
            state_prob = extensions[0][0][0].softmax(dim=-1)
            offsets = extensions[1][0][0]
            if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                deltas = (deltas.reshape(-1, 4) * deltas.new_tensor(cfg.TRAIN.BBOX_NORMALIZE_STDS)
                          + deltas.new_tensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)).reshape(1, -1, 12)
            boxes = clip_boxes(bbox_transform_inv(rois[:, :, 1:5], deltas, 1), info, 1)[0] / scale
            scores = probabilities[0]
            selected = {}
            for label, threshold in ((1, self.options["object_confidence_threshold"]), (2, self.options["hand_confidence_threshold"])):
                indices = torch.where(scores[:, label] >= threshold)[0]
                keep = nms(boxes[indices, label * 4:(label + 1) * 4], scores[indices, label], cfg.TEST.NMS)
                selected[label] = indices[keep].cpu().tolist()
            object_boxes = [tuple(map(float, boxes[j, 4:8].cpu().tolist())) for j in selected[1]]
            raw = []
            names = ("none", "self", "other person", "portable object", "stationary object")
            for j in selected[2]:
                hand_box = tuple(map(float, boxes[j, 8:12].cpu().tolist()))
                state = int(state_prob[j].argmax())
                # A backpack is portable. Self/person contact cannot support it.
                detected = state == 3
                confidence = float(scores[j, 2] * state_prob[j, 3])
                obj = None
                if detected and object_boxes:
                    magnitude, dx, dy = offsets[j].cpu().tolist()
                    hc = center(hand_box)
                    # Published matching rule uses normalized offset magnitude *10000.
                    target = (hc[0] + magnitude * 10000 * dx, hc[1] + magnitude * 10000 * dy)
                    obj = min(object_boxes, key=lambda b: distance(center(b), target))
                raw.append(RawContact(hand_box, detected, confidence, obj, names[state]))
            return raw
