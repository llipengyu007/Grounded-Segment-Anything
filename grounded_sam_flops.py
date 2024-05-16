import argparse
import os
import copy

import numpy as np
import json
import torch
from PIL import Image, ImageDraw, ImageFont

# Grounding DINO
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

from ptflops import get_model_complexity_info

# segment anything
from segment_anything import (
    # build_sam,
    # build_sam_hq,
    SamPredictor,
    build_sam_vit_b,
    build_sam_vit_l,
    build_sam_vit_h,
)


import segment_anything
print("SAM PATH:{}".format(segment_anything.__path__))

build_sam = build_sam_vit_h

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm


def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Grounded-Segment-Anything Demo", add_help=True)
    parser.add_argument(
        "--sam_checkpoint", type=str, required=False, help="path to sam checkpoint file"
    )
    parser.add_argument(
        "--sam_hq_checkpoint", type=str, default=None, help="path to sam-hq checkpoint file"
    )
    parser.add_argument(
        "--use_sam_hq", action="store_true", help="using sam-hq for prediction"
    )
    parser.add_argument("--device", type=str, default="cpu", help="running on cpu only!, default=False")
    parser.add_argument("--grid_stride", type=int, default=1,
                        help="using str type, 1 means no use, positive value only use in global attn, negative vaule means use all attn")
    parser.add_argument("--hourglass_num_cluster", type=int, default=144)
    args = parser.parse_args()

    # cfg

    sam_checkpoint = args.sam_checkpoint
    grid_stride = args.grid_stride
    sam_hq_checkpoint = args.sam_hq_checkpoint
    use_sam_hq = args.use_sam_hq

    image_size = (3, 1024, 1024)
    # image_size = (3, 512, 512)

    device = args.device
    hourglass_num_cluster = args.hourglass_num_cluster

    # initialize SAM
    if use_sam_hq:
        print("using HQ SAM")
        predictor = SamPredictor(build_sam_hq(checkpoint=sam_hq_checkpoint).to(device))
    else:
        print("using original SAM")
        if sam_checkpoint.find("vit_h") >= 0:
            predictor = SamPredictor(build_sam_vit_h(grid_stride=grid_stride,
                                                         hourglass_num_cluster=hourglass_num_cluster).to(device))
        elif sam_checkpoint.find("vit_l") >= 0:
            predictor = SamPredictor(build_sam_vit_l(grid_stride=grid_stride,
                                                         hourglass_num_cluster=hourglass_num_cluster).to(device))
        elif sam_checkpoint.find("vit_b") >= 0:
            predictor = SamPredictor(build_sam_vit_b(grid_stride=grid_stride,
                                                         hourglass_num_cluster=hourglass_num_cluster).to(device))
        else:
            assert False, "sam_checkpoint error {}".format(sam_checkpoint)

        macs, params = get_model_complexity_info(predictor.model.image_encoder, image_size, as_strings=True,
                                           print_per_layer_stat=True, verbose=True)

    print("grid_stride:", grid_stride)
    print("macs", macs)
    print("params", params)
    print("complete")

