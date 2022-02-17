# --------------------------------------------------------
# Modified from https://github.com/pengzhiliang/MAE-pytorch
# --------------------------------------------------------'

import argparse
import datetime
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from PIL import Image
import cv2

from timm.models import create_model
from datasets import DataAugmentationForMAE


def get_args():
    parser = argparse.ArgumentParser('MAE visualization reconstruction script', add_help=False)
    parser.add_argument('img_path', type=str, help='input image path')
    parser.add_argument('save_path', type=str, help='save image path')
    parser.add_argument('model_path', type=str, help='checkpoint path of model')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size for backbone')
    parser.add_argument('--device', default='cuda:0',
                        help='device to use for training / testing')
    parser.add_argument('--imagenet_default_mean_and_std', default=False, action='store_false')
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='ratio of the visual tokens/patches need be masked')
    # Model parameters
    parser.add_argument('--model', default='pretrain_mae_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to vis')
    parser.add_argument('--drop_path', type=float, default=0.0, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    
    return parser.parse_args()


def get_model(args):
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
    )

    return model


def main(args):
    print(args)

    device = torch.device(args.device)
    cudnn.benchmark = True

    model = get_model(args)
    patch_size = model.encoder.patch_embed.patch_size
    print("Patch size = %s" % str(patch_size))
    args.window_size = (args.input_size // patch_size[0], args.input_size // patch_size[1])
    args.patch_size = patch_size

    model.to(device)
    checkpoint = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval()

    features_all = []
    imgs = np.load(args.img_path)
    for i, img in enumerate(imgs):
        print('extracting img ', i)
        if img.shape[0] != args.input_size:
            img = cv2.resize(img, (args.input_size, args.input_size))
        img = Image.fromarray(img.astype(np.uint8))
        img.convert('RGB')

        transforms = DataAugmentationForMAE(args)
        img, bool_masked_pos = transforms(img)
        bool_masked_pos = torch.from_numpy(bool_masked_pos)
    
        with torch.no_grad():
            img = img[None, :]
            bool_masked_pos = bool_masked_pos[None, :]
            img = img.to(device, non_blocking=True)
            bool_masked_pos = bool_masked_pos.to(device, non_blocking=True).flatten(1).to(torch.bool)
            
            features = model.encoder.forward(img, bool_masked_pos)
            features_pool = features.mean(dim = 1)
            features_all.append(features_pool.detach().cpu().numpy())
    
    np.save(args.save_path, np.asarray(features_all).squeeze())


if __name__ == '__main__':
    opts = get_args()
    main(opts)
