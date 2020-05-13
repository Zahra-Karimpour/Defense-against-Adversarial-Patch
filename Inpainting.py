import os
import random
from argparse import ArgumentParser
import imageio
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.utils as vutils

from _inpainting_generator.py import Generator
from _inpainting_utils.py import get_config, random_bbox, mask_image, is_image_file, default_loader, normalize, get_model_list

def load_weights(path, device):
    model_weights = torch.load(path)
    return {
        k: v.to(device)
        for k, v in model_weights.items()
    }
def upcast(x):
    return np.clip((x + 1) * 127.5 , 0, 255).astype(np.uint8)
  
def inpainting(img_addr, patch):
  #   image_under_test = './generative_inpainting/examples/imagenet/imagenet_patches_ILSVRC2012_val_00008210_input.png'
  # image_under_test = patch_samples_dir + adv_name
  image_under_test = img_addr
  #   mask_under_test = './generative_inpainting/examples/center_mask_256.png'
  # mask_under_test = patch_samples_dir + mask_name
  mask_under_test = patch
  model_path = './model_address....../torch_model.p'
  config_yaml = './config_yaml_address......./config.yaml'
  


  config = get_config(config_yaml)
  #config = get_config(args.config)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  
  trainer = Trainer(config)
  #trainer.load_state_dict(load_weights(args.model_path, device), strict=False)
  trainer.load_state_dict(load_weights(model_path, device), strict=False)
  trainer.eval()

  #image = imageio.imread(args.image)
  image = imageio.imread(image_under_test)
  image = torch.FloatTensor(image).permute(2, 0, 1).unsqueeze(0).cuda()
  #mask = imageio.imread(args.mask)
  mask = imageio.imread(mask_under_test)
  mask = (torch.FloatTensor(mask[:, :, 0]) / 255).unsqueeze(0).unsqueeze(0).cuda()

  x = (image / 127.5 - 1) * (1 - mask).cuda()
  with torch.no_grad():
      _, result, _ = trainer.netG(x, mask)

  #imageio.imwrite(args.output, upcast(result[0].permute(1, 2, 0).detach().cpu().numpy()))
  return result[0]