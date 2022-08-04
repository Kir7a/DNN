from itertools import chain
import torch
import torch.nn.functional as F
from tqdm import tqdm
from copy import deepcopy


def generate_det_row(det_size, start_pos_x, start_pos_y, det_step, N_det):
  p = []
  for i in range(N_det):
    left = start_pos_x+i*(int(det_step)+det_size)
    right = left + det_size
    up = start_pos_y
    down = start_pos_y + det_size
    p.append((up, down, left, right))
  return p

def set_det_pos(det_size=20, start_pos_x = 46, start_pos_y = 46):
  p = []
  p.append(generate_det_row(det_size, start_pos_x, start_pos_y, 2*det_size, 3))
  p.append(generate_det_row(det_size, start_pos_x, start_pos_y+3*det_size, 1*det_size, 4))
  p.append(generate_det_row(det_size, start_pos_x, start_pos_y+6*det_size, 2*det_size, 3))
  return list(chain.from_iterable(p))

def get_detector_imgs(detector_pos, N_pixels):
    labels_image_tensors=torch.zeros((10,N_pixels,N_pixels), dtype = torch.double)
    for ind,pos in enumerate(detector_pos):
        pos_l, pos_r, pos_u, pos_d = pos
        labels_image_tensors[ind,pos_l+1:pos_r+1, pos_u+1:pos_d+1] = 1
        labels_image_tensors[ind] = labels_image_tensors[ind]
    return labels_image_tensors