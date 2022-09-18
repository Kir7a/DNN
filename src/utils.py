from itertools import chain
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch.nn.functional as F
import numpy as np


def generate_det_row(det_size, start_pos_x, start_pos_y, det_step, N_det):
  p = []
  for i in range(N_det):
    left = start_pos_x+i*(int(det_step)+det_size)
    right = left + det_size
    up = start_pos_y
    down = start_pos_y + det_size
    p.append((up, down, left, right))
  return p

def set_det_pos(det_size=20, edge_x = 10, edge_y = 20, N_pixels = 200):
    p = []
    det_step_x_1 = (N_pixels - 2 * edge_x - 3 * det_size)//2
    det_step_x_2 = (N_pixels - 2 * edge_x - 4 * det_size)//3
    det_step_y = (N_pixels - 2 * edge_y - 3 * det_size)//2 + det_size
    p.append(generate_det_row(det_size, edge_x, edge_y, det_step_x_1, 3))
    p.append(generate_det_row(det_size, edge_x, edge_y + det_step_y, det_step_x_2, 4))
    p.append(generate_det_row(det_size, edge_x, edge_y + 2 * det_step_y, det_step_x_1, 3))
    return list(chain.from_iterable(p))

def get_detector_imgs(det_size=20, edge_x = 10, edge_y = 20, N_pixels = 200, visualize = True):
    detector_pos = set_det_pos(det_size, edge_x, edge_y, N_pixels)
    labels_image_tensors=torch.zeros((10,N_pixels,N_pixels), dtype = torch.double)
    for ind,pos in enumerate(detector_pos):
        pos_l, pos_r, pos_u, pos_d = pos
        labels_image_tensors[ind,pos_l+1:pos_r+1, pos_u+1:pos_d+1] = 1
        labels_image_tensors[ind] = labels_image_tensors[ind]
    if visualize:
      plt.imshow(np.zeros((N_pixels, N_pixels)))
      for det in detector_pos:
        rect = patches.Rectangle((det[2], det[0]), det_size, det_size, linewidth=1, edgecolor='r', facecolor='none')
        plt.gca().add_patch(rect)
      plt.show()
    return labels_image_tensors, detector_pos

def visualize(model, example, padding = 58):
  ex = F.pad(example[0], pad=(padding, padding, padding, padding))
  device = model.device
  out = model(ex.to(device))
  plt.subplot(1,2,1)
  plt.imshow(ex[0], interpolation='none')
  plt.title(f'Input image with label {example[1]}')
  output_image = out.detach().cpu()[0]
  plt.subplot(1,2,2)
  plt.imshow(output_image, interpolation='none')
  plt.title(f'Output image')
  #plt.colorbar()
  plt.show()

def mask_visualization(model):
  plt.figure(figsize = (30, 4))
  for ind, mask in enumerate(model.mask_layers):
    plt.subplot(1, len(model.mask_layers), ind+1)
    plt.imshow(torch.sigmoid(mask.phase.detach().cpu())*360, interpolation = 'none')
    plt.title(f'Mask of layer {ind+1}')
    plt.colorbar(label = 'Phase, deg.')

def visualize_n_samples(model, dataset, n, padding = 58, detector_pos = None):
  plt.figure(figsize = (25, 8))
  rand_ind = np.random.choice(range(len(dataset)), size=n, replace=False)
  device = model.device
  for number, ind in enumerate(rand_ind):
    ex = F.pad(dataset[ind][0], pad=(padding, padding, padding, padding))
    out, _ = model(ex.to(device))
    plt.subplot(2, n, number+1)
    plt.imshow(ex[0], interpolation='none')
    plt.title(f'Input image with label {dataset[ind][1]}')
    output_image = out.detach().cpu()[0]
    plt.subplot(2, n, n + number + 1)
    plt.imshow(output_image, interpolation='none')
    if detector_pos is not None:
      det_size = detector_pos[0][1] - detector_pos[0][0]
      for det in detector_pos:
        rect = patches.Rectangle((det[2], det[0]), det_size, det_size, linewidth=1, edgecolor='r', facecolor='none')
        plt.gca().add_patch(rect)
    plt.title(f'Output image')

def prop_vis(model, example, padding = 58, mode = 'abs', name_list = None):
  ex = F.pad(example, pad=(padding, padding, padding, padding))
  device = model.device
  final, imgs = model(ex.to(device))
  for ind, img in enumerate(imgs):
    if mode == 'abs':
      plt.imshow(img[0].abs().detach().cpu())
    elif mode == 'phase':
      plt.imshow(img[0].abs().detach().cpu())
    else:
      print(f'Do not support mode = "{mode}". Only "abs" or "phase"')
    if name_list is not None:
      plt.title(label = name_list[ind])
    plt.show()
  plt.imshow(final[0].abs().detach().cpu())
  if name_list is not None:
    plt.title(label = name_list[-1])
  plt.show()