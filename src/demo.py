from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import src._init_paths

import os

import cv2
import numpy as np
import torch
import torch.utils.data
from src.lib.opts import opts
from model import create_model
from src.lib.utils.debugger import Debugger
from src.lib.utils.image import get_affine_transform, transform_preds
from src.lib.utils.eval import get_preds, get_preds_3d
from src.lib.models.msra_resnet import get_pose_net

image_ext = ['jpg', 'jpeg', 'png']
mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)

def is_image(file_name):
  ext = file_name[file_name.rfind('.') + 1:].lower()
  return ext in image_ext


def demo_image(image):
  heads = {'hm': 16,'depth':16}
  model = get_pose_net(50, heads)
  optimizer = torch.optim.Adam(model.parameters(), 0.001)
  start_epoch = 1
  checkpoint = torch.load(
    './models/fusion_3d_var.pth', map_location=lambda storage, loc: storage)
  if type(checkpoint) == type({}):
    state_dict = checkpoint['state_dict']
  else:
    state_dict = checkpoint.state_dict()
  model.load_state_dict(state_dict, strict=False)
  '''
  if opt.resume:
    print('resuming optimizer')
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['epoch'] + 1
    for state in optimizer.state.values():
      for k, v in state.items():
        if isinstance(v, torch.Tensor):
          state[k] = v.cuda(opt.device, non_blocking=True)
  '''
  model = model.to(torch.device('cpu'))
  model.eval()
  ## deal with input img
  s = max(image.shape[0], image.shape[1]) * 1.0
  # central point
  c = np.array([image.shape[1] / 2., image.shape[0] / 2.], dtype=np.float32)
  trans_input = get_affine_transform(
      c, s, 0, [256, 256])
  inp = cv2.warpAffine(image, trans_input, (256, 256),
                         flags=cv2.INTER_LINEAR)
  inp = (inp / 255. - mean) / std
  inp = inp.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
  inp = torch.from_numpy(inp).to(torch.device('cpu'))
  ## to output the result
  out = model(inp)[-1]
  pred = get_preds(out['hm'].detach().cpu().numpy())[0]
  # here we get the 2D posi
  pred = transform_preds(pred, c, s, (64,64))
  pred_3d = get_preds_3d(out['hm'].detach().cpu().numpy(), 
                         out['depth'].detach().cpu().numpy())[0]
  return pred, pred_3d
  '''
  debugger = Debugger()
  debugger.add_img(image)
  debugger.add_point_2d(pred, (255, 0, 0))
  debugger.add_point_3d(pred_3d, 'b')
  debugger.show_all_imgs(pause=False)
  debugger.show_3d()
  
def main(opt):
  opt.heads['depth'] = opt.num_output
  if opt.load_model == '':
    opt.load_model = '../models/fusion_3d_var.pth'
  if opt.gpus[0] >= 0:
    opt.device = torch.device('cuda:{}'.format(opt.gpus[0]))
  else:
    opt.device = torch.device('cpu')
  
  model, _, _ = create_model(opt)
  model = model.to(opt.device)
  model.eval()

  if os.path.isdir(opt.demo):
    ls = os.listdir(opt.demo)
    for file_name in sorted(ls):
      if is_image(file_name):
        image_name = os.path.join(opt.demo, file_name)
        print('Running {} ...'.format(image_name))
        image = cv2.imread(image_name)
        demo_image(image, model, opt)
  elif is_image(opt.demo):
    print('Running {} ...'.format(opt.demo))
    image = cv2.imread(opt.demo)
    demo_image(image, model, opt)
    

if __name__ == '__main__':
  opt = opts().parse()
  main(opt)
'''