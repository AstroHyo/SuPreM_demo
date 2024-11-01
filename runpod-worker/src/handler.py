import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import os
import nibabel as nib
import scipy.ndimage as ndimage
from monai.inferers import sliding_window_inference
from model.Universal_model import Universal_model
from dataset.dataloader_test import get_loader
from utils.utils import threshold_organ, pseudo_label_all_organ, pseudo_label_single_organ
from utils.utils import TEMPLATE, NUM_CLASS, ORGAN_NAME_LOW
from utils.utils import organ_post_process, threshold_organ, invert_transform

print("Loading model...")
num_samples = int(os.environ.get('NUM_SAMPLES', 1))

backbone = os.environ.get('BACKBONE', 'unet')
checkpoint_name = os.environ.get('CHECKPOINT', 'supervised_suprem_unet_2100')

model = Universal_model(
  img_size=(96, 96, 96),
  in_channels=1,
  out_channels=NUM_CLASS,
  backbone=backbone,
  encoding='word_embedding',
)

checkpoint = torch.load(f'./pretrained_checkpoints/{checkpoint_name}.pth')

allowed_keys = { key for key in model.state_dict().keys() }
loaded_dict = { key.removeprefix('module.'): value for key, value in checkpoint['net'].items() }

for key in loaded_dict.keys():
  if key not in allowed_keys:
    raise ValueError(f'Key {key} is not allowed in the model.')

model.load_state_dict(loaded_dict)
model.cuda()

ORGAN_NAME_TO_INDEX = {}
for organ_index in TEMPLATE['target']:
  organ_name = ORGAN_NAME_LOW[organ_index - 1]
  ORGAN_NAME_TO_INDEX[organ_name] = organ_index

ORGAN_IDS_VOLUME_NA = {ORGAN_NAME_TO_INDEX[i] for i in ['aorta', 'postcava']}

def voxelThreshold(slice):
  num_voxels = len(slice[slice > 0])
  return num_voxels < 100
  
def getCalcVolumeState(img_data, organ):
  if organ in ORGAN_IDS_VOLUME_NA:
    return "NA" # blood vessel, volume is NA
  slices = [
    img_data[:, :, 0],
    img_data[:, :, -1],
    img_data[0, :, :],
    img_data[-1, :, :],
    img_data[:, 0, :],
    img_data[:, -1, :],
  ]
  for slice in slices:
    if voxelThreshold(slice) is False:
      return "incomplete"
  return "complete"

DECIMAL_PRECISION_VOLUME = 2
DECIMAL_PRECISION_HU = 1
EROSION_PIXELS = 4
cube_len = (2 * EROSION_PIXELS) + 1
STRUCTURING_ELEMENT = np.ones([cube_len, cube_len, cube_len], dtype=bool)

def processMasks(ct_fdata, seg_id, seg):
  organ_data = {}
  seg_data = seg.get_fdata()
  state = getCalcVolumeState(seg_data, seg_id)
  if state == "complete":
    volume_cm = round(float(nib.imagestats.mask_volume(seg)/1000), DECIMAL_PRECISION_VOLUME)
    organ_data['volume_cm'] = volume_cm
  elif state == "incomplete":
    organ_data['volume_cm'] = "Incomplete organ"
  elif state == "NA":
    organ_data['volume_cm'] = "N/A"
  
  erosion_data = ndimage.binary_erosion(seg_data, structure=STRUCTURING_ELEMENT)
  hu_values = ct_fdata[erosion_data > 0]
  if len(hu_values) == 0:
    organ_data['mean_hu'] = 'N/A'
  else:
    mean_hu = round(float(np.mean(hu_values)), DECIMAL_PRECISION_HU)
    organ_data['mean_hu'] = mean_hu

  return organ_data

def validation(model, ValLoader, val_transforms, args):
  organ_data = {}

  save_dir = args.save_dir
  if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
  model.eval()
  dice_list = {}
  for key in TEMPLATE.keys():
    dice_list[key] = np.zeros((2, NUM_CLASS)) # 1st row for dice, 2nd row for count
  for index, batch in enumerate(tqdm(ValLoader)):
    image, name_img = batch["image"].cuda(), batch["name_img"]
    image_file_path = os.path.join(args.data_root_path, name_img[0], 'ct.nii.gz')
    case_save_path = os.path.join(save_dir, name_img[0].split('/')[0])
    print(case_save_path)
    if not os.path.isdir(case_save_path):
      os.makedirs(case_save_path)
    organ_seg_save_path = os.path.join(save_dir, name_img[0].split('/')[0], 'segmentations')
    print(image_file_path)
    print(image.shape)
    print(name_img)
    # if you want to copy ct file to the save_dir as well, uncomment the following lines
    # destination_ct = os.path.join(case_save_path, 'ct.nii.gz')
    # if not os.path.isfile(destination_ct):
    #     shutil.copy(image_file_path, destination_ct)
    #     print("Image File copied successfully.")
    ct = nib.load(image_file_path)
    ct_fdata = ct.get_fdata()
    affine_temp = ct.affine
    with torch.no_grad():
      pred = sliding_window_inference(image, (args.roi_x, args.roi_y, args.roi_z), 1, model, overlap=0.75, mode='gaussian')
      pred_sigmoid = F.sigmoid(pred)
    pred_hard = threshold_organ(pred_sigmoid, args)
    pred_hard = pred_hard.cpu()
    torch.cuda.empty_cache()

    B = pred_hard.shape[0]
    for b in range(B):
      organ_list_all = TEMPLATE['target'] # post processing target organ
      pred_hard_post, _ = organ_post_process(pred_hard.numpy(), organ_list_all, case_save_path, args)
      pred_hard_post = torch.tensor(pred_hard_post)
    
    if args.store_result:
      if not os.path.isdir(organ_seg_save_path):
        os.makedirs(organ_seg_save_path)

      # organ_index_all = TEMPLATE['target']      
      for organ_index in args.organ_indices:
        print(f'Processing organ {organ_index}')
        pseudo_label_single = pseudo_label_single_organ(pred_hard_post, organ_index, args)
        organ_name = ORGAN_NAME_LOW[organ_index-1]
        batch[organ_name]=pseudo_label_single.cpu()
        BATCH = invert_transform(organ_name, batch, val_transforms)
        organ_invertd = np.squeeze(BATCH[0][organ_name].numpy(), axis = 0)
        organ_save = nib.Nifti1Image(organ_invertd, affine_temp)
        new_name = os.path.join(organ_seg_save_path, organ_name + '.nii.gz')
        nib.save(organ_save, new_name)
        print('organ seg saved in path: %s'%(new_name))
        
        organ_data[organ_name] = processMasks(ct_fdata, organ_index, organ_save)

      if args.include_combined:
        print('Processing combined labels')
        pseudo_label_all = pseudo_label_all_organ(pred_hard_post, args)
        batch['pseudo_label'] = pseudo_label_all.cpu()
        BATCH = invert_transform('pseudo_label', batch, val_transforms)
        pseudo_label_invertd = np.squeeze(BATCH[0]['pseudo_label'].numpy(), axis = 0)
        pseudo_label_save = nib.Nifti1Image(pseudo_label_invertd, affine_temp)
        new_name = os.path.join(case_save_path, 'combined_labels.nii.gz')
        nib.save(pseudo_label_save, new_name)
        print('pseudo label saved in path: %s'%(new_name))
          
    torch.cuda.empty_cache()
  
  return organ_data

class AttrDict(dict):
  def __getattr__(self, name):
    try:
      return self[name]
    except KeyError:
      raise AttributeError(f"'AttrDict' object has no attribute '{name}'")
  
  def __setattr__(self, name, value):
    self[name] = value
  
  def __delattr__(self, name):
    try:
      del self[name]
    except KeyError:
      raise AttributeError(f"'AttrDict' object has no attribute '{name}'")

print("Model loaded.")

import runpod
import tempfile
import requests
import base64
import glob

# If your handler runs inference on a model, load the model here.
# You will want models to be loaded into memory before starting serverless.

def handler(job):
  """ Handler function that will be used to process jobs. """
  job_input = job['input']

  url = job_input.get('url')

  workdir = tempfile.TemporaryDirectory()

  input_dir = os.path.join(workdir.name, 'inputs')
  sample_dir = os.path.join(input_dir, 'sample')
  os.makedirs(sample_dir, exist_ok=True)

  output_dir = os.path.join(workdir.name, 'outputs')
  os.makedirs(output_dir, exist_ok=True)
  
  targets = job_input.get('targets', [])
  organ_indices = []
  include_combined = False
  for target in targets:
    if target == 'all':
      include_combined = True
      continue
    if target not in ORGAN_NAME_TO_INDEX:
      raise ValueError(f'Invalid target: {target}')
    organ_indices.append(ORGAN_NAME_TO_INDEX[target])
  
  if not organ_indices and not include_combined:
    raise ValueError('No targets specified')
  
  space_x = float(job_input.get('space_x', 1.5))
  space_y = float(job_input.get('space_y', 1.5))
  space_z = float(job_input.get('space_z', 1.5))
  
  a_min = float(job_input.get('a_min', -175))
  a_max = float(job_input.get('a_max', 250))
  b_min = float(job_input.get('b_min', 0.0))
  b_max = float(job_input.get('b_max', 1.0))
  
  roi_x = int(job_input.get('roi_x', 96))
  roi_y = int(job_input.get('roi_y', 96))
  roi_z = int(job_input.get('roi_z', 96))
  
  num_samples = int(job_input.get('num_samples', 1))

  print(f'Downloading {url} to {sample_dir}')
  response = requests.get(url, headers={
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36',
  }, stream=True)
  if response.status_code != 200:
    raise ValueError(f'Failed to download {url}: {response.status_code}')
  with open(os.path.join(sample_dir, 'ct.nii.gz'), 'wb') as f:
    for chunk in response.iter_content(chunk_size=128):
      if chunk:
        f.write(chunk)
  print(f'Downloaded {url} to {sample_dir}')
  
  test_loader, val_transformers = get_loader(AttrDict({
    'space_x': space_x,
    'space_y': space_y,
    'space_z': space_z,
    'a_min': a_min,
    'a_max': a_max,
    'b_min': b_min,
    'b_max': b_max,
    'roi_x': roi_x,
    'roi_y': roi_y,
    'roi_z': roi_z,
    'num_samples': num_samples,
    'data_root_path': input_dir,
    'original_label': False,
    'cache_dataset': False,
    'phase': 'test',
  }))
  
  organ_data = validation(model, test_loader, val_transformers, AttrDict({
    'save_dir': output_dir,
    'data_root_path': input_dir,
    'roi_x': roi_x,
    'roi_y': roi_y,
    'roi_z': roi_z,
    'store_result': True,
    'create_dataset': False,
    'cpu': False,
    'backbone': backbone,
    'organ_indices': organ_indices,
    'include_combined': include_combined,
  }))
  
  result = {}
  
  # for name in glob.glob(os.path.join(output_dir, 'sample', '*.nii.gz')):
  #   result[os.path.basename(name)] = base64.b64encode(open(name, 'rb').read()).decode('ascii')
  
  for name in glob.glob(os.path.join(output_dir, 'sample/segmentations', '*.nii.gz')):
    organ_name = os.path.basename(name).split('.')[0]
    result[organ_name] = {
      **organ_data[organ_name],
      'content': base64.b64encode(open(name, 'rb').read()).decode('ascii'),
    }
    # result[os.path.basename(name)] = base64.b64encode(open(name, 'rb').read()).decode('ascii')

  workdir.cleanup()
  
  return result

runpod.serverless.start({"handler": handler})
