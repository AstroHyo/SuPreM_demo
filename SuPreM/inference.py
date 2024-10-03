import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import os
import argparse
import nibabel as nib
from monai.inferers import sliding_window_inference
from model.Universal_model import Universal_model
from dataset.dataloader_test import get_loader
from utils.utils import threshold_organ, pseudo_label_all_organ, pseudo_label_single_organ
from utils.utils import TEMPLATE, NUM_CLASS,ORGAN_NAME_LOW
from utils.utils import organ_post_process, threshold_organ, invert_transform

torch.multiprocessing.set_sharing_strategy('file_system')

### Model loading part

# Set num_samples, backbone, checkpoint_name from environment variables
print("Loading model...")
num_samples = int(os.environ.get('NUM_SAMPLES', 1))
backbone = os.environ.get('BACKBONE', 'unet') 
checkpoint_name = os.environ.get('CHECKPOINT', 'supervised_suprem_unet_2100')

# Load model
model = Universal_model(
  img_size=(96, 96, 96),
  in_channels=1,
  out_channels=NUM_CLASS,
  backbone=backbone,
  encoding='word_embedding',
)

# Load checkpoint
checkpoint = torch.load(f'./pretrained_checkpoints/{checkpoint_name}.pth')

# Load weighted values as key from the model and saved checkpoint 
allowed_keys = { key for key in model.state_dict().keys() }
loaded_dict = { key.removeprefix('module.'): value for key, value in checkpoint['net'].items() }

# Compare both of them have same key set
for key in loaded_dict.keys():
  if key not in allowed_keys:
    raise ValueError(f'Key {key} is not allowed in the model.')

# Put the checkpoint weighted values into the model, and send it to GPU
model.load_state_dict(loaded_dict)
model.cuda()

### Validation part
def validation(model, ValLoader, val_transforms, args):
    save_dir = args.save_dir
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model.eval()
    dice_list = {}
    for key in TEMPLATE.keys():
        dice_list[key] = np.zeros((2, NUM_CLASS)) # 1st row for dice, 2nd row for count
    for index, batch in enumerate(tqdm(ValLoader)):
        image,name_img = batch["image"].cuda(), batch["name_img"]
        image_file_path = os.path.join(args.data_root_path,name_img[0], 'ct.nii.gz')
        case_save_path = os.path.join(save_dir, name_img[0].split('/')[0])
        print(case_save_path)
        if not os.path.isdir(case_save_path):
            os.makedirs(case_save_path)
        organ_seg_save_path = os.path.join(save_dir, name_img[0].split('/')[0],'segmentations')
        print(image_file_path)
        print(image.shape)
        print(name_img)
        # if you want to copy ct file to the save_dir as well, uncomment the following lines
        # destination_ct = os.path.join(case_save_path,'ct.nii.gz')
        # if not os.path.isfile(destination_ct):
        #     shutil.copy(image_file_path, destination_ct)
        #     print("Image File copied successfully.")
        affine_temp = nib.load(image_file_path).affine
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
            # Process organs selectively
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
            # Process the combined result under the condition
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

# AttrDict part ?


# Handler part: Get the input from url, inference, and return the result in base64 format





def main():
    parser = argparse.ArgumentParser()
    ## for distributed training
    parser.add_argument('--dist', dest='dist', type=bool, default=False,
                        help='distributed training or not')
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--device")
    parser.add_argument("--epoch", default=0,type = int)
    ## logging
    parser.add_argument('--save_dir', default='/workspace/outputs', help='The dataset save path')
    ## model load
    parser.add_argument('--resume', default='./pretrained_checkpoints/supervised_suprem_unet_2100.pth', help='The path resume from checkpoint')
    parser.add_argument('--pretrain', default='...', 
                        help='The path of pretrain model')
    ## hyperparameter
    parser.add_argument('--max_epoch', default=1000, type=int, help='Number of training epoches')
    parser.add_argument('--store_num', default=10, type=int, help='Store model how often')
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--weight_decay', default=1e-5, type=float, help='Weight Decay')

    ## dataset
    parser.add_argument('--data_root_path', default='/workspace/inputs', help='data root path')
    parser.add_argument('--batch_size', default=1, type=int, help='batch size')
    parser.add_argument('--num_workers', default=8, type=int, help='workers numebr for DataLoader')
    parser.add_argument('--a_min', default=-175, type=float, help='a_min in ScaleIntensityRanged')
    parser.add_argument('--a_max', default=250, type=float, help='a_max in ScaleIntensityRanged')
    parser.add_argument('--b_min', default=0.0, type=float, help='b_min in ScaleIntensityRanged')
    parser.add_argument('--b_max', default=1.0, type=float, help='b_max in ScaleIntensityRanged')
    parser.add_argument('--space_x', default=1.5, type=float, help='spacing in x direction')
    parser.add_argument('--space_y', default=1.5, type=float, help='spacing in y direction')
    parser.add_argument('--space_z', default=1.5, type=float, help='spacing in z direction')
    parser.add_argument('--roi_x', default=96, type=int, help='roi size in x direction')
    parser.add_argument('--roi_y', default=96, type=int, help='roi size in y direction')
    parser.add_argument('--roi_z', default=96, type=int, help='roi size in z direction')
    parser.add_argument('--num_samples', default=1, type=int, help='sample number in each ct')

    parser.add_argument('--phase', default='test', help='train or validation or test')
    parser.add_argument('--original_label',action="store_true",default=False,help='whether dataset has original label')
    parser.add_argument('--cache_dataset', action="store_true", default=False, help='whether use cache dataset')
    parser.add_argument('--store_result', action="store_true", default=False, help='whether save prediction result')
    parser.add_argument('--cache_rate', default=0.6, type=float, help='The percentage of cached data in total')
    parser.add_argument('--cpu',action="store_true", default=False, help='The entire inference process is performed on the GPU ')
    parser.add_argument('--threshold_organ', default='Pancreas Tumor')
    parser.add_argument('--threshold', default=0.6, type=float)
    parser.add_argument('--backbone', default='unet', help='backbone [swinunetr or unet]')
    parser.add_argument('--create_dataset',action="store_true", default=False)

    args = parser.parse_args()

    # prepare the 3D model
 
    model = Universal_model(img_size=(args.roi_x, args.roi_y, args.roi_z),
                    in_channels=1,
                    out_channels=NUM_CLASS,
                    backbone=args.backbone,
                    encoding='word_embedding'
                    )
    #Load pre-trained weights
    store_dict = model.state_dict()
    store_dict_keys = [key for key, value in store_dict.items()]
    checkpoint = torch.load(args.resume)
    load_dict = checkpoint['net']
    load_dict_value = [value for key, value in load_dict.items()]

    for i in range(len(store_dict)):
        store_dict[store_dict_keys[i]] = load_dict_value[i]
        
    model.load_state_dict(store_dict)
    print('Use pretrained weights')
    model.cuda()
    torch.backends.cudnn.benchmark = True
    test_loader, val_transforms = get_loader(args)
    validation(model, test_loader, val_transforms, args)

if __name__ == "__main__":
    main()
