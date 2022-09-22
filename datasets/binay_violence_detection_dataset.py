from pathlib import Path
import collections
import random
import cv2
import numpy as np
from tqdm import tqdm
from functools import partial

import torch

from torch.utils.data import Dataset

tqdm = partial(tqdm, dynamic_ncols=True)


class BinaryViolenceDetectionDataset(Dataset):
    """  """
    
    def __init__(
        self,
        root='data/RWF-2000',
        split='all',
        split_seed=None,
        num_train_val_samples=None,
        num_test_samples=None,
        max_num_train_val_sample=None,
        frame_size=320,
        num_target_frames=64,
        in_memory=True,
        batch_size=0,
        target=None,
        transforms=None,
        mode='rgb',
        crop_black_borders=False,
    ):
        """ Dataset constructor.
        Args:
        """
        assert split in (
            'all', 'train', 'validation', 'test'), "Split must be one of ('train', 'validation', 'test', 'all')"
        assert split == 'all' or ((split_seed is not None) and (
                    num_train_val_samples is not None)), "You must supply split_seed and num_train_val_samples when split != 'all'"
        assert split == 'all' or (isinstance(num_train_val_samples, collections.abc.Sequence) and len(
            num_train_val_samples) == 2), 'num_train_val_samples must be a tuple of two ints'
        assert split == 'all' or sum(num_train_val_samples) <= max_num_train_val_sample, \
            f'n_train + n_val samples must be <= {max_num_train_val_sample}'
        assert mode in (
            'rgb', 'frame-difference'), "Mode must be one of ('rgb', 'frame-difference)"
        assert crop_black_borders is False or len(crop_black_borders) == 2, 'crop_black_borders must be a tuple of two ints'
        
        self.root = Path(root)
        
        self.split = split
        self.split_seed = split_seed
        self.num_train_val_samples = num_train_val_samples
        self.num_test_samples = num_test_samples
        self.target = target
        self.transforms = transforms
        self.mode = mode
        self.frame_difference = True if mode == 'frame_difference' else False

        self.in_memory = in_memory
        
        self.frame_size = (frame_size, frame_size)
        self.batch_size = batch_size
        self.crop_black_borders = crop_black_borders
        self.num_target_frames = num_target_frames
        
        # get list of videos in the given split
        self.video_paths = self._get_videos_in_split()
        
        if in_memory:
            self.data = [self._get_rgb_video_frames(video_path=video_path, resize=self.frame_size, num_target_frames=num_target_frames, crop_black_borders=crop_black_borders) 
                         for video_path in tqdm(self.video_paths)]
        
    def __len__(self):
        return len(self.video_paths)

    def _get_videos_in_split(self):
        videos_paths = self.root.rglob('*.[amm][vpp][ig4]')
        videos_paths = sorted(videos_paths)

        if self.split == 'all':
            return videos_paths

        # reproducible shuffle
        random.Random(self.split_seed).shuffle(videos_paths)

        n_train, n_val = self.num_train_val_samples
        if self.split == 'train':
            return videos_paths[:n_train]
        elif self.split == 'validation':
            return videos_paths[n_train:n_train + n_val]
        elif self.split == 'test' and self.num_test_samples:
            return videos_paths[n_train + n_val:n_train + n_val + self.num_test_samples]
        
    def _uniform_sampling(self, video_frames, num_target_frames=64):
        # calculate sampling interval 
        len_frames = int(len(video_frames))
        interval = int(np.ceil(len_frames / num_target_frames))
        
        # compute sampled video
        sampled_video = []
        for i in range(0, len_frames, interval):
            sampled_video.append(video_frames[i])     
            
        # eventually add padded frames 
        num_pad = num_target_frames - len(sampled_video)
        if num_pad > 0:
            padding = []
            for i in range(-num_pad, 0):
                try: 
                    padding.append(video_frames[i])
                except:
                    padding.append(video_frames[0])
            sampled_video += padding     
            
        return np.array(sampled_video, dtype=np.uint8)

    def _get_rgb_video_frames(
        self, 
        video_path=None, 
        resize=(224, 224), 
        num_target_frames=64, 
        crop_black_borders=False, 
    ):  
        # Load video and extract frames
        cap = cv2.VideoCapture(video_path.as_posix())
        len_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        try:
            frames = []
            for i in range(len_frames-1):
                _, frame = cap.read()
                frame_h, frame_w = frame.shape[:2]
                if crop_black_borders:
                    frame = frame[crop_black_borders[0]:frame_h-crop_black_borders[0], crop_black_borders[1]:frame_w-crop_black_borders[1], :]
                frame = cv2.resize(frame, resize, interpolation=cv2.INTER_AREA)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = np.reshape(frame, (resize[0], resize[1], 3))
                frames.append(frame)   
        except:
            print("Error: ", video_path, len_frames, i)
        finally:
            frames = np.array(frames)
            cap.release()
            
        # Sampling
        frames = self._uniform_sampling(frames, num_target_frames)
            
        # Converting to tensor
        frames = torch.from_numpy(frames)
            
        return frames
    
    def _get_video_frame_differences(self, frames, frame_interval=1):
        out = []
        
        for i in range(frames.shape[1] - frame_interval):
            out.append(frames[:, i+frame_interval, ...] - frames[:, i, ...])
            
        return torch.stack(out).moveaxis(1, 0)
    
    def __getitem__(self, index):
        video_id = self.video_paths[index].parts[-1]
        
        if self.in_memory:
            frames = self.data[index]
        else:
            frames = self._get_rgb_video_frames(self.video_paths[index], resize=self.frame_size, num_target_frames=self.num_target_frames, crop_black_borders=self.crop_black_borders)

        if self.transforms:
            frames = self.transforms(frames)
            
        if self.frame_difference:
            frames = self._get_video_frame_differences(frames)

        if self.target:
            # 0: NoFight, 1: Fight
            target = np.array([1], dtype=np.float32) if (self.video_paths[index].parts[-2] == "Fight") or (self.video_paths[index].parts[-2] == "Violence") else np.array([0], dtype=np.float32)
            datum = (frames, target)
        else:
            datum = (frames,)
            
        return datum + (video_id,)
        
    def __str__(self):
        s = f'{self.__class__.__name__}: ' \
            f'{self.split} split, ' \
            f'{len(self)} images, ' \
            f'frame size ({self.frame_size[0]}x{self.frame_size[1]})'
        return s
    
    

# Test code
def main():
    from torch.utils.data import DataLoader
    import numpy as np
    from PIL import Image
    from torchvision.transforms import Compose
    from torchvision.transforms._transforms_video import RandomHorizontalFlipVideo, ToTensorVideo, NormalizeVideo
    import torchvision.transforms._functional_video as F
    
    mean = [0.45, 0.45, 0.45]
    std = [0.225, 0.225, 0.225]
    num_frames = 64
    
    mode = "rgb"    # rgb or frame-difference
    denormalize = True

    transforms = Compose([
        ToTensorVideo(),
        RandomHorizontalFlipVideo(),
        NormalizeVideo(mean=mean, std=std)
    ])
    
    train_dataset_params = {
        'root': "/mnt/Dati_SSD_2/datasets/violence_detection/bus-violence/final",
        'split': "train",
        'split_seed': 87,
        'num_train_val_samples': (840, 280),
        'max_num_train_val_sample': 1120,
        'in_memory': False,
        'batch_size': 0,
        'target': True,
        'frame_size': 320,
        'mode': mode,
        'crop_black_borders': False,
        'num_target_frames': num_frames,
        'transforms': transforms,
    }
    
    batch_size = 1
    num_workers = 0
    collate_fn = None
    
    debug_dir = Path('datasets/trash')
    debug_dir.mkdir(exist_ok=True)

    train_dataset = BinaryViolenceDetectionDataset(**train_dataset_params)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
    
    for sample in train_loader:
        frames, labels, video_ids = sample
        
        for v in range(frames.shape[0]):
            video = frames[v, ...]
            video_name = video_ids[v]
            debug_dir = Path('datasets/trash/{}'.format(video_name.rsplit(".", 1)[0]))
            debug_dir.mkdir(exist_ok=True)
            if denormalize:
                video = F.normalize(video, mean=[0., 0., 0.], std=[1/0.225, 1/0.225, 1/0.225])
                video = F.normalize(video, mean=[-0.45, -0.45, -0.45], std=[1., 1., 1.])
            
            for f in range(video.shape[1]):
                frame = video[:, f, ...]
                frame = np.array(frame*255, dtype=np.uint8)
                frame = np.moveaxis(frame, 0, -1)
                Image.fromarray(frame).save(debug_dir / '{}_{}.png'.format(video_name, f))     

    
if __name__ == "__main__":
    main()
