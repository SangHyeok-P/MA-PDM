import numpy as np
from collections import OrderedDict
import os
import glob
import cv2
import torch
import ipdb
from PIL import Image
import json
import torch.utils.data as data
import torchvision.transforms as transforms
import random
def np_load_frame(filename, resize_height, resize_width):
    """
    Load image path and convert it to numpy.ndarray. Notes that the color channels are BGR and the color space
    is normalized from [0, 255] to [-1, 1].

    :param filename: the full path of image
    :param resize_height: resized height
    :param resize_width: resized width
    :return: numpy.ndarray
    """
    image_decoded = cv2.imread(filename)
    image_decoded = image_decoded[:,:,::-1]
    image_resized = cv2.resize(image_decoded, (resize_width, resize_height))
    return image_resized


class ADLoader(data.Dataset):
    def __init__(self, video_folder, dataset_type, phase, transform, resize_height, resize_width, time_step=6, num_pred=1, patch_size = 64, parse_patches=True):    
        self.dir = os.path.join(video_folder,dataset_type,"{}ing/frames".format(phase))
        self.transform = transform
        self.parse_patches = parse_patches
        self.patch_size = patch_size
        self.videos = OrderedDict()
        self._resize_height = resize_height
        self._resize_width = resize_width
        self._time_step = time_step
        self._num_pred = num_pred
        self.dataset_type = dataset_type
        self.phase = phase
        
        if "xd" in self.dir or "ucf" in self.dir:
            self.video = []
            with open(f'datasets/split_{self.phase}_{self.dataset_type}.json', 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for i in data.keys():    
                self.video.append(os.path.join(self.dir,i))
        else:
            self.video = glob.glob(os.path.join(self.dir, '*'))
        self.setup()
        self.samples = self.get_all_samples()
        self.n = (resize_height//patch_size)**2



    @staticmethod
    def get_params(imsize, output_size, n):
        w = h = imsize
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i_list = [random.randint(0, h - th) for _ in range(n)]
        j_list = [random.randint(0, w - tw) for _ in range(n)]
        return i_list, j_list, th, tw

    @staticmethod
    def random_crops(img, x, y, h, w):
        
        crops = []
        for i in range(len(x)):
            new_crop = img[:,:,x[i]:x[i]+h,y[i]:y[i]+w].unsqueeze(0)
            crops.append(new_crop)
        return crops

    def setup(self):
        if "xd" in self.dir or "ucf" in self.dir:
            videos = []
            with open(f'datasets/split_{self.phase}_{self.dataset_type}.json', 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for i in data.keys():    
                videos.append(os.path.join(self.dir,i))
        else:
            videos = glob.glob(os.path.join(self.dir, '*'))
        videos=[c.replace("\\","/") for c in videos ]
        for video in sorted(videos):
            video_name = video.split('/')[-1]
            self.videos[video_name] = {}
            self.videos[video_name]['path'] = video
            self.videos[video_name]['frame'] = glob.glob(os.path.join(video, '*.jpg'))
            self.videos[video_name]['frame'].sort()
            self.videos[video_name]['frame']=[c.replace("\\","/") for c in self.videos[video_name]['frame']]
            self.videos[video_name]['length'] = len(self.videos[video_name]['frame'])
            
            
    def get_all_samples(self):
        frames = []
        if "xd" in self.dir or "ucf" in self.dir:
            videos = []
            with open(f'datasets/split_{self.phase}_{self.dataset_type}.json', 'r', encoding='utf-8') as f:
                data = json.load(f)
            for i in data.keys():    
                videos.append(os.path.join(self.dir,i))
        else:
            videos = glob.glob(os.path.join(self.dir, '*'))
        videos=[c.replace("\\","/") for c in videos ]
        for video in sorted(videos):
            video_name = video.split('/')[-1]
            for i in range(len(self.videos[video_name]['frame'])-self._time_step):
                frames.append(self.videos[video_name]['frame'][i])
                           
        return frames               
            
        
    def __getitem__(self, index):
        
        video_name = self.samples[index].split('/')[-2]

        if self.dataset_type=="shanghai":
            frame_name = int(self.samples[index].split('/')[-1].split('.')[0])-1
        else:
            frame_name = int(self.samples[index].split('/')[-1].split('.')[0])
        rgb_img = torch.zeros(0)
         
        
        for i in range(self._time_step+self._num_pred):
            image = np_load_frame(self.videos[video_name]['frame'][frame_name+i], self._resize_height, self._resize_width)
            if self.transform is not None:
                rgb_img = torch.cat((rgb_img,self.transform(image).unsqueeze(0)))
        if self.parse_patches:
            i, j, h, w = self.get_params(self._resize_height, (self.patch_size, self.patch_size), self.n)
            input_img = self.random_crops(rgb_img, i, j, h, w) 
            return torch.cat(input_img,dim=0),video_name,np.array((i,j)).T
        else:
            return rgb_img,video_name,torch.zeros(0)
        
    def __len__(self):
        return len(self.samples)
if __name__ == "__main__":
    dl = ADLoader(video_folder="/home/ud202180593/data/datasets/",dataset_type="xd",phase="test",transform=transforms.Compose([
             transforms.ToTensor(),]),resize_height=256,resize_width=256,parse_patches=False) 
    print(dl.__len__())
    # ipdb.set_trace()
    # for i in range(dl.__len__()):
    print(dl.__getitem__(0)[0].shape)
    #     # ipdb.set_trace()
    #     pass