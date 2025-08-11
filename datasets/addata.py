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
            escaped_dir = glob.escape(self.dir)
            base_path_pattern = os.path.join(escaped_dir, '*')
            print(f"Searching with escaped pattern: {base_path_pattern}")
            
            all_paths = glob.glob(base_path_pattern)
            video_folders = [p for p in all_paths if os.path.isdir(p)]
            
            if not video_folders:
                print(f"Warning: No subdirectories found in {self.dir}")
                return

            print(f"Found {len(video_folders)} directories.")
            
            videos = [c.replace("\\", "/") for c in video_folders]

            for video_path in sorted(videos):
                video_name = os.path.basename(video_path)
                self.videos[video_name] = {}
                self.videos[video_name]['path'] = video_path
                
                escaped_video_path = glob.escape(video_path)
                image_pattern = os.path.join(escaped_video_path, '*.[jJ][pP][gG]') 
                # --- 수정 끝 ---
                
                frame_paths = glob.glob(image_pattern)
                frame_paths_sorted = sorted([c.replace("\\", "/") for c in frame_paths])
                self.videos[video_name]['frame'] = frame_paths_sorted
                self.videos[video_name]['length'] = len(frame_paths_sorted)
                print(f"  - In '{video_name}': Found {self.videos[video_name]['length']} frames.")
                    
            
    def get_all_samples(self):
        frames = []
        
        # setup()에서 이미 모든 정보를 self.videos에 로드했습니다.
        # 따라서 파일 시스템을 다시 검색할 필요 없이 self.videos의 키(비디오 이름)를 직접 사용합니다.
        # sorted()를 사용해 '01', '02', ... 순서로 일관되게 처리합니다.
        for video_name in sorted(self.videos.keys()):
            
            # self._time_step 만큼의 길이를 가진 샘플을 만들 수 있는 구간까지만 반복합니다.
            # 이 로직은 사용자님의 기존 코드와 동일하며, 올바른 방식입니다.
            num_frames = self.videos[video_name]['length']
            for i in range(num_frames - self._time_step):
                # 각 샘플의 시작 프레임 경로를 리스트에 추가합니다.
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