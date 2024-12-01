import os
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset
from tools import *


class SVMFDataset(Dataset):
    def __init__(self, path_vid, token, datamode='title+ocr'):

        self.data_complete = pd.read_json('../dataset_fakesv/data/data_complete.json', orient='records', dtype=False,
                                          lines=True)
        self.data_complete = self.data_complete[self.data_complete['annotation'] != '辟谣']
        self.maefeapath = '../dataset_fakesv/data/mae_fea'
        self.hubert_path = '../dataset_fakesv/data/hubert_ems/'
        self.vid = []

        with open('../dataset_fakesv/data-split/temporal/' + path_vid, "r") as fr:
            for line in fr.readlines():
                self.vid.append(line.strip())

        self.data = self.data_complete[self.data_complete.video_id.isin(self.vid)]
        # 设置类标签
        self.data['video_id'] = self.data['video_id'].astype('category')
        # 改变标签类别
        self.data['video_id'].cat.set_categories(self.vid)
        # 按照类别排序，而不是字母排序
        self.data.sort_values('video_id', ascending=True, inplace=True)
        # 重置数据帧的索引，并使用默认索引
        self.data.reset_index(inplace=True)

        self.tokenizer = token
        self.datamode = datamode

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        vid = item['video_id']

        # label
        label = 0 if item['annotation'] == '真' else 1
        label = torch.tensor(label)

        # text
        if self.datamode == 'title+ocr':
            title_tokens = self.tokenizer(item['title'] + ' ' + item['ocr'] + '' + item['keywords'], max_length=512,
                                          padding='max_length', truncation=True)
        elif self.datamode == 'ocr':
            title_tokens = self.tokenizer(item['ocr'] + '' + item['keywords'], max_length=512, padding='max_length',
                                          truncation=True)
        elif self.datamode == 'title':
            title_tokens = self.tokenizer(item['title'] + '' + item['keywords'], max_length=512, padding='max_length',
                                          truncation=True)
        title_inputid = torch.LongTensor(title_tokens['input_ids'])
        title_mask = torch.LongTensor(title_tokens['attention_mask'])

        # audio
        audio_item_path = self.hubert_path + vid + '.pkl'
        audio_fea = torch.load(audio_item_path)

        file_path = os.path.join(self.maefeapath, vid + '.pkl')
        # try:
        f = open(file_path, 'rb')
        frames = torch.load(f, map_location='cpu')
        frames = torch.FloatTensor(frames)

        return {
            'label': label,
            'title_inputid': title_inputid,
            'title_mask': title_mask,
            'audio_fea': audio_fea,
            'frames': frames,
        }


def pad_frame_sequence(seq_len, lst):
    attention_masks = []
    result = []
    for video in lst:
        video = torch.squeeze(video)
        video = torch.FloatTensor(video)
        ori_len = video.shape[0]
        if ori_len >= seq_len:
            gap = ori_len // seq_len
            video = video[::gap][:seq_len]
            mask = np.ones((seq_len))
        else:
            video = torch.cat((video, torch.zeros([seq_len - ori_len, video.shape[1]], dtype=torch.float)), dim=0)
            mask = np.append(np.ones(ori_len), np.zeros(seq_len - ori_len))
        result.append(video)
        mask = torch.IntTensor(mask)
        attention_masks.append(mask)
    return torch.stack(result), torch.stack(attention_masks)


def pad_video_frame_sequence(seq_len, lst):
    attention_masks = []
    result = []
    # print(len(lst))
    for video in lst:
        if len(video.shape) == 1:
            video = video.unsqueeze(0)
        ori_len = video.shape[0]
        if ori_len >= seq_len:
            gap = ori_len // seq_len
            video = video[::gap][:seq_len]
            mask = np.ones((seq_len))
        else:
            video = torch.cat((video, torch.zeros([seq_len - ori_len, video.shape[1]], dtype=torch.float)), dim=0)
            mask = np.append(np.ones(ori_len), np.zeros(seq_len - ori_len))
        result.append(video)
        mask = torch.IntTensor(mask)
        attention_masks.append(mask)
    return torch.stack(result), torch.stack(attention_masks)


def collate_fn(batch):
    num_frames = 86
    num_audioframes = 80

    title_inputid = [item['title_inputid'] for item in batch]
    title_mask = [item['title_mask'] for item in batch]

    # 根据帧数补齐关键帧特征
    frames = [item['frames'] for item in batch]
    frames, frames_masks = pad_video_frame_sequence(num_frames, frames)

    # 根据语音帧数补齐语音特征
    audio_feas = [item['audio_fea'] for item in batch]
    audio_feas, audiofeas_masks = pad_frame_sequence(num_audioframes, audio_feas)

    label = [item['label'] for item in batch]

    return {
        'label': torch.stack(label),
        'title_inputid': torch.stack(title_inputid),
        'title_mask': torch.stack(title_mask),
        'audio_feas': audio_feas,
        'audiofeas_masks': audiofeas_masks,
        'frames': frames,
        'frames_masks': frames_masks,
    }
#fakeTT
'''
class SVFENDDataset(Dataset):
    def __init__(self, path_vid, token, datamode='title+ocr'):
        # self.dataset = dataset
        self.vid = []
        self.tokenizer = token
        self.datamode = datamode
        
        self.data_complete = pd.read_json('../dataset_fakett/data/data.json', orient='records', dtype=False,
                                          lines=True)
        self.maefeapath = '../dataset_fakett/data/mae_fea'
        self.hubert_path = '../dataset_fakett/data/hubert_fea/'

        with open('../dataset_fakett/data-split/temporal/' + path_vid, "r") as fr:
            for line in fr.readlines():
                self.vid.append(line.strip())

        self.data = self.data_complete[self.data_complete.video_id.isin(self.vid)]
        self.data['video_id'] = self.data['video_id'].astype('category')
        self.data['video_id'].cat.set_categories(self.vid)
        self.data.sort_values('video_id', ascending=True, inplace=True)
        self.data.reset_index(inplace=True)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        vid = item['video_id']

        # label
        label = 0 if item['annotation'] == 'real' else 1
        # text
        if self.datamode == 'title+ocr':
            title_tokens = self.tokenizer(item['description'] + ' ' + item['recognize_ocr'] + '' + item['event'],
                                              max_length=512, padding='max_length', truncation=True)
        else:
            title_tokens = self.tokenizer(item['recognize_ocr'] + '' + item['event'], max_length=512,
                                              padding='max_length', truncation=True)
        label = torch.tensor(label)
        title_inputid = torch.LongTensor(title_tokens['input_ids'])
        title_mask = torch.LongTensor(title_tokens['attention_mask'])

        # audio
        audio_item_path = self.hubert_path + vid + '.pkl'
        audio_fea = torch.load(audio_item_path)
        
        # frames
        file_path = os.path.join(self.maefeapath, vid + '.pkl')
        f = open(file_path, 'rb')
        frames = torch.load(f, map_location='cpu')
        frames = torch.FloatTensor(frames)

        return {
            'label': label,
            'title_inputid': title_inputid,
            'title_mask': title_mask,
            'audio_fea': audio_fea,
            'frames': frames,
        }
'''