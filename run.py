from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from untils.dataloader import *
from untils.Trainer_3set import Trainer3
from model.SVMF import *


def pad_sequence(seq_len, lst, emb):
    result = []
    for video in lst:
        if isinstance(video, list):
            video = torch.stack(video)
        ori_len = video.shape[0]
        if ori_len == 0:
            video = torch.zeros([seq_len, emb], dtype=torch.long)
        elif ori_len >= seq_len:
            if emb == 200:
                video = torch.FloatTensor(video[:seq_len])
            else:
                video = torch.LongTensor(video[:seq_len])
        else:
            video = torch.cat([video, torch.zeros([seq_len - ori_len, video.shape[1]], dtype=torch.long)], dim=0)
            if emb == 200:
                video = torch.FloatTensor(video)
            else:
                video = torch.LongTensor(video)
        result.append(video)
    return torch.stack(result)


def pad_frame_sequence(seq_len, lst):
    attention_masks = []
    result = []
    for video in lst:
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


def _init_fn(worker_id):
    np.random.seed(2022)


def SVMF_collate_fn(batch):
    num_frames = 83
    num_audioframes = 50

    title_inputid = [item['title_inputid'] for item in batch]
    title_mask = [item['title_mask'] for item in batch]

    frames = [item['frames'] for item in batch]
    frames, frames_masks = pad_frame_sequence(num_frames, frames)

    audioframes = [item['audioframes'] for item in batch]
    audioframes, audioframes_masks = pad_frame_sequence(num_audioframes, audioframes)

    label = [item['label'] for item in batch]

    return {
        'label': torch.stack(label),
        'title_inputid': torch.stack(title_inputid),
        'title_mask': torch.stack(title_mask),
        'audioframes': audioframes,
        'audioframes_masks': audioframes_masks,
        'frames': frames,
        'frames_masks': frames_masks,
    }


class Run():
    def __init__(self,
                 config
                 ):

        self.model_name = config['model_name']
        self.mode_eval = config['mode_eval']
        # self.fold = config['fold']
        self.data_type = 'SVMF'

        self.epoches = config['epoches']
        self.batch_size = config['batch_size']
        self.num_workers = config['num_workers']
        self.epoch_stop = config['epoch_stop']
        self.seed = config['seed']
        self.device = config['device']
        self.lr = config['lr']
        self.lambd = config['lambd']
        self.save_param_dir = config['path_param']
        self.path_tensorboard = config['path_tensorboard']
        self.dropout = config['dropout']
        self.weight_decay = config['weight_decay']
        self.event_num = 616
        self.mode = 'normal'

    def get_dataloader(self, data_type, data_fold):
        collate_fn = None

        if data_type == 'SVMF':
            dataset_train = SVMFDataset(f'vid_fold_no_{data_fold}.txt')
            dataset_test = SVMFDataset(f'vid_fold_{data_fold}.txt')
            collate_fn = SVMF_collate_fn

        train_dataloader = DataLoader(dataset_train, batch_size=self.batch_size,
                                      num_workers=self.num_workers,
                                      pin_memory=True,
                                      shuffle=True,
                                      worker_init_fn=_init_fn,
                                      collate_fn=collate_fn)

        test_dataloader = DataLoader(dataset_test, batch_size=self.batch_size,
                                     num_workers=self.num_workers,
                                     pin_memory=True,
                                     shuffle=False,
                                     worker_init_fn=_init_fn,
                                     collate_fn=collate_fn)

        dataloaders = dict(zip(['train', 'test'], [train_dataloader, test_dataloader]))

        return dataloaders

    def get_dataloader_temporal(self):
        token = pretrain_bert_token()
        dataset_train = SVMFDataset('vid_time3_train.txt',token)
        dataset_val = SVMFDataset('vid_time3_val.txt',token)
        dataset_test = SVMFDataset('vid_time3_test.txt',token)

        train_dataloader = DataLoader(dataset_train, batch_size=self.batch_size,
                                      num_workers=self.num_workers,
                                      pin_memory=True,
                                      shuffle=True,
                                      worker_init_fn=_init_fn,
                                      collate_fn=collate_fn)
        val_dataloader = DataLoader(dataset_val, batch_size=self.batch_size,
                                    num_workers=self.num_workers,
                                    pin_memory=True,
                                    shuffle=False,
                                    worker_init_fn=_init_fn,
                                    collate_fn=collate_fn)
        test_dataloader = DataLoader(dataset_test, batch_size=self.batch_size,
                                     num_workers=self.num_workers,
                                     pin_memory=True,
                                     shuffle=False,
                                     worker_init_fn=_init_fn,
                                     collate_fn=collate_fn)

        dataloaders = dict(zip(['train', 'val', 'test'], [train_dataloader, val_dataloader, test_dataloader]))

        return dataloaders

    def get_model(self):
        if self.model_name == 'SVMF':
            self.model = SVMFModel(bert_model='pretrain_bert_models', fea_dim=128, dropout=self.dropout)#原来是128
        return self.model

    def main(self):
        if self.mode_eval == "temporal":
            self.model = self.get_model()
            dataloaders = self.get_dataloader_temporal()
            trainer = Trainer3(model=self.model, device=self.device, lr=self.lr, dataloaders=dataloaders,
                               epoches=self.epoches, dropout=self.dropout, weight_decay=self.weight_decay,
                               mode=self.mode, model_name=self.model_name, event_num=self.event_num,
                               epoch_stop=self.epoch_stop,
                               save_param_path=self.save_param_dir + self.data_type + "/" + self.model_name + "/",
                               writer=SummaryWriter(self.path_tensorboard))
            result = trainer.train()
            return result

        else:
            print("Not Available")
