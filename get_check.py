import torch
from tqdm import tqdm

from model.SVMF import SVMFModel
from untils.tools import pretrain_bert_models, pretrain_bert_token
from untils.dataloader import SVMFDataset, collate_fn
from torch.utils.data import DataLoader
from untils.metrics import metrics, get_confusionmatrix_fnd

path = 'check_points/SVMF/SVMF/_test_epoch22_0.8782'

def load_checkpoint(path):
    """
    加载模型及权重
    """
    bert = pretrain_bert_models()
    model = SVMFModel(bert, dropout=0.1, fea_dim=128)
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint)
    model.eval()
    return model.cuda()

def get_dataloader():
    """
    创建测试集的 DataLoader
    """
    token = pretrain_bert_token()
    dataset_test = SVMFDataset('vid_time3_test.txt', token=token)
    test_dataloader = DataLoader(
        dataset_test, 
        batch_size=16,
        num_workers=0,
        pin_memory=True,
        shuffle=False,
        collate_fn=collate_fn
    )
    return test_dataloader

def test():
    """
    模型测试
    """
    model = load_checkpoint(path)
    test_dataloader = get_dataloader()
    tpred = []
    tlabel = []

    for batch in tqdm(test_dataloader, desc="Testing"):
        batch_data = batch
        # 将每个样本放到 GPU 上
        for k, v in batch_data.items():
            batch_data[k] = v.cuda()
        
        label = batch_data['label']
        
        with torch.no_grad():
            outputs, _ = model(**batch_data)  # outputs 是一个元组
            _, preds = torch.max(outputs, dim=1)  # 获取最大概率的分类结果

        tlabel.extend(label.cpu().numpy().tolist())
        tpred.extend(preds.cpu().numpy().tolist())
    
    # 计算指标
    results = metrics(tlabel, tpred)
    get_confusionmatrix_fnd(tpred, tlabel)
    print(results)

if __name__ == '__main__':
    test()
