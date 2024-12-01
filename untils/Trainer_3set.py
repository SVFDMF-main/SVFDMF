import copy
import json
import os
import time
import tqdm
from tqdm import tqdm
from metrics import *
from layers import *


class Trainer3():
    def __init__(self,
                 model,
                 device,
                 lr,
                 dropout,
                 dataloaders,
                 weight_decay,
                 save_param_path,
                 writer,
                 epoch_stop,
                 epoches,
                 mode,
                 model_name,
                 event_num,
                 save_threshold=0.0,
                 start_epoch=0,
                 ):

        self.model = model

        self.device = device
        self.mode = mode
        self.model_name = model_name
        self.event_num = event_num

        self.dataloaders = dataloaders
        self.start_epoch = start_epoch
        self.num_epochs = epoches
        self.epoch_stop = epoch_stop
        self.save_threshold = save_threshold
        self.writer = writer

        if os.path.exists(save_param_path):
            self.save_param_path = save_param_path
        else:
            self.save_param_path = os.makedirs(save_param_path)
            self.save_param_path = save_param_path

        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout

        self.criterion = nn.CrossEntropyLoss()

    def train(self):

        since = time.time()

        self.model.cuda()

        best_model_wts_val = copy.deepcopy(self.model.state_dict())
        best_acc_val = 0.0
        best_epoch_val = 0

        is_earlystop = False

        last_save_path = ''
        val_accs = []
        accs = []
        for epoch in range(self.start_epoch, self.start_epoch + self.num_epochs):
            if is_earlystop:
                break
            print('-' * 50)
            print('Epoch {}/{}'.format(epoch + 1, self.start_epoch + self.num_epochs))
            print('-' * 50)

            #更新学习率
            p = float(epoch) / 100
            lr = self.lr / (1. + 10 * p) ** 0.75
            #优化器
            self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=lr)

            for phase in ['train', 'val', 'test']:
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()
                print('-' * 10)
                print(phase.upper())
                print('-' * 10)

                running_loss_fnd = 0.0
                running_loss = 0.0
                tpred = []
                tlabel = []


                for batch in tqdm(self.dataloaders[phase]):
                    batch_data = batch
                    for k, v in batch_data.items():
                        batch_data[k] = v.cuda()
                    label = batch_data['label']
                    # self.optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs, fea = self.model(**batch_data)
                        _, preds = torch.max(outputs, 1)
                        loss = self.criterion(outputs, label)

                        if phase == 'train':
                            self.optimizer.zero_grad()
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                            self.optimizer.step()


                    tlabel.extend(label.detach().cpu().numpy().tolist())
                    tpred.extend(preds.detach().cpu().numpy().tolist())
                    running_loss += loss.item() * label.size(0)

                epoch_loss = running_loss / len(self.dataloaders[phase].dataset)
                print('Loss: {:.4f} '.format(epoch_loss))
                results = metrics(tlabel, tpred)
                print(results)
                self.writer.add_scalar('Loss/' + phase, epoch_loss, epoch + 1)
                self.writer.add_scalar('Acc/' + phase, results['acc'], epoch + 1)
                self.writer.add_scalar('F1/' + phase, results['f1'], epoch + 1)
                if phase == 'test':
                    accs.append(results['acc'])
                if phase == 'val':
                    val_accs.append(results['acc'])
                # start
                if phase == 'test' and results['acc'] > best_acc_val:
                    best_acc_val = results['acc']
                    best_model_wts_val = copy.deepcopy(self.model.state_dict())
                    best_epoch_val = epoch + 1
                    if best_acc_val > self.save_threshold:
                        if os.path.exists(last_save_path):
                            print('删除上一个检查点')
                            os.remove(last_save_path)
                        save_path = self.save_param_path + "_test_epoch" + str(best_epoch_val) + "_{0:.4f}".format(
                            best_acc_val)
                        torch.save(self.model.state_dict(), save_path)
                        last_save_path = save_path
                        print("saved " + self.save_param_path + "_test_epoch" + str(
                            best_epoch_val) + "_{0:.4f}".format(best_acc_val))

                    # 早停机制
                    # if epoch - best_epoch_val >= self.epoch_stop - 1:
                    # is_earlystop = True
                    # print("Early stopping triggered.")
                # end

        f = open("results.txt", "w")
        f.write("Validation Accuracies: " + str(val_accs) + "\n")
        f.write("Test Accuracies: " + str(accs) + "\n")
        f.close()
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print("Best model on val: epoch" + str(best_epoch_val) + "_" + str(best_acc_val))


        self.model.load_state_dict(best_model_wts_val)

        print("test result when using best model on val")
        return self.test()

    def test(self):
        since = time.time()

        self.model.cuda()
        self.model.eval()

        pred = []
        label = []


        for batch in tqdm(self.dataloaders['test']):
            with torch.no_grad():
                batch_data = batch
                for k, v in batch_data.items():
                    batch_data[k] = v.cuda()
                batch_label = batch_data['label']

                batch_outputs, fea = self.model(**batch_data)

                _, batch_preds = torch.max(batch_outputs, 1)

                label.extend(batch_label.detach().cpu().numpy().tolist())
                pred.extend(batch_preds.detach().cpu().numpy().tolist())

        print(get_confusionmatrix_fnd(np.array(pred), np.array(label)))
        print(metrics(label, pred))


        return metrics(label, pred)

