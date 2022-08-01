import pandas as pd
import pdb
import torch
import numpy as np
import torch.utils.data as Data
from torch.utils.data import IterableDataset
from tqdm import tqdm, trange
import os
import sys
import torch.nn as nn
from sklearn.metrics import roc_auc_score
import nmslib as nms
import copy
import torch.nn.functional as F
from xgboost import XGBClassifier
sys.path.append('.')

class criteo_iter_dataset(IterableDataset):
    def __init__(self, path):
        self.file_path = path
        self.file = pd.read_csv(self.file_path, chunksize = 100000)
         
    def __iter__(self):
        for chunk in self.file:
            labels = chunk['label']
            all_dense_features = chunk[chunk.columns.tolist()[1:14]].fillna(999)
            yield torch.FloatTensor(np.array(all_dense_features)), torch.Tensor(np.array(labels))

class mlp(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = 64
        self.baseline_mlp = nn.Sequential(nn.Linear(self.input_size, self.hidden_size), nn.ReLU(),
                                          nn.Linear(self.hidden_size, self.hidden_size // 2), nn.ReLU(),
                                          nn.Linear(self.hidden_size // 2, 1), nn.Sigmoid())

    def forward(self, x):
        return self.baseline_mlp(x)


class wide_deep(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size  = input_size
        self.hidden_size = 64
        self.emb = nn.Linear(self.input_size, self.input_size)
        self.dnn = nn.Sequential(nn.Linear(6, self.hidden_size), nn.BatchNorm1d(self.hidden_size), nn.ReLU(), 
                                 nn.Linear(self.hidden_size, self.hidden_size // 2), nn.BatchNorm1d(self.hidden_size // 2), nn.ReLU(), 
                                 nn.Linear(self.hidden_size // 2, self.hidden_size))
        self.wide = nn.Linear(7, self.hidden_size)
        self.classifier = nn.Sequential(nn.Linear(self.hidden_size, 1), nn.Sigmoid())
        
    def nn(self, batch_points, query):
        ids, distances = batch_points.knnQuery(query, k = self.input_size)
        # pdb.set_trace()
        return ids.tolist()[-6:], distances.tolist()[-6:]
        
    def forward(self, x):
        initial_embedding = self.emb(x)
        index = nms.init(method = 'hnsw', space = 'cosinesimil')
        
        batch_data_points = np.array(copy.deepcopy(initial_embedding.permute(1, 0).detach().numpy())).astype(np.float32)
        index.addDataPointBatch(batch_data_points)
        index.createIndex({'post':2}, print_progress = False)
        
        all_pair_list = []
        use_first_match = 1
        for _ in range(self.input_size):
            current_target_id, __ = self.nn(index, np.array(copy.deepcopy(initial_embedding[:, _].detach().numpy())))
            current_pair_list = [_] + current_target_id
            if use_first_match:
                all_pair_list = current_pair_list; break
            if current_pair_list not in all_pair_list and current_pair_list[::-1] not in all_pair_list:
                all_pair_list.append(current_pair_list)
       
        # pdb.set_trace()
        all_pair_list.sort()
        wide_side = x.permute(1, 0)[all_pair_list].permute(1, 0)
        deep_side = x.permute(1, 0)[list(set([___ for ___ in range(13)]) - set(all_pair_list))].permute(1, 0)
        

        dnn_emb = self.dnn(deep_side)
        wide_emb = self.wide(wide_side)
        all_emb = F.glu(torch.cat((dnn_emb, wide_emb), dim = -1))

        # return all_emb     # prediction
        # pdb.set_trace()
        return self.classifier(all_emb)  # pre-train
    
def train_nn():
    csv_data_path = './data/criteo_sample_50w.csv'
    n_epochs = 50
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model_name = 'improved_wide_deep'
    model = wide_deep(input_size = 13).to(device)
    
    optimizer_adam = torch.optim.Adam(model.parameters(), lr = 0.001) # , weight_decay = 1e-4)
    scheduler_adam = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_adam, T_0 = 5, T_mult = 1)

    optimizer_sgd = torch.optim.SGD(model.parameters(), lr = 0.001)
    scheduler_sgd = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_sgd, T_0 = 5, T_mult = 1)

    loss_fcn = nn.BCELoss()
    save_root_path = './ret'

    for current_epoch in trange(n_epochs):

        if current_epoch < 5:
            optimizer = optimizer_adam
            scheduler = scheduler_adam
        else:
            optimizer = optimizer_sgd
            scheduler = scheduler_sgd

        train_loss = []
        all_eval_auc = []
        iter_dataset = criteo_iter_dataset(csv_data_path)
        for features, labels in iter_dataset:
            for _ in range(features.shape[1]):
                max_value = features[:, _].max()
                min_value = features[:, _].min()
                features[:, _] = (features[:, _] - min_value) / (max_value - min_value)
            
            split_point = 4 * labels.shape[0] // 5
            all_train_data, all_eval_data = torch.cat((features[:split_point], labels[:split_point].unsqueeze(dim = -1)), dim = -1), torch.cat((features[split_point:], labels[split_point:].unsqueeze(dim = -1)), dim = -1)
            train_set = Data.TensorDataset(all_train_data[:, :-1], all_train_data[:, -1])
            train_loader = Data.DataLoader(train_set, batch_size = 256, shuffle = True)
            eval_set = Data.TensorDataset(all_eval_data[:, :-1], all_eval_data[:, -1])
            eval_loader = Data.DataLoader(eval_set, batch_size = 256, shuffle = False)

            model.train()
            for idx, current_batch in enumerate(train_loader):
                features, labels = current_batch[0].to(device), current_batch[1].to(device)
                preds = model(features)
                # pdb.set_trace()
                loss = loss_fcn(preds.view(-1), labels.view(-1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss.append(loss.cpu().item())
            
            with torch.no_grad():
                model.eval()
                all_eval_preds = []
                all_eval_labels = []
                for eval_idx, eval_current_batch in enumerate(eval_loader):
                   current_eval_preds = model(eval_current_batch[0].to(device))
                   all_eval_preds  += current_eval_preds.tolist()
                   all_eval_labels += eval_current_batch[1].tolist()
                eval_auc_score = roc_auc_score(all_eval_labels, all_eval_preds)
                all_eval_auc.append(eval_auc_score)

        print ('current epoch: ' + str(current_epoch) + ' | ' + 'train BCE loss: ' + str(np.mean(train_loss)) + ' | ' + 'eval auc score: ' + str(np.mean(all_eval_auc)))
        scheduler.step()
        if current_epoch % 2 == 0:
            if not os.path.exists(save_root_path):
                os.makedirs(save_root_path)
            torch.save(model.state_dict(), save_root_path + '/' + str(model_name) + '_epoch_' + str(current_epoch) + '.pth')

def xgb_head():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    all_df_data = pd.read_csv('./data/criteo_sample_50w.csv')
    backbone_model = wide_deep(input_size = 13).to(device)
    pre_weights = torch.load('ret/improved_wide_deep_epoch_8.pth')
    pred_dict = {k: v for k, v in pre_weights.items() if "classifier" not in k}
    backbone_model.load_state_dict(pred_dict, strict = False)
    backbone_model.eval()
    
    n_samples = 100000
    split_point = 4 * n_samples // 5
    train_x, train_y, test_x, test_y = np.array(all_df_data[all_df_data.columns.tolist()[1:14]].fillna(999))[:n_samples][:split_point], np.array(all_df_data['label'])[:n_samples][:split_point], np.array(all_df_data[all_df_data.columns.tolist()[1:14]].fillna(999))[:n_samples][split_point:], np.array(all_df_data['label'])[:n_samples][split_point:]
    train_features, test_features = backbone_model(torch.Tensor(train_x)), backbone_model(torch.Tensor(test_x))
    # pdb.set_trace()
    
    xgb_head_model = XGBClassifier(learning_rate = 0.01, n_estimators = 10, max_depth = 5, min_child_weight = 1,
                                   gamma = 0., subsample = 1, colsample_btree = 1, 
                                   scale_pos_weight = 1, random_state = 27, silent = 0)
    xgb_head_model.fit(np.array(train_features.detach().numpy()).astype(np.float32), train_y)
    y_pred = xgb_head_model.predict(test_features.detach().numpy())
    final_eval_auc = roc_auc_score(test_y, y_pred)
    print ('final auc score: ' + str(final_eval_auc))

if __name__ == '__main__':
    train_nn()
    # xgb_head()
