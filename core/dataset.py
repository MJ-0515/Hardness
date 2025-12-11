import pickle
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer  #把原始文本转成token ids（input_ids、attention_mask等）

__all__ = ['MMDataLoader']


class MMDataset(Dataset):
    def __init__(self, args, mode='train'):
        self.mode = mode
        self.train_mode = args['base']['train_mode'] #regression
        self.datasetName = args['dataset']['datasetName']
        self.dataPath = args['dataset']['dataPath']
        self.missing_rate_eval_test = args['base']['missing_rate_eval_test'] #0.5
        self.missing_seed = args['base']['seed']  #1112
        self.token_length = args['model']['feature_extractor']['input_length'][0] #50
        #tokenizer 由预训练名（如 bert-base-uncased）实例化，用于把 rawText → input_ids/attention_mask
        self.tokenizer = AutoTokenizer.from_pretrained(args['model']['feature_extractor']['bert_pretrained'])
        DATA_MAP = {
            'mosi': self.__init_mosi,
            'mosei': self.__init_mosei,
            'sims': self.__init_sims,
        }
        DATA_MAP[self.datasetName]()

    def __init_mosi(self):
        with open(self.dataPath, 'rb') as f:
            data = pickle.load(f)

        self.data = data

        self.text = data[self.mode]['text_bert'].astype(np.float32)
        self.vision = data[self.mode]['vision'].astype(np.float32)
        self.audio = data[self.mode]['audio'].astype(np.float32)

        self.rawText = data[self.mode]['raw_text']

        for i in range(len(self.rawText)):
            dic = self.tokenizer(self.rawText[i], padding="max_length", truncation=True, max_length=self.token_length,
                                 return_tensors="pt")
            input_id = dic['input_ids'] # shape: [1, L]
            attention_mask = dic['attention_mask'] # shape: [1, L]
            self.text[i, 0, :] = input_id  
            self.text[i, 1, :] = attention_mask  # 第 3 路 self.text[i, 2, :]（segment ids）保持为原 pkl 内的值
        self.ids = data[self.mode]['id']
        self.labels = {
            'M': data[self.mode][self.train_mode + '_labels'].astype(np.float32),
            'missing_rate_l': np.zeros_like(data[self.mode][self.train_mode + '_labels']).astype(np.float32), #np.zeros_like(1284)
            'missing_rate_a': np.zeros_like(data[self.mode][self.train_mode + '_labels']).astype(np.float32), #[0:1284]
            'missing_rate_v': np.zeros_like(data[self.mode][self.train_mode + '_labels']).astype(np.float32),#[0:1284]
        }

        if self.datasetName == 'sims':
            for m in "TAV":
                self.labels[m] = data[self.mode][self.train_mode + '_labels_' + m]
        #音频/视觉用长度数组指明有效帧数，便于后面生成 mask。
        self.audio_lengths = data[self.mode]['audio_lengths']
        self.vision_lengths = data[self.mode]['vision_lengths']
        self.audio[self.audio == -np.inf] = 0  #-inf 清零，避免后续数值污染。
        #训练：为每个样本、每个模态随机设缺失率
        if self.mode == 'train':
            # 在训练时，为三个模态（l, a, v）生成缺失率向量. Generate a random matrix uniformly distributed within the interval of [0,1), with each sample having a random missing_rate.
            # 为每个模态创建一个形状为 [样本数, 1] 的随机向量，数值 ∈ [0, 1)，表示初始的“保留概率”或“模态激活强度”
            # range(3) 代表 L/A/V 三个模态
            # self.train_mode + '_labels'：regression_labels :确定样本总数
            # 每个模态的每个样本都拥有一个随机缺失率
            missing_rate = [np.random.uniform(size=(len(data[self.mode][self.train_mode + '_labels']), 1)) for i in
                            range(3)]
            for i in range(3):
                # 对每个模态，将其中一半样本的缺失率直接设为 0，表示这些样本在该模态是“完全不缺失”的
                # random.sample(...) 是无放回地随机选择样本下标
                # missing_rate[i][sample_idx] = 0 会将这些位置的随机数直接归 0
                # 将原始均匀分布“截断”成一半随机、一半 0，模拟训练过程中的随机模态缺失情况
                sample_idx = random.sample([i for i in range(len(missing_rate[i]))], int(len(missing_rate[i]) / 2))  #长度为642的list
                missing_rate[i][sample_idx] = 0
            
            # 将这三个模态的缺失率数组分别赋值，后续模型或 data loader 可以读取这些信息
            self.labels['missing_rate_l'] = missing_rate[0]
            self.labels['missing_rate_a'] = missing_rate[1]
            self.labels['missing_rate_v'] = missing_rate[2]

        else:#验证/测试：
            # If it is a dict, it means that the modality is missing
            if isinstance(self.missing_rate_eval_test, dict): #字典：精确控制模态是否可用（布尔）
                t_modality = self.missing_rate_eval_test['l']
                a_modality = self.missing_rate_eval_test['a']
                v_modality = self.missing_rate_eval_test['v']
                if t_modality:
                    self.labels['missing_rate_l'] = np.zeros((len(data[self.mode][self.train_mode + '_labels']), 1))
                else:
                    self.labels['missing_rate_l'] = np.ones((len(data[self.mode][self.train_mode + '_labels']), 1))
                if a_modality:
                    self.labels['missing_rate_a'] = np.zeros((len(data[self.mode][self.train_mode + '_labels']), 1))
                else:
                    self.labels['missing_rate_a'] = np.ones((len(data[self.mode][self.train_mode + '_labels']), 1))
                if v_modality:
                    self.labels['missing_rate_v'] = np.zeros((len(data[self.mode][self.train_mode + '_labels']), 1))
                else:
                    self.labels['missing_rate_v'] = np.ones((len(data[self.mode][self.train_mode + '_labels']), 1))
            else: #标量：所有样本、所有位置统一缺失率 r
                missing_rate = [
                    self.missing_rate_eval_test * np.ones((len(data[self.mode][self.train_mode + '_labels']), 1)) for i
                    in range(3)]
                self.labels['missing_rate_l'] = missing_rate[0]
                self.labels['missing_rate_a'] = missing_rate[1]
                self.labels['missing_rate_v'] = missing_rate[2]
        #调用三次生成：文本/音频/视觉的缺失版输入，以及长度/有效 mask/缺失 mask
        self.text_m, self.text_length, self.text_mask, self.text_missing_mask = self.generate_m(self.text[:, 0, :], # 只取 input_ids 这一路
                                                                                                self.text[:, 1, :], # attention_mask
                                                                                                None,
                                                                                                self.labels[
                                                                                                    'missing_rate_l'],
                                                                                                self.missing_seed,
                                                                                                mode='text')
        Input_ids_m = np.expand_dims(self.text_m, 1)
        Input_mask = np.expand_dims(self.text_mask, 1)
        Segment_ids = np.expand_dims(self.text[:, 2, :], 1)
        self.text_m = np.concatenate((Input_ids_m, Input_mask, Segment_ids), axis=1)

        self.audio_m, self.audio_length, self.audio_mask, self.audio_missing_mask = self.generate_m(self.audio, None,
                                                                                                    self.audio_lengths,
                                                                                                    self.labels[
                                                                                                        'missing_rate_a'],
                                                                                                    self.missing_seed,
                                                                                                    mode='audio')
        self.vision_m, self.vision_length, self.vision_mask, self.vision_missing_mask = self.generate_m(self.vision,
                                                                                                        None,
                                                                                                        self.vision_lengths,
                                                                                                        self.labels[
                                                                                                            'missing_rate_v'],
                                                                                                        self.missing_seed,
                                                                                                        mode='vision')

    def __init_mosei(self):
        return self.__init_mosi()

    def __init_sims(self):
        return self.__init_mosi()

    def generate_m(self, modality, input_mask, input_len, missing_rate, missing_seed, mode='text'):

        if mode == 'text':
            input_len = np.argmin(input_mask, axis=1)
        elif mode == 'audio' or mode == 'vision':
            input_mask = np.array([np.array([1] * length + [0] * (modality.shape[1] - length)) for length in input_len])
        np.random.seed(missing_seed)
        pad_id = self.tokenizer.pad_token_id
        missing_mask = (np.random.uniform(size=input_mask.shape) > missing_rate.repeat(input_mask.shape[1],
                                                                                       1)) * input_mask

        assert missing_mask.shape == input_mask.shape

        if mode == 'text':
            unk_id = self.tokenizer.unk_token_id

            # CLS SEG Token unchanged.
            for i, instance in enumerate(missing_mask):
                instance[0] = instance[input_len[i] - 1] = 1

            modality_m = missing_mask * modality + (unk_id * np.ones_like(modality)) * (
                        input_mask - missing_mask) + pad_id * (1 - input_mask)  # UNK token: 100 bert roberta
        elif mode == 'audio' or mode == 'vision':
            modality_m = missing_mask.reshape(modality.shape[0], modality.shape[1], 1) * modality

        return modality_m, input_len, input_mask, missing_mask

    def __len__(self):
        return len(self.labels['M'])

    def __getitem__(self, index):
        if (self.mode == 'train') and (index == 0):
            # missing_rate = [np.random.uniform(0, 0.5, size=(len(self.data[self.mode][self.train_mode+'_labels']), 1)) for i in range(3)]
            missing_rate = [np.random.uniform(size=(len(self.data[self.mode][self.train_mode + '_labels']), 1)) for i in
                            range(3)]

            for i in range(3):
                sample_idx = random.sample([i for i in range(len(missing_rate[i]))], int(len(missing_rate[i]) / 2))
                missing_rate[i][sample_idx] = 0

            self.labels['missing_rate_l'] = missing_rate[0]
            self.labels['missing_rate_a'] = missing_rate[1]
            self.labels['missing_rate_v'] = missing_rate[2]

            self.text_m, self.text_length, self.text_mask, self.text_missing_mask = self.generate_m(self.text[:, 0, :],
                                                                                                    self.text[:, 1, :],
                                                                                                    None,
                                                                                                    missing_rate[0],
                                                                                                    self.missing_seed,
                                                                                                    mode='text')
            Input_ids_m = np.expand_dims(self.text_m, 1)
            Input_mask = np.expand_dims(self.text_mask, 1)
            Segment_ids = np.expand_dims(self.text[:, 2, :], 1)
            self.text_m = np.concatenate((Input_ids_m, Input_mask, Segment_ids), axis=1)

            self.audio_m, self.audio_length, self.audio_mask, self.audio_missing_mask = self.generate_m(self.audio,
                                                                                                        None,
                                                                                                        self.audio_lengths,
                                                                                                        missing_rate[1],
                                                                                                        self.missing_seed,
                                                                                                        mode='audio')
            self.vision_m, self.vision_length, self.vision_mask, self.vision_missing_mask = self.generate_m(self.vision,
                                                                                                            None,
                                                                                                            self.vision_lengths,
                                                                                                            missing_rate[
                                                                                                                2],
                                                                                                            self.missing_seed,
                                                                                                            mode='vision')

        sample = {
            'text': torch.Tensor(self.text[index]),#原始文本
            'text_m': torch.Tensor(self.text_m[index]),  #缺失后的文本
            'audio': torch.Tensor(self.audio[index]),    # 原始音频
            'audio_m': torch.Tensor(self.audio_m[index]),  # 缺失后的音频
            'vision': torch.Tensor(self.vision[index]),     # 原始视觉
            'vision_m': torch.Tensor(self.vision_m[index]),   #缺失后的视觉
            'index': index,
            'id': self.ids[index],
            'labels': {k: torch.Tensor(v[index].reshape(-1)) for k, v in self.labels.items()}
        }  # labels:{'M'：主任务（主任务标签）,'missing_rate_l':标量缺失率,'missing_rate_a':标量缺失率,'missing_rate_v':标量缺失率}
        return sample


def MMDataLoader(args):
    datasets = {
        'train': MMDataset(args, mode='train'),
        'valid': MMDataset(args, mode='valid'),
        'test': MMDataset(args, mode='test')
    }

    dataLoader = {
        ds: DataLoader(datasets[ds],
                       batch_size=args['base']['batch_size'],
                       num_workers=args['base']['num_workers'],
                       shuffle=True if ds == 'train' else False)
        for ds in datasets.keys()
    }

    return dataLoader


def MMDataEvaluationLoader(args):
    datasets = MMDataset(args, mode='test')

    dataLoader = DataLoader(datasets,
                            batch_size=args['base']['batch_size'],
                            num_workers=args['base']['num_workers'],
                            shuffle=False)

    return dataLoader
