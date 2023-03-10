from model.informer import Informer
from model.lstm import LSTM
from arguments import *
from dataset import get_dataloaders, create_dataloaders
import os
from util import EarlyStopping, linear_interpolation, np_to_csv, Weighted_MSE, MOUTH_BS_WEIGHT
from augmentation import *
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam, SGD
from transformers import HubertModel, Wav2Vec2FeatureExtractor

class InvalidArguments(Exception):
    def __init__(self, message) -> None:
        self.message = message
        super().__init__(message)


class Runner():
    def __init__(self,args) -> None:
        # data
        self.mode = args.mode
        if self.mode=="train" or self.mode=="test":
            if args.create_dataset:
                create_dataloaders(args)
            self.dataset = get_dataloaders(args.dataset_path)
        else:
            self.dataset = None
        self.sampling_rate = args.sampling_rate
        self.output_fps = 60

        # base_model
        self.device = args.device
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(args.base_model_path)
        self.base_model, self.base_fps = self._load_base_model(args.base_model_path, "Hubert", args.device)

        # model
        self.model_name = args.model_name
        self.model_type = self._get_model_type(self.model_name)
        self.model_save_path = os.path.join(args.model_path, self.model_type, self.model_name+".pth")
        self.checkpoint = torch.load(self.model_save_path) if os.path.exists(self.model_save_path) else None
        self.model = self._load_model(args)
        self.current_loss = self.checkpoint["loss"] if self.checkpoint is not None else 999999
        print("current loss", self.current_loss)

        # optimizer
        if self.mode == "train":
            self.epochs = args.epochs
            self.optimizer = self._load_optimizer(args.optimizer, args.load_optimizer, args.learning_rate)
            self.schedular = self._load_schedular(args.schedular)
            self.early_stopping = EarlyStopping(model_name=self.model_name, best_score=-self.current_loss, patience=args.patience)
        else:
            self.optimizer = None
            self.schedular = None
            self.early_stopping = None
        
        # critirion
        if self.mode == "test" or self.mode=="train":
            self.criterion = self._load_criterion(args.criterion, self.device)
        else:
            self.criterion = None

        # augmentation
        self.augmentation = args.augmentation
        self.split = True

        # split
        self.frac = 20   # 最小音频切分单元，也就是1/20秒的音频
        self.max_num_frac = args.audio_section_length * self.frac    # 切分的最大fraction数，比如1s音频就是20个fraction
        self.min_num_frac = 10                                       # 切分的最小fraction数，默认0.5s，10个fraction
        self.mid_num_frac = (self.max_num_frac+self.min_num_frac)//2 # 验证用的fraction数

        

        # test
        self.test_data_path = args.test_data_path
  

    def train(self):
        assert(self.mode=="train")
        train_loss_list = []
        valid_loss_list = []
        
        for epoch in range(self.epochs):
            with tqdm(total=len(self.dataset["Train"]) + len(self.dataset["Valid"])) as t:
                self.model.train()
                train_loss = []
                for j, (x_train, y_train) in enumerate(self.dataset["Train"]):
                    # torch.Size([batch, 1, audio_length]) torch.Size([batch, csv_length, bs_dim])
                    self.model.reset_hidden_cell()
                    if self.augmentation:
                        x_train, y_train = self._augmentation(x_train,y_train)
                    x_train, y_train = self._split(x_train, y_train, truncated=True, padded=True)
                    y_pred = []
                    y_true = []
                    for inputs,labels in zip(x_train, y_train):
                        inputs = inputs.to(device=self.device,dtype=torch.float32)
                        labels = labels.to(device=self.device,dtype=torch.float32)
                        inputs, labels = self._preprocessing_input(inputs, labels)
                        # x: batch, frames, 1024
                        # y: batch, frames, 31
                        y_pred.append(self.model(inputs))
                        y_true.append(labels)
                        del inputs
                        del labels
                        torch.cuda.empty_cache()
                    
                    y_pred = torch.cat(y_pred, dim=1)
                    y_true = torch.cat(y_true, dim=1)
                    self.optimizer.zero_grad()
                    loss = self.criterion(y_pred, y_true)
                    loss.backward()
                    self.optimizer.step()

                    train_loss.append(loss.cpu().detach().numpy())

                    del loss
                    del y_pred
                    del y_true
                    torch.cuda.empty_cache()
                    # Description will be displayed on the left
                    if j%20==0:
                        t.set_description('Epoch %i training' % epoch)
                        t.set_postfix(train_loss=np.mean(train_loss))
                    t.update(1)
                train_loss_list.append(np.mean(train_loss))
                torch.cuda.empty_cache()
                

                self.model.eval()
                valid_loss = []
                for j, (x_valid, y_valid) in enumerate(self.dataset["Valid"]):
                    self.model.reset_hidden_cell()
                    x_valid, y_valid = self._split(x_valid, y_valid, truncated=False, padded=True)
                    y_pred = []
                    y_true = []
                    for inputs,labels in zip(x_valid, y_valid):
                        inputs = inputs.to(device=self.device,dtype=torch.float32)
                        labels = labels.to(device=self.device,dtype=torch.float32)
                        inputs, labels = self._preprocessing_input(inputs, labels)

                        with torch.no_grad():
                            y_pred.append(torch.clamp(self.model(inputs),0,1))
                            y_true.append(labels)
                        del inputs
                        del labels
                        torch.cuda.empty_cache()
                    y_pred = torch.cat(y_pred, dim=1)
                    y_true = torch.cat(y_true, dim=1)
                    loss = self.criterion(y_pred, y_true)
                    valid_loss.append(loss.cpu().detach().numpy())

                    del loss
                    del y_pred
                    del y_true
                    torch.cuda.empty_cache()

                    if j%10==0:
                        t.set_description('Epoch %i validation' % epoch)
                        t.set_postfix(train_loss=np.mean(train_loss), val_loss = np.mean(valid_loss))
                    t.update(1)
                mean_val_loss = np.mean(valid_loss)
                valid_loss_list.append(mean_val_loss)
                t.set_description('Epoch %i' % epoch)
                t.set_postfix(train_loss=train_loss_list[-1], val_loss=mean_val_loss, lr=self._get_lr())
                if self.schedular is not None:
                    self.schedular.step(mean_val_loss)
                if self.early_stopping is not None:
                    self.early_stopping(mean_val_loss, model=self.model, optimizer=self.optimizer, path=self.model_save_path)
                    if self.early_stopping.early_stop:
                        break
                del train_loss
                del valid_loss

    def inference(self):
        assert(self.mode=="eval")
        wav_path_list = self._get_test_wavs(self.test_data_path)
        for wav_path in tqdm(wav_path_list):
            self.model.reset_hidden_cell()
            audio = self._preprocessing_audio(wav_path)
            split_size = int(self.mid_num_frac/self.frac * self.sampling_rate)
            chunks = torch.split(audio, split_size, dim=1)
            output = []
            for chunk in chunks:
                print(chunk.shape)
                if chunk.shape[1]<640:
                    break
                x = torch.FloatTensor(chunk).to(device=self.device)
                x, _ = self._preprocessing_input(x)
                with torch.no_grad():
                    x = torch.clamp(self.model(x),0,1).detach().cpu().numpy()
                
                output.append(x)
                torch.cuda.empty_cache()
            x = np.concatenate(output,axis=1)
            px = np_to_csv(x, calibration=False)
            px.to_csv(wav_path[:-4]+"_"+self.model_name+".csv",index=False)
            torch.cuda.empty_cache()

    def test(self):
        pass
    
    def run(self):
        if self.mode=="train":
            self.train()
        elif self.mode=="eval":
            self.inference()
        elif self.mode=="test":
            self.test()
        
    def _get_lr(self):
        return self.optimizer.state_dict()['param_groups'][0]['lr']

    def _get_model_type(self,model_name):
        if "lstm" in model_name.lower():
            return "LSTM"
        if "informer" in model_name.lower():
            return "Informer"
        return "Unknown"

    def _load_model(self,args):
        if "LSTM" in self.model_name:
            args = get_LSTM_args(args)
            model = LSTM(args)
        elif "Informer" in self.model_name:
            args = get_informer_args(args)
            model = Informer(args)
        else:
            raise InvalidArguments("def _load_model"+str(self.model_name))
        if args.load_model and self.checkpoint is not None:
            model.load_state_dict(self.checkpoint['model_state_dict'])
        model.to(args.device)
        return model
    
    def _load_base_model(self, model_path, model_type, device):
        if model_type=="Hubert":
            model = HubertModel.from_pretrained(model_path)
            model_fps = 49
        else:
            raise NameError
        model.to(device)
        model.eval()
        return model, model_fps

    def _load_optimizer(self, optimizer_name:str, load_checkpoint:bool, lr:float, reset_lr:bool=True):
        if optimizer_name == "Adam":
            optimizer=Adam(self.model.parameters(), lr=lr)
        else:
            optimizer=SGD(self.model.parameters(), lr=lr)
        
        if load_checkpoint and self.checkpoint is not None:
            optimizer.load_state_dict(self.checkpoint["optimizer_state_dict"])
            if reset_lr:
                for g in optimizer.param_groups:
                    g['lr'] = lr
        return optimizer

    def _load_schedular(self, schedular_name:str)->object:
        if schedular_name == "ReduceLROnPlateau":
            schedular=ReduceLROnPlateau(self.optimizer, patience=2, mode='min', min_lr=1e-6, factor=0.6)
        else:
            raise InvalidArguments("Invalid schedular name:",str(schedular_name))
        return schedular
    
    def _load_criterion(self, criterion_name:str, device:str):
        if criterion_name == "MSE":
            criterion = torch.nn.MSELoss()
        elif criterion_name == "WMSE":
            criterion = Weighted_MSE(MOUTH_BS_WEIGHT)
        else:
            raise InvalidArguments("Invalid Criterion name")
        criterion.to(device)
        return criterion

    def _augmentation(self, x_train, y_train):
        # torch.Size([batch, 1, audio_length]) torch.Size([batch, csv_length, bs_dim])
        x_train,y_train = speed_changing(x_train,y_train)
        x_train = augmentation(x_train, self.sampling_rate)
        return x_train, y_train
    
    def _split(self, x_train, y_train, truncated=True, padded=True):
        # x: batch, 1, audio_length
        # y: batch, frames, 31
        split_size = random.randint(self.min_num_frac, self.max_num_frac) / 20
        split_size_x = int(self.sampling_rate * split_size)
        split_size_y = int(self.output_fps    * split_size)
        xs = list(torch.split(x_train, split_size_x, dim=2)) # x: batch, 1, split_size_x
        ys = list(torch.split(y_train, split_size_y, dim=1)) # y: batch, split_size_y, 31
        if len(xs)-len(ys)==1:
            xs.pop(-1)
        assert len(xs)==len(ys), str(len(xs))+" "+str(len(ys))+" "+str(xs[-1].shape)
        start_i = random.randint(0,min(0,len(xs)//2-2)) if truncated else 0
        end_i = random.randint(max(len(xs)//2+2,len(xs)),len(xs)) if truncated else len(xs)
        if padded:
            last_x_shape = xs[-1].shape
            last_y_shape = ys[-1].shape
            x_padded_length = split_size_x - last_x_shape[2]
            y_padded_length = split_size_y - last_y_shape[1]
            x_padded = torch.zeros((last_x_shape[0], last_x_shape[1], x_padded_length))
            y_padded = torch.zeros((last_y_shape[0], y_padded_length, last_y_shape[2]))
            xs[-1] = torch.cat((xs[-1], x_padded), dim=2)
            ys[-1] = torch.cat((ys[-1], y_padded), dim=1)

        return xs[start_i:end_i], ys[start_i:end_i]

    def _preprocessing_input(self, x, y=None):
        
        
        if len(x.shape)==3 and x.shape[1]==1:
            x = x.squeeze(1)
        # x: batch, audio_length
        # y: batch, frames, 31
        with torch.no_grad():
            last_hidden_state = self.base_model(x).last_hidden_state
        # 1, length, 1024
        out_len = y.shape[-2] if y is not None else None
        x = linear_interpolation(last_hidden_state, input_fps=self.base_fps, output_fps=self.output_fps, output_len=out_len)
        return x, y

    def _preprocessing_audio(self, wav_path):
        sig, rate = librosa.load(wav_path, sr=self.sampling_rate)
        audio = self.processor(sig, return_tensors="pt", sampling_rate=rate).input_values # (1, audiolength)
        return audio

    def _get_test_wavs(self, test_data_path:str):
        files = os.listdir(test_data_path)
        wav_paths = []
        for f in files:
            if f.endswith(".wav"):
                wav_paths.append(os.path.join(test_data_path, f))
        return wav_paths