import torch
import random
import numpy as np
from tqdm import tqdm
import librosa
from util import linear_interpolation, MOUTH_BS
from transformers import HubertModel,Wav2Vec2FeatureExtractor
import pandas as pd
import time
import os
from torch_augmentation import augmentation

class EarlyStopping():
    def __init__(self, model_name, best_score=-np.Inf, patience=4, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = best_score
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.model_name= model_name
    def __call__(self,val_loss,model,optimizer,path):
        if self.verbose:
            print("val_loss={}".format(val_loss))
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss,model,optimizer,path)
        elif score < self.best_score+self.delta:
            self.counter+=1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter>=self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss,model,optimizer,path)
            self.counter = 0
    def save_checkpoint(self,val_loss,model,optimizer,path):
        if self.verbose:
            print(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss,
        }, path+'/'+self.model_name+'.pth')
        #torch.save(model.state_dict(), path+'/'+model_name+'.pth')
        self.val_loss_min = val_loss

def load_base_model(model_path,model_type, device):
    if model_type=="Hubert":
        model = HubertModel.from_pretrained(model_path)
        model_fps = 50
    else:
        raise NameError
    model.to(device)
    return model, model_fps

def speed_numpy(samples, speed):
    samples = samples.copy()  # frombuffer()导致数据不可更改因此使用拷贝
    data_type = samples[0].dtype
    old_length = samples.shape[0]
    new_length = int(old_length / speed)
    old_indices = np.arange(old_length)  # (0,1,2,...old_length-1)
    new_indices = np.linspace(start=0, stop=old_length, num=new_length)  # 在指定的间隔内返回均匀间隔的数字
    samples = np.interp(new_indices, old_indices, samples)  # 一维线性插值
    samples = samples.astype(data_type)
    return samples

def preprocessing_input(x, y, base_model, model_fps):
    assert len(x.shape)==3 and x.shape[1]==1, "Expected 3D shape but get "+str(x.shape)
    x = x.squeeze(1)
    base_model.eval()
    last_hidden_state = base_model(x).last_hidden_state
    # 1, length, 1024
    x = linear_interpolation(last_hidden_state, input_fps=model_fps, output_fps=60, output_len=y.shape[-2])
    return x, y

def speed_changing(x,y):
    speed = random.uniform(1.0,1.5)
    sig = x.numpy()
    out_x = librosa.effects.time_stretch(sig, rate=speed)
    out_y = np.apply_along_axis(speed_numpy, 1, y.numpy(), speed=speed)
    return torch.from_numpy(out_x), torch.from_numpy(out_y)


def train(args, model, dataset, criterion, optimizer, device, current_loss):
    train_loss_list = []
    valid_loss_list = []
    early_stopping = EarlyStopping(model_name=args.model_name, best_score=-current_loss, patience=args.patience)
    
    base_model,fps = load_base_model(args.base_model_path, "Hubert", args.device)
    for epoch in range(args.epochs):
        with tqdm(total=len(dataset["Train"]) + len(dataset["Valid"])) as t:
            model.train()
            train_loss = []
            for j, (x_train,y_train) in enumerate(dataset["Train"]):
                # torch.Size([batch, 1, audio_length]) torch.Size([batch, csv_length, bs_dim])
                model.reset_hidden_cell()
                if args.augmentation:
                    x_train,y_train = speed_changing(x_train,y_train)
                    x_train = augmentation(x_train, args.sampling_rate)
                xs = torch.split(x_train, args.sampling_rate*args.max_audio_length, dim=2)
                ys = torch.split(y_train, 60*args.max_audio_length, dim=1)
                outputs = []
                for inputs,labels in zip(xs,ys):
                    
                    inputs = inputs.to(device=device,dtype=torch.float32)
                    labels = labels.to(device=device,dtype=torch.float32)
                    inputs, labels = preprocessing_input(inputs, labels, base_model, fps)
                    # torch.Size([batch，csv_length, feature_dim]) torch.Size([batch，csv_length, bs_dim])
                    assert(labels.shape[0]==args.batch_size and labels.shape[2]==31 and labels.shape[1]==inputs.shape[1])
                    outputs.append(model(inputs))
                    torch.cuda.empty_cache()
                outputs = torch.cat(outputs,dim=1)
                optimizer.zero_grad()
                loss = criterion(outputs, y_train.to(device=device,dtype=torch.float32))
                loss.backward()
                optimizer.step()

                train_loss.append(loss.cpu().detach().numpy())
                torch.cuda.empty_cache()
                # Description will be displayed on the left
                if j%10==0:
                    t.set_description('Epoch %i training' % epoch)
                    t.set_postfix(train_loss=np.mean(train_loss))
                t.update(1)
            train_loss_list.append(np.mean(train_loss))
            torch.cuda.empty_cache()

            model.eval()
            valid_loss = []
            for j, (x_valid, y_valid) in enumerate(dataset["Valid"]):
                model.reset_hidden_cell()
                xs = torch.split(x_valid, args.sampling_rate*args.max_audio_length, dim=2)
                ys = torch.split(y_valid, 60*args.max_audio_length, dim=1)
                outputs = []
                for inputs,labels in zip(xs,ys):
                    inputs = inputs.to(device=device,dtype=torch.float32)
                    labels = labels.to(device=device,dtype=torch.float32)
                    inputs, labels = preprocessing_input(inputs, labels, base_model, fps)
                    with torch.no_grad():
                        outputs.append(model(inputs))
                    torch.cuda.empty_cache()
                outputs = torch.cat(outputs,dim=1)
                loss = criterion(outputs, y_valid.to(device=device,dtype=torch.float32))
                valid_loss.append(loss.cpu().detach().numpy())
                    
                if j%20==0:
                    t.set_description('Epoch %i validation' % epoch)
                    t.set_postfix(train_loss=np.mean(train_loss), val_loss = np.mean(valid_loss))
                t.update(1)
            valid_loss_list.append(np.mean(valid_loss))
            t.set_description('Epoch %i' % epoch)
            t.set_postfix(train_loss=train_loss_list[-1], val_loss=valid_loss_list[-1])
            early_stopping(valid_loss_list[-1], model=model, optimizer=optimizer, path=args.model_path)
            if early_stopping.early_stop:
                break
    return train_loss_list, valid_loss_list

    
def np_to_csv(x, calibration):
    x=np.squeeze(x)
    if calibration:
        x -= x[0]
        x[x<0]=0
    pre_x = np.ones((x.shape[0],2))*31
    px = pd.DataFrame(np.concatenate([pre_x,x],axis=1),columns=["Timecode", "BlendShapeCount"]+MOUTH_BS)
    for i in range(x.shape[0]):
        h = int(i//216000)
        m = int(i//3600)-h*60
        s = int(i//60)-m*60
        f = int(i%60)
        timecode = '%02d:%02d:%02d:%02d.%d' % (h,m,s,f,i)
        px.iloc[i,0] = timecode
    return px
    
def inference(args, model, checkpoint_path, wav_lst, calibration):
    torch.cuda.empty_cache()
    t1 = time.time() # Load Processor
    processor = Wav2Vec2FeatureExtractor.from_pretrained(args.base_model_path)

    t2 = time.time()  # Load Hubert model
    base_model, fps = load_base_model(args.base_model_path, "Hubert", args.device)
    base_model.eval()
    

    t3 = time.time()  # Load LSTM model
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    t4 = time.time() 
    print("Load Processor: %dms" % (int(round((t2-t1) * 1000))))
    print("Load Base model (Hubert): %dms" % (int(round((t3-t2) * 1000))))
    print("Load LSTM model (LSTM): %dms\n" % (int(round((t4-t3) * 1000))))
    
    for wav in wav_lst:
        wav_path = os.path.join(args.test_data_path, wav)

        t5 = time.time() # Load audio
        sig, rate = librosa.load(wav_path, sr=args.sampling_rate)
        audio = processor(sig, return_tensors="pt", sampling_rate=rate).input_values # (1, audiolength)
        
        """chunk_audio_length = 16000*args.audio_section_length
        n_chunks = audio.shape[-1]//chunk_audio_length
        
        if audio.shape[-1] % chunk_audio_length != 0:
            n_chunks += 1
        chunks = []
        for i in range(n_chunks):
            chunks.append(audio[:,i*chunk_audio_length:min((i+1)*chunk_audio_length,len(audio[0]))])"""
        chunks = torch.split(audio,args.audio_section_length*args.sampling_rate, dim=1)
        output = []

        base_model_infer_time = 0
        lstm_model_infer_time = 0

        model.reset_hidden_cell()
        for chunk in chunks:
            
            x = torch.FloatTensor(chunk).to(device=args.device)

            t6 = time.time() # Base model inference
            with torch.no_grad():
                last_h_state = base_model(x).last_hidden_state
            x = linear_interpolation(last_h_state, input_fps=fps, output_fps=60)
            t7 = time.time() # LSTM inference
            with torch.no_grad():
                x = model(x).detach().cpu().numpy()
            #print(model.hidden_cell[0].shape,model.hidden_cell[1].shape)
            t8 = time.time()
            
            base_model_infer_time += t7-t6
            lstm_model_infer_time += t8-t7
            output.append(x)
            del x
            del chunk
            del last_h_state
            torch.cuda.empty_cache()
        t9 = time.time()
        x = np.concatenate(output,axis=1)
        px = np_to_csv(x, calibration=calibration)
        px.to_csv(wav_path[:-4]+"_"+args.model_name+".csv",index=False)
        t10 = time.time() # output to csv
        torch.cuda.empty_cache()
        print("Audio length: %ds" % (audio.shape[-1]/16000))
        print("Load and process audio: %dms" % (int(round((t6-t5) * 1000))))
        print("Base model inference (feature extraction): %dms" % (int(round((base_model_infer_time) * 1000))))
        print("LSTM model inference: %dms" % (int(round((lstm_model_infer_time) * 1000))))
        print("Output blendshape to csv: %dms\n" % (int(round((t10-t9) * 1000))))

def test(args, model, checkpoint_path, dataset, criterion):
    base_model,fps = load_base_model(args.base_model_path, "Hubert", args.device)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    with tqdm(total=len(dataset["Train"]) + len(dataset["Valid"])+ len(dataset["Test"])) as t:
        model.eval()
        train_loss = []
        valid_loss = []
        test_loss = []
        for j, (inputs, labels) in enumerate(dataset["Train"]):
            inputs = inputs.to(device=args.device,dtype=torch.float32)
            labels = labels.to(device=args.device,dtype=torch.float32)
            inputs, labels = preprocessing_input(inputs, labels, base_model, fps)
            with torch.no_grad():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            train_loss.append(loss.cpu().detach().numpy())
            if j%20==0:
                t.set_description('Train')
                t.set_postfix(train_loss=np.mean(train_loss))
            t.update(1)
            torch.cuda.empty_cache()
        t.set_postfix(train_loss=np.mean(train_loss))
        for j, (inputs, labels) in enumerate(dataset["Valid"]):
            inputs = inputs.to(device=args.device,dtype=torch.float32)
            labels = labels.to(device=args.device,dtype=torch.float32)
            inputs, labels = preprocessing_input(inputs, labels, base_model, fps)
            with torch.no_grad():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            valid_loss.append(loss.cpu().detach().numpy())
            if j%20==0:
                t.set_description('Valid')
                t.set_postfix(valid_loss=np.mean(valid_loss))
            t.update(1)
            torch.cuda.empty_cache()
        t.set_postfix(valid_loss=np.mean(valid_loss))
        for j, (inputs, labels) in enumerate(dataset["Test"]):
            inputs = inputs.to(device=args.device,dtype=torch.float32)
            labels = labels.to(device=args.device,dtype=torch.float32)
            inputs, labels = preprocessing_input(inputs, labels, base_model, fps)
            with torch.no_grad():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            test_loss.append(loss.cpu().detach().numpy())
            if j%20==0:
                t.set_description('Valid')
                t.set_postfix(vtest_loss=np.mean(test_loss))
            t.update(1)
            torch.cuda.empty_cache()
        t.set_postfix(train_loss=np.mean(train_loss),valid_loss=np.mean(valid_loss),test_loss=np.mean(test_loss))
    return train_loss, valid_loss, test_loss