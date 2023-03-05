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
from torch.optim.lr_scheduler import ReduceLROnPlateau

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
        model_fps = 49
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
    with torch.no_grad():
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
    for g in optimizer.param_groups:
        g['lr'] = args.learning_rate
    scheduler = ReduceLROnPlateau(optimizer, patience=2,mode='min',min_lr=1e-6, factor=0.6)
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
                frac = 1/20
                max_num_frac = args.max_audio_length*20
                split_size = random.randint(10, max_num_frac) * frac
                split_size_x = int(args.sampling_rate * split_size)
                split_size_y = int(60* split_size)
                xs = torch.split(x_train, split_size_x, dim=2)
                ys = torch.split(y_train, split_size_y, dim=1)
                outputs = []
                for inputs,labels in zip(xs,ys):
                    if labels.shape[1]<4:
                        continue
                    elif inputs.shape[2]<split_size_x:
                        n_frame = labels.shape[1]
                        assert(n_frame>3)
                        inputs = inputs[:,:,:min(n_frame*267,inputs.shape[2])]
                    inputs = inputs.to(device=device,dtype=torch.float32)
                    labels = labels.to(device=device,dtype=torch.float32)
                    inputs, labels = preprocessing_input(inputs, labels, base_model, fps)
                    # torch.Size([batch，csv_length, feature_dim]) torch.Size([batch，csv_length, bs_dim])
                    assert(labels.shape[0]==args.batch_size and labels.shape[2]==31 and labels.shape[1]==inputs.shape[1])
                    outputs.append(model(inputs))
                    del inputs
                    del labels
                    torch.cuda.empty_cache()
                
                outputs = torch.cat(outputs,dim=1)
                optimizer.zero_grad()
                loss = criterion(outputs, y_train[:,:outputs.shape[1],:].to(device=device,dtype=torch.float32))
                loss.backward()
                optimizer.step()

                train_loss.append(loss.cpu().detach().numpy())

                del loss
                del outputs
                del y_train
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
                frac = 1/20
                max_num_frac = args.max_audio_length*20
                split_size = random.randint(10, max_num_frac) * frac
                split_size_x = int(args.sampling_rate * split_size)
                split_size_y = int(60 * split_size)
                xs = torch.split(x_valid, split_size_x, dim=2)
                ys = torch.split(y_valid, split_size_y, dim=1)
                outputs = []
                for inputs,labels in zip(xs,ys):
                    if labels.shape[1]<4:
                        continue
                    elif inputs.shape[2]<split_size_x:
                        n_frame = labels.shape[1]
                        inputs = inputs[:,:,:min(n_frame*267,inputs.shape[2])]
                    inputs = inputs.to(device=device,dtype=torch.float32)
                    labels = labels.to(device=device,dtype=torch.float32)
                    
                    inputs, labels = preprocessing_input(inputs, labels, base_model, fps)
                    with torch.no_grad():
                        outputs.append(torch.clamp(model(inputs),0,1))

                    del inputs
                    del labels
                    torch.cuda.empty_cache()
                outputs = torch.cat(outputs,dim=1)
                loss = criterion(outputs, y_valid[:,:outputs.shape[1],:].to(device=device,dtype=torch.float32))
                valid_loss.append(loss.cpu().detach().numpy())

                del loss
                del outputs
                del y_valid
                torch.cuda.empty_cache()
                    
                if j%20==0:
                    t.set_description('Epoch %i validation' % epoch)
                    t.set_postfix(train_loss=np.mean(train_loss), val_loss = np.mean(valid_loss))
                t.update(1)
            epoch_val_loss = np.mean(valid_loss)
            valid_loss_list.append(epoch_val_loss)
            t.set_description('Epoch %i' % epoch)
            t.set_postfix(train_loss=train_loss_list[-1], val_loss=epoch_val_loss, lr=optimizer.state_dict()['param_groups'][0]['lr'])
            early_stopping(valid_loss_list[-1], model=model, optimizer=optimizer, path=args.model_path)
            scheduler.step(epoch_val_loss)
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
    from torch_augmentation import gain
    torch.cuda.empty_cache()
    t1 = time.time() # Load Processor
    processor = Wav2Vec2FeatureExtractor.from_pretrained(args.base_model_path)

    t2 = time.time()  # Load Hubert model
    base_model, fps = load_base_model(args.base_model_path, "Hubert", args.device)
    base_model.eval()
    

    t3 = time.time()  # Load LSTM model
    checkpoint = torch.load(checkpoint_path)
    print(checkpoint["loss"])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    t4 = time.time() 
    print("Load Processor: %dms" % (int(round((t2-t1) * 1000))))
    print("Load Base model (Hubert): %dms" % (int(round((t3-t2) * 1000))))
    print("Load LSTM model (LSTM): %dms\n" % (int(round((t4-t3) * 1000))))
    
    for wav in wav_lst:
        model.reset_hidden_cell()
        wav_path = os.path.join(args.test_data_path, wav)

        t5 = time.time() # Load audio
        sig, rate = librosa.load(wav_path, sr=args.sampling_rate)
        
        audio = processor(sig, return_tensors="pt", sampling_rate=rate).input_values # (1, audiolength)
        audio = gain(audio.unsqueeze(0),rate).squeeze(0)
        split_size = args.audio_section_length*args.sampling_rate
        chunks = torch.split(audio,split_size, dim=1)
        
        output = []

        base_model_infer_time = 0
        lstm_model_infer_time = 0

        for chunk in chunks:
            x = torch.FloatTensor(chunk).to(device=args.device)
            if chunk.shape[1]<640:
                break
            t6 = time.time() # Base model inference
            with torch.no_grad():
                last_h_state = base_model(x).last_hidden_state
            x = linear_interpolation(last_h_state, input_fps=fps, output_fps=60)
            t7 = time.time() # LSTM inference
            with torch.no_grad():
                x = torch.clamp(model(x),0,1).detach().cpu().numpy()
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
            model.reset_hidden_cell()
            
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
            model.reset_hidden_cell()
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
            model.reset_hidden_cell()
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