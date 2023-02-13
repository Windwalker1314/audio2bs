import torch
import numpy as np
from tqdm import tqdm
import librosa
from util import linear_interpolation, MOUTH_BS
from transformers import HubertModel,Wav2Vec2FeatureExtractor
import pandas as pd
import time
import os

def load_base_model(model_path,model_type, device):
    if model_type=="Hubert":
        model = HubertModel.from_pretrained(model_path)
        model_fps = 49
    else:
        raise NameError
    model.to(device)
    return model, model_fps


def preprocessing_input(x, y, base_model, model_fps):
    assert len(x.shape)==3 and x.shape[1]==1, "Expected 3D shape but get "+str(x.shape)
    x = x.squeeze(1)
    base_model.eval()
    last_hidden_state = base_model(x).last_hidden_state
    # 1, length, 1024
    x = linear_interpolation(last_hidden_state, input_fps=model_fps, output_fps=60, output_len=y.shape[-2])
    return x, y

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
        model.reset_hidden_cell()
        wav_path = os.path.join(args.test_data_path, wav)

        t5 = time.time() # Load audio
        sig, rate = librosa.load(wav_path, sr=args.sampling_rate)
        audio = processor(sig, return_tensors="pt", sampling_rate=rate).input_values # (1, audiolength)
        
        chunks = torch.split(audio,args.audio_section_length*args.sampling_rate+1, dim=1)
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