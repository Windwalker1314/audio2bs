from arguments import get_common_args,get_train_args
from dataset import get_dataloaders,create_dataloaders
from trainer import train, inference,test
import torch.optim as optim
from model import LSTM, Faceformer, Conformer,Transformer
import torch.nn as nn

#from transformers import HubertModel
import os
import torch

def runner(args):
    #path = create_dataloaders(args)
    dataset = get_dataloaders(args.dataset_path)
    
    if "LSTM" in args.model_name:
        model = LSTM()
    elif "Faceformer" in args.model_name:
        model = Faceformer(args)
    elif "Transformer" in args.model_name:
        model = Transformer()
    elif "Conformer" in args.model_name:
        model = Conformer()
    model.to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    criterion = nn.MSELoss()
    current_loss = 999999
    if args.load_model:
        checkpoint = torch.load(os.path.join(args.model_path, args.model_name+".pth"))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        current_loss = checkpoint['loss']
        print(current_loss)
    criterion.to(args.device)
    
    train(args, model, dataset,criterion=criterion, optimizer=optimizer, device=args.device, current_loss=current_loss)

    """print(torch.cuda.memory_reserved())
    model = HubertModel.from_pretrained(args.base_model_path)
    model = model.to(args.device)
    print(torch.cuda.memory_reserved()*9.3132E-10)"""
    #model = HubertModel.from_pretrained(args.base_model_path)

    # torch.Size([1, 1, 187360]) torch.Size([1, 700, 31])

def infer(args):
    if "LSTM" in args.model_name:
        model = LSTM()
    elif "Faceformer" in args.model_name:
        model = Faceformer(args)
    elif "Transformer" in args.model_name:
        model = Transformer()
    elif "Conformer" in args.model_name:
        model = Conformer()
    model.to(args.device)
    test_data = os.listdir(args.test_data_path)
    wavs = []
    for i in test_data:
        if i.endswith(".wav"):
            wavs.append(i)
    inference(args, model, checkpoint_path=os.path.join(args.model_path, args.model_name+".pth"), wav_lst=wavs, calibration=False)

def test_model(args):
    #path = create_dataloaders(args)
    dataset = get_dataloaders(args.dataset_path)
    if "LSTM" in args.model_name:
        model = LSTM()
    elif "Faceformer" in args.model_name:
        model = Faceformer(args)
    elif "Transformer" in args.model_name:
        model = Transformer()
    elif "Conformer" in args.model_name:
        model = Conformer()
    model.to(args.device)
    criterion = nn.MSELoss()
    train_loss, valid_loss, test_loss = test(args, model, checkpoint_path=os.path.join(args.model_path, args.model_name+".pth"), dataset=dataset, criterion=criterion)
    indexes = []
    for i,loss in zip(dataset["Train_key"], train_loss):
        if loss>0.005:
            print(i,loss)
            indexes.append(i)
    print()
    for i,loss in zip(dataset["Val_key"], valid_loss):
        if loss>0.005:
            print(i,loss)
            indexes.append(i)
    print()
    for i,loss in zip(dataset["Test_key"], test_loss):
        if loss>0.005:
            print(i,loss)
            indexes.append(i)
    print(indexes)
if __name__=="__main__":
    args = get_common_args()
    args = get_train_args(args)
    runner(args)
    #infer(args)
    #test_model(args)


    #LSTM  0.0166
    #LSTM1 0.017
    #LSTM aug 0.021