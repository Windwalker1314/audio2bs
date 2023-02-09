import pandas as pd
import torch.nn.functional as F

MOUTH_BS = ['JawForward', 'JawRight', 'JawLeft', 'JawOpen',
       'MouthClose', 'MouthFunnel', 'MouthPucker', 'MouthRight', 'MouthLeft',
       'MouthSmileLeft', 'MouthSmileRight', 'MouthFrownLeft',
       'MouthFrownRight', 'MouthDimpleLeft', 'MouthDimpleRight',
       'MouthStretchLeft', 'MouthStretchRight', 'MouthRollLower',
       'MouthRollUpper', 'MouthShrugLower', 'MouthShrugUpper',
       'MouthPressLeft', 'MouthPressRight', 'MouthLowerDownLeft',
       'MouthLowerDownRight', 'MouthUpperUpLeft', 'MouthUpperUpRight',
       'CheekPuff', 'CheekSquintLeft', 'CheekSquintRight','TongueOut']

def linear_interpolation(features, input_fps, output_fps, output_len=None):
    features = features.transpose(1, 2)
    seq_len = features.shape[2] / float(input_fps)
    if output_len is None:
        output_len = int(seq_len * output_fps)
    output_features = F.interpolate(features,size=output_len,align_corners=True,mode='linear')
    return output_features.transpose(1, 2)

def bs_handler(path,calibration=True):
    df = pd.read_csv(path)
    out_array = df[MOUTH_BS].to_numpy()
    if calibration:
        out_array -= out_array[0]
    out_array[out_array<0] = 0
    return out_array

