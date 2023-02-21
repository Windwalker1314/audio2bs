# coding=utf-8
import websocket#需要安装websocket_client==0.57.0
import json
import base64
import os
import datetime
import jwt##需要安装PyJWT==1.5.3
import time
import numpy as np

def GetJWTToken(key, secret):
    today = datetime.datetime.now()
    delta = datetime.timedelta(hours=0.5)
    exp = today + delta
    dt = exp.strftime("%Y-%m-%d %H:%M:%S")
    timeArray = time.strptime(dt, "%Y-%m-%d %H:%M:%S")
    exp = int(time.mktime(timeArray))
    time_cur = int(time.time())
    token_dict = {
            'iss': key,
            'iat': time_cur,  # 时间戳
            'exp': exp
            }
    headers = {
            "alg": "HS256",  # 声明所使用的算法
            "typ": "JWT"
            }
    jwt_token_temp = jwt.encode(token_dict,  # payload, 有效载体
                                secret,  # 进行加密签名的密钥
                                algorithm="HS256",  # 指明签名算法方式, 默认也是HS256
                                headers=headers  # json web token 数据结构包含两部分, payload(有效载体), headers(标头)
                                )
    #jwt_token = jwt.decode(jwt_token_temp, secret, algorithms="HS256")# python3 编码后得到 bytes, 再进行解码(指明解码的格式), 得到一个str
    return jwt_token_temp


def wss_tts(url, key, secret, txt, return_mode, speed, energy, sample_rate, audio_format, bit, name_save = 'output.pcm'):
    
    jwt_token = GetJWTToken(key, secret)
    header={"Content_type:application/json;charset=utf-8","Authorization:Bearer {}".format(jwt_token)}
    ws = websocket.create_connection(url,header = header)
    data = {'content':txt, 'return_mode':return_mode, 'speed':speed, 'sample_rate': sample_rate, 'energy':energy, 'audio_format':audio_format, 'bit':bit}
    data = json.dumps(data)
    fw = open(name_save, 'wb')
    ws.send(data)
    finish = False
    while True:
        try:
            res = ws.recv()
        except:
            break

        if return_mode == "sentence":
            res = json.loads(res)
            status = res['status']
            audio = base64.b64decode(res['audio'])
            with open ("example.json",'w') as f:
                json.dump({"wav":res,"rate":22050,"return_mode":"sentence"},f)
            text_raw = res['text_raw']
            text_normalized = res['text_normalized']
            print("text_normalized:", text_normalized)
            fw.write(audio)
            if status == 1:
                print("完成")
                break
        elif return_mode == 'stream':
            if isinstance(res, str):
                res = json.loads(res)
                status = res['status']
                text_raw = res['text_raw']
                text_normalized = res['text_normalized']
                print("text_normalized:", text_normalized)
                if status == 1:
                    finish = True
            elif isinstance(res, bytes):
                print(len(np.frombuffer(res, dtype=np.int16)))
                fw.write(res)
                if finish:
                    print("完成")
                    break
        elif return_mode == 'only_audio':
            if isinstance(res, bytes):
                fw.write(res)
            elif isinstance(res, str):
                if json.loads(res)['status'] == 1:
                    print("完成")
                    break


    fw.close()
    ws.close()
    
    

if __name__ == '__main__':
    txt = "这是中国联通开发的语音合成系统，测试完成，运行正常"##待合成的文本，必填项
    url = 'wss://ai.cubigdata.cn:5001/openapi/speech/tts?speeker=ttsw01'
    speed = 1.0##声音的快慢 支持0.5到1.0， 默认1.0
    energy = 1.0##声音的音量大小 支持0.5到1.0， 默认1.0
    sample_rate = 22050###声音的采样率，支持8000、16000、22050
    return_mode = 'sentence'###返回模式，必填，支持stream、sentence、only_audio
    audio_format = 'pcm'
    bit = 16
    name_save = 'test_audio.pcm'
    key = "nFqRzTL8DGvsRCNiI4dNHoqraVa6OfNS"##内部测试权限
    secret = "PXgZvMfHOQfHaostEyPLNJ3T2B9fm32o"#内部测试权限
    wss_tts(url = url, key = key, secret = secret,txt = txt, return_mode = return_mode, speed = speed, energy = energy, sample_rate = sample_rate, audio_format = audio_format, bit = bit, name_save = name_save)
