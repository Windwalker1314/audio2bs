import json
import os
import requests
import time
import datetime
import jwt
import json
import requests

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
    jwt_token = jwt_token_temp.decode('ascii')# python3 编码后得到 bytes, 再进行解码(指明解码的格式), 得到一个str
    return jwt_token



def http_tts(url, key, secret, content, speed, energy, sample_rate, bit, audio_format, kbps, name_save = 'output.wav'):
    s = json.dumps({'content': content, 'speed': speed, "energy":energy, "sample_rate":sample_rate, "audio_format":audio_format, "bit":bit, "kbps":kbps})
    jwt_token = GetJWTToken(key, secret)
    headers={"Content_type":"application/json;charset=utf-8","Authentication":"Bearer {}".format(jwt_token)}
    r = requests.post(url, data=s, headers = headers)
    #print(r)
    #print(r.content)
    audio = r.content    
    with open(name_save, 'wb') as wf:
        wf.write(audio)

url = "https://ai.cubigdata.cn:5001/openapi/speech/tts/short?speeker=ttsw01"
key = "bg3mTvHWlhpXwlcRullOKxUY"##内部测试权限
secret = "SVsUZjWf1aANZ5XDeC6tKARdpccsWiw7"#内部测试权限
content = "宝贝我爱你"###需要转为音频的文本，必填，长度最长为3000
speed = 1.0###音频的速度，0.5到1.5
energy = 1.0###音频的音量，0.5到1.5
sample_rate = 22050###采样率，可选8000,16000,22050
bit = 16###采样宽度，可选8、16
kbps = 128
audio_format = 'mp3'
name_save = f'./audio/output-http-{kbps}.{audio_format}'###本地保存文件名

#kbps = 256
http_tts(url, key, secret, content, speed, energy, sample_rate, bit, audio_format, kbps, name_save)
