import json
import asyncio
import websockets
from arguments import get_common_args,get_train_args,get_server_args
from audio2bs import Audio2BS
import os
import base64
import numpy as np

# 检测客户端权限，用户名密码通过才能退出循环 
async def check_permit(websocket):
    while True:
        recv_str = await websocket.recv()
        cred_dict = recv_str.split(":")
        if cred_dict[0] == "admin" and cred_dict[1] == "123456":
            response_str = "congratulation, you have connect with server\r\nnow, you can do something else"
            await websocket.send(response_str)
            return True
        else:
            response_str = "sorry, the username or password is wrong, please submit again"
            await websocket.send(response_str)

# 接收从客户端发来的消息并处理，告知客户端
#输入是json文件，
# {"wav": {"status":..., "audio":..., "text_normalized":..., "text_raw":...}          # 从client_ws_gateway得到的raw json file
#  "rate": 22050,                    #音频采样率设置
#  "return_mode": "sentence"         #建议使用这个mode, 其它的还未测试
# }

# 输出是json 文件,
# {
#   "result":[[...],[...]]   # blendshape输出，float，shape(number_of_frames, 31)    
#   "bs_name":[]             # blendshape名称，str, shape(31)
#   "status":[]              # 401:给的数据有问题， 501：服务器推理错误， 200:成功
#   "message": ""            # 如果成功就输出"Success",否则输出错误log
# }
async def serverRecv(websocket, model):
    while True:
        data = await websocket.recv()
        message = ""
        audio = None
        status = 401
        try:
            data = json.loads(data)
            if "text" in data.keys():
                message = data["text"]
            else:
                audio, rate, status = handel_result(data)
        except json.decoder.JSONDecodeError as e:
            message = "Json Decode Error:" + getattr(e, 'message', repr(e))
        except KeyError as e:
            message = "Json Key Error: " + str(e)
        except Exception as e:
            message = str(type(e))+ str(e)

        if audio is not None:
            try:
                result = model.inference(audio, rate).squeeze().tolist()
                message = "Inference Success"
            except Exception as e:
                message = "Model Inference Failure "+str(type(e)) + str(e)
                status = 501
        else:
            result = []
        out_data = json.dumps({"result": result, "bs_name":model.MOUTH_BS, "status":status,"message":message},ensure_ascii=False).encode('UTF-8')
        await websocket.send(out_data)

def handel_result(data):
    res = data["wav"]
    rate = data["rate"]
    return_mode = data["return_mode"]
    audio = res['audio']
    status = 200
    if return_mode == "sentence":
        audio = base64.b64decode(res['audio'])
        text_normalized = res['text_normalized']
        print("text_normalized:", text_normalized)
        audio = np.frombuffer(audio, dtype=np.int16)
        status += res["status"]
    elif return_mode == 'stream':
        if isinstance(res, str):
            status += res['status']
            text_normalized = res['text_normalized']
            print("text_normalized", text_normalized)
            audio=None
        elif isinstance(res, bytes):
            audio = np.frombuffer(res, dtype=np.int16)
    elif return_mode == 'only_audio':
        if isinstance(res, bytes):
            audio = np.frombuffer(res, dtype=np.int16)
        elif isinstance(res, str):
            status += res["status"]
    if status == 617:
        status = 401

    return audio, rate, status

def init():
    args = get_common_args()
    args = get_train_args(args)
    args = get_server_args(args)
    base_model_path = args.base_model_path
    model_path = os.path.join(args.model_path,args.model_name+".pth")
    device = args.device
    # Load model 
    global my_model 
    my_model =  Audio2BS(base_model_path, model_path, device)
    server = websockets.serve(serverRun, args.IP, args.port)
    return server

# 握手并且接收数据
async def serverRun(websocket, path):
    await check_permit(websocket)
    try:
        await serverRecv(websocket, my_model)
    except websockets.exceptions.ConnectionClosedOK as e:
        print("Client exit!")
    except websockets.exceptions.ConnectionClosedError as e:
        print("Connection closed abnormally: ", str(e.reason))

#main function
if __name__ == '__main__':
    print("========Server main begin==========")
    server = init()
    print("======Initialization completed======")
    
    asyncio.get_event_loop().run_until_complete(server)
    asyncio.get_event_loop().run_forever()