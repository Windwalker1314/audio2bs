import json
import asyncio
import websockets
from arguments import get_common_args,get_train_args
from audio2bs import Audio2BS
from transformers import Wav2Vec2FeatureExtractor
import os

IP_ADDR = "127.0.0.1"
IP_PORT = "7890"
 
key = "123"

# 握手，通过接收hello，发送"123"来握手。
async def serverHands(websocket):
    print("ServerHands")
    while True:
        recv_text = await websocket.recv()
        print("recv_text=" + recv_text)
        if recv_text == "hello":
            print("connected success")
            await websocket.send(key)
            return True
        else:
            await websocket.send("connected fail")
 
 
# 接收从客户端发来的消息并处理，告知客户端
async def serverRecv(websocket, model):
    while True:
        data = await websocket.recv()
        data = json.loads(data)
        result = model.inference(data["wav"], data["rate"])
        out_data = json.dumps({"result": result.tolist()},ensure_ascii=False).encode('gbk')
        await websocket.send(out_data)
        

async def init_model(websocket):
    args = get_common_args()
    args = get_train_args(args)
    base_model_path = args.base_model_path
    model_path = os.path.join(args.model_path,args.model_name+".pth")
    device = args.device
    # Load model 
    my_model = Audio2BS(base_model_path, model_path, device)
    await websocket.send("Model Loaded")
    return my_model


 
# 握手并且接收数据
async def serverRun(websocket, path):
    print(path)
    connected = await serverHands(websocket)
    if connected:
        model = await init_model(websocket)
    await serverRecv(websocket, model)


 
#main function
if __name__ == '__main__':
    print("======server main begin======")
    server = websockets.serve(serverRun, IP_ADDR, IP_PORT)
    asyncio.get_event_loop().run_until_complete(server)
    asyncio.get_event_loop().run_forever()