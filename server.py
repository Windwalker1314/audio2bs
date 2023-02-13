import json
import asyncio
import websockets
from arguments import get_common_args,get_train_args
from inference import inference_simple, load_base_model, load_model
from transformers import Wav2Vec2FeatureExtractor
 
IP_ADDR = "127.0.0.1"
IP_PORT = "7890"
 
key = "123"

# 握手，通过接收hello，发送"123"来进行双方的握手。
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
async def serverRecv(websocket, args, processor, model, base_model, fps):
    while True:
        data = await websocket.recv()
        data = json.loads(data)
        result = inference_simple(args, data["wav"], processor, model, base_model, fps)
        out_data = json.dumps({"result": result.tolist()},ensure_ascii=False).encode('gbk')
        await websocket.send(out_data)
        

async def init_model(websocket):
    args = get_common_args()
    args = get_train_args(args)
    processor = Wav2Vec2FeatureExtractor.from_pretrained(args.base_model_path)
    model = load_model(args)
    base_model,fps = load_base_model(args.base_model_path, "Hubert", args.device)
    model.reset_hidden_cell()
    await websocket.send("Model Loaded")
    return args, processor, model, base_model, fps


 
# 握手并且接收数据
async def serverRun(websocket, path):
    print(path)
    connected = await serverHands(websocket)
    if connected:
        args, processor, model, base_model, fps = await init_model(websocket)
    await serverRecv(websocket, args, processor, model, base_model, fps)


 
#main function
if __name__ == '__main__':
    print("======server main begin======")
    server = websockets.serve(serverRun, IP_ADDR, IP_PORT)
    asyncio.get_event_loop().run_until_complete(server)
    asyncio.get_event_loop().run_forever()