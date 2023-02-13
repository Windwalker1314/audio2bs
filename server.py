import json
import asyncio
import websockets
from arguments import get_common_args,get_train_args
from audio2bs import Audio2BS
import os

IP_ADDR = "127.0.0.1"
IP_PORT = "7890"


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
async def serverRecv(websocket, model):
    while True:
        data = await websocket.recv()
        data = json.loads(data)
        result = model.inference(data["wav"], data["rate"])
        out_data = json.dumps({"result": result.tolist()},ensure_ascii=False).encode('gbk')
        await websocket.send(out_data)
        

async def init_model(websocket):
    await websocket.send("Loading model")
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
async def serverRun(websocket,path):
    connected = await check_permit(websocket)
    if connected:
        model = await init_model(websocket)
    await serverRecv(websocket, model)


 
#main function
if __name__ == '__main__':
    print("======server main begin======")
    server = websockets.serve(serverRun, IP_ADDR, IP_PORT)
    asyncio.get_event_loop().run_until_complete(server)
    asyncio.get_event_loop().run_forever()