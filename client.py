import asyncio
import websockets
import json
import numpy as np
from arguments import get_common_args,get_train_args,get_server_args

count = 0


# 向服务器端认证，用户名密码通过才能退出循环 直接输入  admin:123456
async def auth_system(websocket): 
    while True:
        cred_text = "admin:123456"#input("please enter your username and password: ")
        await websocket.send(cred_text)
        response_str = await websocket.recv()
        if "congratulation" in response_str:
            return True
 
async def check_init_model(websocket):
    while True:
        recv_text = await websocket.recv()
        print(f"{recv_text}")
        if recv_text == "Initialization Completed":
            return True

# 向服务器端发送消息   
async def clientSend(websocket):
    while True:
        try:
            input_text = input("Enter json or txt path:")
            if input_text.endswith(".json"):
                # load json file
                with open(input_text,"r") as f:
                    data = json.load(f)      # 将json文件读取为python dictionary
                data_send = json.dumps(data,ensure_ascii=False).encode("UTF-8") # 将python dictionary转为json object

                # 发送接受数据
                await websocket.send(data_send)
                result = await websocket.recv()
                result = json.loads(result)
                print("result:", np.array(result["result"]).shape, 
                    "bs_name:",np.array(result["bs_name"]).shape,
                    "status",result["status"],
                    "message",result["message"])
                continue
            # 输入时字符串
            elif input_text.endswith(".txt"):
                with open(input_text,"r") as f:
                    data = json.loads(f.read().strip())  # json loads: 字符串转成dictionary
                data_send = json.dumps(data,ensure_ascii=False).encode("UTF-8")   # json dumps: dictionary转成json object

                await websocket.send(data_send)  # 发送json object
                result = await websocket.recv()
                result = json.loads(result)
                print("result:", np.array(result["result"]).shape, 
                    "bs_name:",np.array(result["bs_name"]).shape,
                    "status:",result["status"],
                    "message:",result["message"])
                continue
            # 输入exit 断开链接
            elif input_text == "exit":
                print(f'"exit", bye!')
                await websocket.close(reason="exit")
                return False
            # 输入其它的话，echo back
            else:
                data_send = json.dumps({"text":input_text},ensure_ascii=False).encode("UTF-8")
                await websocket.send(data_send)
                recv_text = await websocket.recv()
                print(f"{recv_text}")
        except Exception as e:
            print(type(e),str(e))

 
# 进行websocket连接
async def clientRun():
    ipaddress = args.IP + ":" + args.port
    while True:
        try:
            async with websockets.connect("ws://" + ipaddress, ping_interval=None, ping_timeout=None) as websocket:
                await auth_system(websocket)
                await check_init_model(websocket)
                
                if False == await clientSend(websocket):
                    # 断开连接
                    break
                
        except ConnectionRefusedError as e:
            print(e)
            global count
            if count>5:
                return
            count += 1
            await asyncio.sleep(2)
            print("Reconnecting...")
        except websockets.exceptions.ConnectionClosedError as e:
            print("Timeout, Reconnecting...")

def init():
    global args
    args = get_common_args()
    args = get_train_args(args)
    args = get_server_args(args)

#main function
if __name__ == '__main__':
    print("======client main begin======")
    init()
    asyncio.get_event_loop().run_until_complete(clientRun())