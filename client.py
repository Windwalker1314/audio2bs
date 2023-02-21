import asyncio
import websockets
from scipy.io import wavfile
import json
import numpy as np
IP_ADDR = "localhost"
IP_PORT = "2890"
import time

count = 0


# 向服务器端认证，用户名密码通过才能退出循环 直接输入  admin:123456
async def auth_system(websocket): 
    while True:
        cred_text = input("please enter your username and password: ")
        await websocket.send(cred_text)
        response_str = await websocket.recv()
        if "congratulation" in response_str:
            return True
 
# 向服务器端发送消息
        
async def clientSend(websocket):
    while True:
        input_text = "example_short.json"#input("Enter json path:")
        """
        输入example.json得到结果
        """
        if input_text.endswith(".json"):
            # load json file
            with open(input_text,"r") as f:
                data = json.load(f)  # data is a python dictionary
                #data = f.readlines()
                #data="".join(data)
            #data = json.loads(data)
            #print(data)
            data_send = json.dumps(data) # datasend is a json object
            t1 = time.time()
            await websocket.send(data_send)  # 发送json object
            result = await websocket.recv()
            result = json.loads(result)
            """print("Jawopen:", np.array(result["result"]).shape, 
                  "bs_name:",np.array(result["bs_name"]).shape,
                  "status",result["status"],
                  "message",result["message"])"""
            t2 = time.time()
            print("Inference Time:",int(round((t2-t1)*1000)),"ms")
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

 
 
# 进行websocket连接
async def clientRun():
    ipaddress = IP_ADDR + ":" + IP_PORT
    while True:
        try:
            async with websockets.connect("ws://" + ipaddress) as websocket:
                await auth_system(websocket)
                #await init_check(websocket)
                if False == await clientSend(websocket):
                    break
        except ConnectionRefusedError as e:
            print(e)
            global count
            if count==10:
                return
            count += 1
            await asyncio.sleep(2)
            print("Reconnecting...")
 
#main function
if __name__ == '__main__':
    print("======client main begin======")
    asyncio.get_event_loop().run_until_complete(clientRun())