import asyncio
import websockets
from scipy.io import wavfile
import json
import numpy as np
IP_ADDR = "localhost"
IP_PORT = "7890"


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
        input_text = input("Enter wav path:")
        """
        下面这段需要替换，持续输入音频，以json的方式发送到server，再接受预测结果
        """
        if input_text.endswith(".wav"):
            #sig, rate = librosa.load(input_text, sr=16000)
            rate, sig = wavfile.read(input_text)
            n_chunks = int(len(sig)//rate)
            for i in range(n_chunks):
                sig_section=sig[i*rate:min((i+1)*rate, len(sig))].tolist()
                data = json.dumps({"wav": sig_section, "rate": rate},ensure_ascii=False).encode('utf-8')
                await websocket.send(data)
                result = await websocket.recv()
                result = json.loads(result)
                print("Jawopen:", np.array(result["result"])[:,3], "bs_name:",np.array(result["bs_name"]).shape)
            continue
        elif input_text == "exit":
            print(f'"exit", bye!')
            await websocket.close(reason="exit")
            return False
        else:
            await websocket.send(input_text)
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
                await clientSend(websocket)
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