import asyncio
import websockets
import librosa
import json
import numpy as np
IP_ADDR = "localhost"
IP_PORT = "7890"

key = "123"
 
# 握手，通过发送hello，接收"123"来进行双方的握手。
async def clientHands(websocket):
    while True:
        await websocket.send("hello")
        response_str = await websocket.recv()
        if key in response_str:
            print("握手成功")
            recv_text = await websocket.recv()
            print(recv_text)
            return True
 
 
# 向服务器端发送消息
async def clientSend(websocket):
    while True:
        input_text = input()
        """
        下面这段需要替换，持续输入音频，以json的方式发送到server，再接受预测结果
        """
        if input_text.endswith(".wav"):
            sig, rate = librosa.load(input_text, sr=16000)
            n_chunks = int(len(sig)//16000)
            for i in range(n_chunks):
                sig_section=sig[i*16000:min((i+1)*16000, len(sig))].tolist()
                data = json.dumps({"wav": sig_section, "rate": 16000},ensure_ascii=False).encode('gbk')
                await websocket.send(data)
                result = await websocket.recv()
                result = json.loads(result)
                print("output_shape:", np.array(result["result"]).shape)
            continue
        if input_text == "exit":
            print(f'"exit", bye!')
            await websocket.close(reason="exit")
            return False
        await websocket.send(input_text)
        recv_text = await websocket.recv()
        print(f"{recv_text}")
 
 
# 进行websocket连接
async def clientRun():
    ipaddress = IP_ADDR + ":" + IP_PORT
    async with websockets.connect("ws://" + ipaddress) as websocket:
        await clientHands(websocket)

        await clientSend(websocket)
 
#main function
if __name__ == '__main__':
    print("======client main begin======")
    asyncio.get_event_loop().run_until_complete(clientRun())