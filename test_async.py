from scipy.io import wavfile
import librosa
import numpy as np
import resampy

"""a1,a = wavfile.read("./20230220.wav")
b,b1 = librosa.load("./20230220.wav")
print(a1,len(a))
#audio = resampy.resample(a.astype(np.int16),a1,16000)
with open("tts.txt","w") as f:
    for i,line in enumerate((np.array(a)/256).astype(np.int16)):
        f.write(str(i)+" "+str(line)+"\n")"""
"""a = np.array([1,0,1,0,3,0,4,0,-1,0]).astype(np.int16)

arr = np.vectorize(np.binary_repr)(a, width=16)
print(arr)
"""
import wave
def pcm2wav(pcm_file, wav_file, channels=1, bits=16, sample_rate=16000):
    pcmf = open(pcm_file, 'rb')
    pcmdata = pcmf.read()
    pcmf.close()
 
    if bits % 8 != 0:
        raise ValueError("bits % 8 must == 0. now bits:" + str(bits))
 
    wavfile = wave.open(wav_file, 'wb')
    wavfile.setnchannels(channels)
    wavfile.setsampwidth(bits // 8)
    wavfile.setframerate(sample_rate)
    wavfile.writeframes(pcmdata)
    wavfile.close()


import numpy as np
import time
def load_pcm(pcm_file):
    t1 = time.time()
    b = np.fromfile(pcm_file,dtype=np.int16)
    t2 = time.time()
    print(t1)
    print(len(b))
    print(t2)

#pcm2wav("test_audio.pcm","test_audio.wav")
load_pcm("test_audio.pcm")