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
a = np.array([1,0,1,0,3,0,4,0,-1,0]).astype(np.int16)

arr = np.vectorize(np.binary_repr)(a, width=16)
print(arr)