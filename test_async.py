from scipy.io import wavfile
import librosa

a1,a = wavfile.read("./test_data/tts_1.wav")
b,b1 = librosa.load("./test_data/tts_1.wav")
print(a[0],b[0],max(a/32768),max(b))