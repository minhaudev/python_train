import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write, read
from scipy.fft import rfft, rfftfreq

def record_audio(duration=10, filename="output.wav"):
    
    fs = 44100
    print("Bắt đầu ghi âm...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=2)
    sd.wait()  # Chờ ghi âm hoàn tất
    write(filename, fs, audio)  # Lưu vào file
    print("Ghi âm hoàn tất. Đã lưu vào", filename)
    return filename  # Trả về tên file sau khi ghi âm xong
