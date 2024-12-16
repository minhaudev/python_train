import glob
import pickle
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from scipy.io.wavfile import write, read
from scipy.fft import fft, fftfreq
from scipy import signal
from tqdm import tqdm
from typing import List, Dict, Tuple

# 1. Ghi âm và lưu tệp âm thanh
def record_audio(filename: str, duration: int, samplerate: int = 22050):
    print("Recording...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()
    write(filename, samplerate, audio)
    print("Recording saved as:", filename)

# 2. Hiển thị tín hiệu âm thanh theo thời gian
def plot_time_signal(filename: str):
    fs, song = read(filename)
    duration = len(song) / fs
    time = np.linspace(0., duration, len(song))
    plt.figure(figsize=(10, 4))
    plt.plot(time, song)
    plt.title("Tín hiệu âm thanh theo thời gian")
    plt.xlabel("Thời gian (giây)")
    plt.ylabel("Biên độ")
    plt.grid()
    plt.tight_layout()
    # plt.show()

# 3. Phân tích phổ tần số bằng DFT
def plot_frequency_spectrum_dft(filename: str):
    fs, song = read(filename)
    if song.ndim > 1:
        song = song[:, 0]
    N = len(song)
    T = 1.0 / fs
    yf = fft(song)
    xf = fftfreq(N, T)[:N // 2]
    plt.figure(figsize=(10, 4))
    plt.plot(xf, 2.0 / N * np.abs(yf[:N // 2]))
    plt.title("Phổ tần số của tín hiệu âm thanh (DFT)")
    plt.xlabel("Tần số (Hz)")
    plt.ylabel("Biên độ")
    plt.grid()
    plt.tight_layout()
    # plt.show()

# 4. Tạo Spectrogram dựa trên DFT
def plot_spectrogram_dft(filename: str):
    fs, song = read(filename)
    if song.ndim > 1:
        song = song[:, 0]
    segment_size = 1024
    overlap = segment_size // 2
    step = segment_size - overlap
    spectrogram = []
    for start in range(0, len(song) - segment_size, step):
        segment = song[start:start + segment_size]
        fft_segment = fft(segment)[:segment_size // 2]
        spectrogram.append(np.abs(fft_segment))
    spectrogram = np.array(spectrogram).T
    time = np.arange(spectrogram.shape[1]) * step / fs
    freqs = np.linspace(0, fs / 2, spectrogram.shape[0])
    plt.figure(figsize=(10, 4))
    plt.pcolormesh(time, freqs, 10 * np.log10(spectrogram), shading='gouraud')
    plt.title("Spectrogram của tín hiệu âm thanh (DFT)")
    plt.xlabel("Thời gian (giây)")
    plt.ylabel("Tần số (Hz)")
    plt.colorbar(label="Cường độ (dB)")
    plt.tight_layout()
    # plt.show()

# 5. Tạo dấu vân tay từ tín hiệu âm thanh
def create_hashes(audio: np.ndarray, fs: int, song_id=None) -> Dict[int, Tuple[int, int]]:
    hashes = {}
    window_length_samples = int(0.5 * fs)
    frequencies, times, stft = signal.stft(audio, fs, nperseg=window_length_samples, return_onesided=True)
    for time_idx, spectrum in enumerate(np.abs(stft.T)):
        peaks, _ = signal.find_peaks(spectrum, prominence=0.3, distance=150)
        for peak in peaks:
            hash_val = hash((int(frequencies[peak]), time_idx))
            hashes[hash_val] = (time_idx, song_id)
    return hashes

# 6. Xây dựng cơ sở dữ liệu
def build_database(data_path: str, db_filename: str, song_index_filename: str):
    songs = glob.glob(f"{data_path}/*")
    song_index = {}
    database = {}
    for idx, song_file in enumerate(tqdm(songs)):
        song_index[idx] = song_file
        try:
            fs, audio = read(song_file)
            if audio.ndim > 1:
                audio = audio[:, 0]
            hashes = create_hashes(audio, fs, song_id=idx)
            for hash_val, (t, song_id) in hashes.items():
                if hash_val not in database:
                    database[hash_val] = []
                database[hash_val].append((t, song_id))
        except Exception as e:
            print(f"Error processing {song_file}: {e}")
    with open(db_filename, "wb") as db_file, open(song_index_filename, "wb") as idx_file:
        pickle.dump(database, db_file)
        pickle.dump(song_index, idx_file)

# 7. Nhận dạng bài hát
def identify_song(recording_filename: str, db_filename: str, song_index_filename: str):
    with open(db_filename, "rb") as db_file, open(song_index_filename, "rb") as idx_file:
        database = pickle.load(db_file)
        song_index = pickle.load(idx_file)

    fs, audio = read(recording_filename)
    if audio.ndim > 1:
        audio = audio[:, 0]
    hashes = create_hashes(audio, fs)

    match_count = {}
    for hash_val, (t, _) in hashes.items():
        if hash_val in database:
            for _, song_id in database[hash_val]:
                match_count[song_id] = match_count.get(song_id, 0) + 1

    if match_count:
        best_match = max(match_count.items(), key=lambda x: x[1])
        print(f"Best match: {song_index[best_match[0]]} with {best_match[1]} matches.")
    else:
        print("No matching song found.")

# Chạy các hàm chính
if __name__ == "__main__":
    # Ghi âm tín hiệu
    record_audio("output.wav", duration=10)

    # Hiển thị tín hiệu theo thời gian
    plot_time_signal("output.wav")

    # Phổ tần số (DFT)
    plot_frequency_spectrum_dft("output.wav")

    # Spectrogram với DFT
    plot_spectrogram_dft("output.wav")

    # Xây dựng cơ sở dữ liệu
    build_database(data_path="data", db_filename="database.pickle", song_index_filename="song_index.pickle")

    # Nhận dạng bài hát
    identify_song("output.wav", db_filename="database.pickle", song_index_filename="song_index.pickle")
