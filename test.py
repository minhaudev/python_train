import glob
import pickle
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from scipy.io.wavfile import write, read
from scipy import signal
from scipy.fft import fft, fftfreq
from tqdm import tqdm
from pydub import AudioSegment
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
    plt.plot(time, song)
    # plt.title("Sound Signal Over Time")
    # plt.xlabel("Time (s)")
    # plt.ylabel("Amplitude")
    # plt.show()

# 3. Tạo dấu vân tay (constellation) cho bài hát
def create_constellation(audio: np.ndarray, fs: int, peak_count: int = 8) -> List[Tuple[int, float]]:
    window_length_samples = int(0.5 * fs)
    frequencies, times, stft = signal.stft(audio, fs, nperseg=window_length_samples, return_onesided=True)
    constellation_map = []
    for time_idx, spectrum in enumerate(np.abs(stft.T)):
        peaks, _ = signal.find_peaks(spectrum, prominence=0.3, distance=150)
        if len(peaks) > peak_count:
            peaks = peaks[np.argsort(spectrum[peaks])[-peak_count:]]
        for peak in peaks:
            constellation_map.append((time_idx, frequencies[peak]))
    return constellation_map

# 4. Tạo hàm băm từ dấu vân tay
def create_hashes(constellation_map: List[Tuple[int, float]], song_id=None) -> Dict[int, Tuple[int, int]]:
    hashes = {}
    for i, (t1, f1) in enumerate(constellation_map):
        for t2, f2 in constellation_map[i+1:i+20]:
            delta_t = t2 - t1
            if 1 <= delta_t <= 10:
                hash_val = hash((int(f1), int(f2), delta_t))
                hashes[hash_val] = (t1, song_id)
    return hashes

# 5. Đọc file âm thanh
def read_audio_file(file_path: str) -> Tuple[int, np.ndarray]:
    if file_path.endswith('.mp3'):
        audio_segment = AudioSegment.from_mp3(file_path)
        audio_data = np.array(audio_segment.get_array_of_samples())
        fs = audio_segment.frame_rate
    else:
        fs, audio_data = read(file_path)
    if audio_data.ndim > 1:
        audio_data = audio_data[:, 0]  # Lấy kênh đầu tiên nếu là âm thanh stereo
    return fs, audio_data

# 6. Xây dựng cơ sở dữ liệu
def build_database(data_path: str, db_filename: str, song_index_filename: str):
    songs = glob.glob(f"{data_path}/*")
    song_index = {}
    database = {}
    for idx, song_file in enumerate(tqdm(songs)):
        song_index[idx] = song_file
        try:
            fs, audio = read_audio_file(song_file)
            audio = signal.resample(audio, len(audio) // 4)  # Downsample
            constellation = create_constellation(audio, fs // 4)
            hashes = create_hashes(constellation, idx)
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

    fs, audio = read_audio_file(recording_filename)
    audio = signal.resample(audio, len(audio) // 4)  # Downsample
    constellation = create_constellation(audio, fs // 4)
    hashes = create_hashes(constellation)

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

# Chạy các hàm
if __name__ == "__main__":
    # Ghi âm
    record_audio("output.wav", duration=10)
    # Phân tích tín hiệu
    plot_time_signal("output.wav")
    # Xây dựng cơ sở dữ liệu
    build_database(data_path="data", db_filename="database.pickle", song_index_filename="song_index.pickle")
    # Nhận dạng bài hát
    identify_song("output.wav", db_filename="database.pickle", song_index_filename="song_index.pickle") 
