# main.py
import tkinter as tk
from record import record_audio

def start_recording():
    # Gọi hàm record_audio
    record_audio()
    result_label.config(text="Ghi âm hoàn tất!")

def display_result():
    # Gọi hàm calculate_result từ logic.py để lấy kết quả
    search()
    result_label.config(text=f"Kết quả:")

# Tạo cửa sổ chính
window = tk.Tk()
window.title("Giao diện nút bấm và kết quả")
window.geometry("300x150")

# Tạo nút
button = tk.Button(window, text="Ghi âm", command=start_recording)
button.pack(pady=10)

button = tk.Button(window, text="Kết quả", command=display_result)
button.pack(pady=10)

# Tạo nhãn để hiển thị kết quả
result_label = tk.Label(window, text="Kết quả:")
result_label.pack(pady=10)

# Chạy vòng lặp chính của giao diện
window.mainloop()
