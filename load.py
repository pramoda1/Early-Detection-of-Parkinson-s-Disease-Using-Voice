import os
import joblib
import librosa
import sounddevice as sd
import numpy as np
import pyttsx3
import tkinter as tk
from tkinter import Label
from scipy.io.wavfile import write
import traceback
import warnings
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Suppress warnings
warnings.filterwarnings("ignore")

# Paths
MODEL_PATH = r"model.pkl"

# Load model
def load_model(model_path):
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"The model file does not exist at the path: {model_path}")
        print("Loading the model using joblib...")
        model = joblib.load(model_path)
        print("Model loaded successfully!")
        return model
    except Exception as e:
        print("Failed to load the model using joblib. Error:", e)
        traceback.print_exc()
        return None

model = load_model(MODEL_PATH)

# Record voice
def record_audio(sample_rate=22050, duration=3):
    print("Recording started. Speak now...")
    audio = sd.rec(int(sample_rate * duration), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    print("Recording stopped!")
    if np.max(np.abs(audio)) < 0.01:
        print("No significant sound detected. Please speak louder.")
        return None, None
    return np.squeeze(audio), sample_rate

# Feature extraction
def extract_voice_features(audio, sr):
    spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=43, fmax=8000)
    spectrogram_db = librosa.amplitude_to_db(spectrogram, ref=np.max)
    if spectrogram_db.shape[1] < 232:
        pad_width = 232 - spectrogram_db.shape[1]
        spectrogram_db = np.pad(spectrogram_db, ((0, 0), (0, pad_width)), mode='constant')
    elif spectrogram_db.shape[1] > 232:
        spectrogram_db = spectrogram_db[:, :232]
    return spectrogram_db.reshape(1, 43, 232, 1)

# Predict voice and return result with probability
def predict_voice(model, audio, sr):
    if model is None:
        return "Model not loaded", 0.0
    features = extract_voice_features(audio, sr)
    probability = model.predict(features)[0][0]
    result = "Parkinson Detected" if probability > 0.4 else "Healthy"
    return result, probability

# Text-to-speech
def speak_output(result):
    engine = pyttsx3.init()
    engine.setProperty("rate", 150)
    engine.say(f"The prediction is {result}.")
    engine.runAndWait()

# Plot waveform
def plot_waveform(audio, sr):
    global waveform_canvas
    if waveform_canvas:
        waveform_canvas.get_tk_widget().destroy()
        waveform_canvas = None

    fig, ax = plt.subplots(figsize=(5, 2))
    t = np.linspace(0, len(audio) / sr, num=len(audio))
    ax.plot(t, audio)
    ax.set_title("Voice Waveform")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.grid(True)

    waveform_canvas = FigureCanvasTkAgg(fig, master=root)
    waveform_canvas.draw()
    waveform_canvas.get_tk_widget().pack(pady=5)

# RMS loudness analysis (fallback)
def analyze_loudness(audio):
    rms = np.sqrt(np.mean(np.square(audio)))
    print(f"RMS value: {rms}")
    return "Healthy" if rms > 0.02 else "Parkinson Detected"

# Display result with optional probability
def display_results(result, audio=None, sr=None, prob=None):
    info_label.pack_forget()
    if prob is not None:
        accuracy = f"{prob * 100:.2f}%"
        result_label.config(text=f"Prediction: {result} ({accuracy} confidence)")
    else:
        result_label.config(text=f"Prediction: {result}")

    if result == "Healthy":
        root.config(bg="green")
        result_label.config(bg="green")
    elif result == "Parkinson Detected":
        root.config(bg="red")
        result_label.config(bg="red")
    else:
        root.config(bg="yellow")
        result_label.config(bg="yellow")

    speak_output(result)

    if audio is not None and sr is not None:
        plot_waveform(audio, sr)

# Check button action
def check_normal_voice():
    audio, sr = record_audio()
    if audio is None:
        display_results("Please speak louder for accurate prediction.")
        return
    write("normal_voice.wav", sr, (audio * 32767).astype(np.int16))
    if model is not None:
        result, prob = predict_voice(model, audio, sr)
        display_results(result, audio, sr, prob)
    else:
        result = analyze_loudness(audio)
        display_results(result, audio, sr)

# GUI setup
root = tk.Tk()
root.title("Parkinson's Detection Test")
root.geometry("550x500")

waveform_canvas = None

info_text = (
    "Parkinson's disease is a progressive nervous system disorder "
    "that affects movement and speech. Early detection through voice "
    "analysis may help in timely intervention."
)
info_label = Label(root, text=info_text, wraplength=480, justify="left", font=("Arial", 10))
info_label.pack(pady=10)

Label(root, text="Select Voice Type to Test", font=("Arial", 14)).pack(pady=5)

result_label = Label(root, text="", font=("Arial", 14), width=40, height=2)
result_label.pack(pady=10)

tk.Button(root, text="Check Patient's Voice", command=check_normal_voice, font=("Arial", 12), bg="red", fg="white").pack(pady=5)
tk.Button(root, text="Check Normal Voice", command=check_normal_voice, font=("Arial", 12), bg="green", fg="white").pack(pady=5)
tk.Button(root, text="Close", command=root.destroy, font=("Arial", 12)).pack(pady=10)

root.mainloop()
