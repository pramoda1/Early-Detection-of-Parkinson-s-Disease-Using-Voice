🧠 Parkinson’s Detection Using Voice Analysis
A desktop application built with Python, Tkinter, and Machine Learning to detect signs of Parkinson’s disease through voice recordings. The system analyzes vocal features using a trained model and provides an instant prediction, helping with potential early diagnosis.

🎯 Features
🎤 Voice Recording: Real-time voice capture using your microphone.

🧪 Voice-Based Prediction: Uses a trained machine learning model to detect Parkinson’s based on voice features.

📊 Waveform Visualization: Displays recorded voice waveform for analysis.

🔊 Text-to-Speech Output: Announces the prediction result using TTS.

🧠 Fallback Heuristic: If model fails, uses RMS-based loudness analysis for an alternative prediction.

🖥️ GUI: Simple Tkinter-based graphical user interface.

🛠️ Tech Stack
Language: Python 3

GUI: Tkinter

Libraries:

librosa – for audio feature extraction

sounddevice – to record voice

scipy.io.wavfile – to save recordings

matplotlib – to plot audio waveform

joblib – to load the ML model

pyttsx3 – for speech synthesis

Model: Pre-trained ML model (model.pkl) trained on voice data

📁 Project Structure
graphql
Copy
Edit
ParkinsonVoiceDetection/
├── load.py               # Main GUI and logic script
├── model.pkl             # Pre-trained ML model (expected in root)
├── normal_voice.wav      # Output recording file
▶️ How It Works
Press "Check Patient's Voice" or "Check Normal Voice" in the GUI.

Speak into the mic when prompted.

The voice is recorded and features are extracted using Mel Spectrogram.

The model predicts the likelihood of Parkinson’s and displays the result.

Voice waveform is plotted and result is spoken aloud.

⚙️ Setup Instructions
🔧 Prerequisites
Install the required packages:

bash
Copy
Edit
pip install joblib librosa sounddevice numpy pyttsx3 matplotlib
If you're using Windows, also run:

bash
Copy
Edit
pip install pyaudio
🏃 Run the Application
bash
Copy
Edit
python load.py
🧪 Model File
Make sure model.pkl is present in the root directory. You can train your own model using spectrograms of patient and healthy voice samples and save it with:

python
Copy
Edit
joblib.dump(model, "model.pkl")
📊 Example Prediction Output
Healthy

Parkinson Detected

Confidence Score is displayed if available from the model.

💬 Future Improvements
Add option to load .wav files instead of recording only.

Export prediction logs.

Enhance model accuracy with larger datasets.

🤝 Contribution
Feel free to fork this project and contribute via pull requests or issues. Suggestions are always welcome!

🧾 License
This project is open-source and intended for educational purposes only. It is not a certified medical diagnostic tool.
