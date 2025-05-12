import librosa
import numpy as np
import tensorflow as tf
import soundfile as sf
# Load your trained model
model = tf.keras.models.load_model("noise_reduction_model_test_1 (1).keras")

def process_audio(input_file, output_file):
  # 1. Load the audio file
  audio, sr = librosa.load(input_file, sr=None)  # Load with native sample rate

  # 2. Preprocess the audio (adjust if needed)
  # Here, we assume your model expects a single-channel audio
  audio = audio.reshape(-1)  

  # 3. Apply the model for noise reduction
  denoised_audio = model.predict(audio.reshape(1, -1))  # Reshape for prediction
  denoised_audio = denoised_audio.reshape(-1)  # Reshape back

  # 4. Postprocess (if needed)

  # 5. Save the processed audio
  sf.write(output_file, denoised_audio, sr)

# Example usage
input_audio_path = ""
output_audio_path =  ""
process_audio(input_audio_path, output_audio_path)

print(f"✅ Processed audio saved to: {output_audio_path}")
