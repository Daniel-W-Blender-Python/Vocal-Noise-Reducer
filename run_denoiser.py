import argparse
import librosa
import numpy as np
import tensorflow as tf
import soundfile as sf

model = tf.keras.models.load_model("noise_reduction_model_test_1 (1).keras")

def process_audio(input_file, output_file):
    audio, sr = librosa.load(input_file, sr=None)

    audio = audio.reshape(-1)

    denoised_audio = model.predict(audio.reshape(1, -1))
    denoised_audio = denoised_audio.reshape(-1)

    sf.write(output_file, denoised_audio, sr)
    print(f"✅ Processed audio saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Denoise an audio file using a trained model.")
    parser.add_argument("input_audio_path", type=str, help="Path to the input audio file")
    parser.add_argument("output_audio_path", type=str, help="Path to save the denoised output audio")
    args = parser.parse_args()

    process_audio(args.input_audio_path, args.output_audio_path)
