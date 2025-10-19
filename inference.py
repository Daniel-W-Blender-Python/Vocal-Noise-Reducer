import numpy as np
import tensorflow as tf
import librosa
import soundfile as sf
from tensorflow.keras import layers, models
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, filtfilt

# ==== CONFIG ====
SAMPLE_RATE = 16000
CHUNK_SIZE = 8192
HOP_SIZE = CHUNK_SIZE // 2  # 50% overlap
WEIGHT_PATH = "./best_denoiser_v6.keras"
INPUT_FILE = "./sample_noisy.wav"
OUTPUT_FILE = "./denoised_audio.wav"


# ==== LIGHT-WEIGHT MODEL ====
def residual_block_light(x, filters, kernel=9, stride=1):
    skip = x
    x = layers.Conv1D(filters, kernel, padding='same', strides=stride)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv1D(filters, kernel, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    if stride == 1 and skip.shape[-1] == filters:
        x = layers.Add()([x, skip])
    return layers.Activation('relu')(x)


def build_denoiser_light(input_length=CHUNK_SIZE, num_channels=2):
    inp = layers.Input(shape=(input_length, num_channels))

    # ==== Encoder ====
    e1 = residual_block_light(inp, 32)
    e2 = residual_block_light(layers.Conv1D(64, 9, strides=2, padding='same')(e1), 64)
    e3 = residual_block_light(layers.Conv1D(128, 9, strides=2, padding='same')(e2), 128)
    e4 = residual_block_light(layers.Conv1D(256, 9, strides=2, padding='same')(e3), 256)

    # ==== Bottleneck with smaller ConvLSTM ====
    b = layers.Lambda(lambda x: tf.expand_dims(x, axis=1))(e4)
    b = layers.Lambda(lambda x: tf.expand_dims(x, axis=2))(b)
    b = layers.ConvLSTM2D(128, (1,3), padding='same', return_sequences=False)(b)
    b = layers.Lambda(lambda x: tf.squeeze(x, axis=1))(b)
    b = layers.Conv1D(256, 9, padding='same', activation='relu')(b)

    # ==== Decoder ====
    d3 = layers.Conv1DTranspose(128, 9, strides=2, padding='same', activation='relu')(b)
    d3 = layers.Concatenate()([d3, e3])
    d2 = layers.Conv1DTranspose(64, 9, strides=2, padding='same', activation='relu')(d3)
    d2 = layers.Concatenate()([d2, e2])
    d1 = layers.Conv1DTranspose(32, 9, strides=2, padding='same', activation='relu')(d2)
    d1 = layers.Concatenate()([d1, e1])

    out = layers.Conv1D(num_channels, 1, padding='same', activation='tanh')(d1)

    return models.Model(inp, out, name="UNetResidualDenoiser_Light")


# ==== LOAD MODEL WEIGHTS ====
print("Rebuilding model...")
model = build_denoiser_light()
print("Loading weights...")
model.load_weights(WEIGHT_PATH)
print("✅ Model successfully restored!")


# ==== NORMALIZATION ====
def normalize(x):
    return x / (np.max(np.abs(x)) + 1e-8)


def denoise_audio(model, audio, chunk_size=CHUNK_SIZE, hop_size=HOP_SIZE, blend=0.9):
    """
    Enhanced inference with overlap-add smoothing, crossfades, and residual blending.
    """
    # Normalize input
    audio = audio - np.mean(audio)
    audio = audio / (np.max(np.abs(audio)) + 1e-8)

    num_chunks = max(1, (len(audio) - chunk_size) // hop_size + 1)
    output = np.zeros(len(audio) + chunk_size)
    norm = np.zeros_like(output)
    window = np.hanning(chunk_size)
    prev_pred = np.zeros(chunk_size)

    for i in range(num_chunks):
        start = i * hop_size
        end = start + chunk_size
        chunk = audio[start:end]
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)))

        # Model input (2 identical channels)
        inp = np.stack([chunk, chunk], axis=-1)[np.newaxis, ...]
        pred = model.predict(inp, verbose=0)[0, :, 0]

        # (2) Crossfade between chunks to avoid seams
        if i > 0:
            overlap_len = hop_size
            pred[:overlap_len] = (
                0.5 * pred[:overlap_len] + 0.5 * prev_pred[-overlap_len:]
            )

        prev_pred = pred.copy()

        # (3) Overlap-add with window
        output[start:end] += pred * window
        norm[start:end] += window

    # --- Normalize reconstructed signal ---
    enhanced = output[:len(audio)] / (norm[:len(audio)] + 1e-8)

    # (5) Remove DC offset & normalize
    enhanced -= np.mean(enhanced)
    enhanced /= (np.max(np.abs(enhanced)) + 1e-8)

    # (7) Fade in/out to avoid boundary clicks
    fade_len = min(2048, len(enhanced) // 20)
    fade = np.linspace(0, 1, fade_len)
    enhanced[:fade_len] *= fade
    enhanced[-fade_len:] *= fade[::-1]

    return enhanced



# ==== RUN INFERENCE ====
print("Loading input audio...")
noisy_audio, sr = librosa.load(INPUT_FILE, sr=SAMPLE_RATE)
print("Running denoiser...")
enhanced = denoise_audio(model, noisy_audio)
sf.write(OUTPUT_FILE, enhanced, SAMPLE_RATE)
print(f"✅ Denoised file saved at: {OUTPUT_FILE}")
