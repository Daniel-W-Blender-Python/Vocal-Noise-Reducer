import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
import librosa

# ==== CONFIG ====
SAMPLE_RATE = 16000
CHUNK_SIZE = 8192
BATCH_SIZE = 8
EPOCHS = 200
NPZ_PATH = "/content/drive/MyDrive/Denoiser/denoiser_random_15k.npz"
CHECKPOINT_PATH = "/content/drive/MyDrive/Denoiser/best_denoiser_v6.keras"


# ==== DATA GENERATOR ====
class WaveformGenerator(tf.keras.utils.Sequence):
    def __init__(self, reverb_data, clean_data, indices,
                 chunk_size=CHUNK_SIZE, batch_size=BATCH_SIZE,
                 shuffle=True, augment=True):
        self.reverb_data = reverb_data
        self.clean_data = clean_data
        self.indices = indices
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def normalize(self, x):
        return x / (np.max(np.abs(x)) + 1e-8)

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        X, Y = [], []
        for i in batch_indices:
            clean = self.clean_data[i]
            reverb = self.reverb_data[i]

            min_len = min(len(clean), len(reverb))
            clean, reverb = clean[:min_len], reverb[:min_len]

            if len(clean) > self.chunk_size:
                start = np.random.randint(0, len(clean) - self.chunk_size + 1)
                clean = clean[start:start + self.chunk_size]
                reverb = reverb[start:start + self.chunk_size]

            if len(clean) < self.chunk_size:
                pad_len = self.chunk_size - len(clean)
                clean = np.pad(clean, (0, pad_len))
                reverb = np.pad(reverb, (0, pad_len))

            clean = self.normalize(clean)
            reverb = self.normalize(reverb)

            X.append(reverb[:, np.newaxis].repeat(2, axis=1))
            Y.append(np.stack([clean, clean], axis=-1))

        return np.array(X, np.float32), np.array(Y, np.float32)


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



# ==== LOSS WITH SOFT LOG IN BOTH TIME AND STFT DOMAINS ====
def stft_loss_softlog(y_true, y_pred):
    """
    STFT-based magnitude loss with soft log scaling to emphasize
    quieter (high-frequency) components.
    """
    def _stft(x, frame, hop):
        return tf.signal.stft(x, frame_length=frame, frame_step=hop, pad_end=True)

    total = 0
    for ch in range(y_true.shape[-1]):
        yt, yp = y_true[..., ch], y_pred[..., ch]
        for (frame, hop) in [(256,64), (512,128), (1024,256)]:
            st_y = _stft(yt, frame, hop)
            st_p = _stft(yp, frame, hop)

            mag_diff = tf.abs(tf.abs(st_y) - tf.abs(st_p))
            log_mag = tf.math.log(mag_diff + 1.0)  # soft log
            total += tf.reduce_mean(log_mag)

    return total / y_true.shape[-1]

def smoothness_loss(y_pred):
    diff = y_pred[:, 1:, :] - y_pred[:, :-1, :]
    return tf.reduce_mean(tf.abs(diff))

def phase_consistency_loss(y_true, y_pred):
    st_y = tf.signal.stft(y_true[..., 0], 512, 128)
    st_p = tf.signal.stft(y_pred[..., 0], 512, 128)

    eps = 1e-8
    st_y = tf.complex(tf.math.real(st_y), tf.math.imag(st_y))
    st_p = tf.complex(tf.math.real(st_p), tf.math.imag(st_p))
    mag_y = tf.abs(st_y)
    mag_p = tf.abs(st_p)
    phase_y = tf.math.angle(st_y + eps * tf.complex(tf.ones_like(mag_y), 0.0))
    phase_p = tf.math.angle(st_p + eps * tf.complex(tf.ones_like(mag_p), 0.0))
    phase_diff = tf.abs(tf.cos(phase_y - phase_p))
    return 1.0 - tf.reduce_mean(tf.where(tf.math.is_nan(phase_diff), 0.0, phase_diff))


def hybrid_loss(y_true, y_pred):
    time_diff = tf.abs(y_true - y_pred)
    time_loss = tf.reduce_mean(tf.math.log(time_diff + 1.0))
    freq_loss = stft_loss_softlog(y_true, y_pred)
    smooth_loss = smoothness_loss(y_pred)
    phase_loss = phase_consistency_loss(y_true, y_pred)
    return 0.25 * time_loss + 0.55 * freq_loss + 0.10 * smooth_loss + 0.10 * phase_loss


# ==== LOAD DATA ====
data = np.load(NPZ_PATH)
clean_data, reverb_data = data['clean'], data['noisy']
indices = np.arange(len(clean_data))
np.random.shuffle(indices)

split = int(0.9 * len(indices))
train_gen = WaveformGenerator(reverb_data, clean_data, indices[:split])
val_gen = WaveformGenerator(reverb_data, clean_data, indices[split:], augment=False)


# ==== COMPILE AND TRAIN ====
model = build_denoiser_light()
model.compile(optimizer=optimizers.Adam(1e-4), loss=hybrid_loss)

callbacks_list = [
    callbacks.ModelCheckpoint(CHECKPOINT_PATH, save_best_only=True, monitor='val_loss'),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.6, patience=6),
    callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
]

model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=callbacks_list)
