import librosa
import numpy as np

def mp3_to_mel_spectrogram(mp3_file_path, cut=True, cut_length=1024):

    signal, sr = librosa.load(mp3_file_path)
    mel_spectrogram = librosa.feature.melspectrogram(y=signal, sr=sr, n_fft=2048, hop_length=512, n_mels=96)

    if cut:
        num_frames = mel_spectrogram.shape[1]
        if num_frames < cut_length:
            raise ValueError(f"The spectrogram is shorter than the required {cut_length} frames.")
        else:

            # Calculate the start and end frames to cut the middle part
            start_frame = (num_frames - cut_length) // 2
            end_frame = start_frame + cut_length

            # Slice the spectrogram to get the 30-second segment
            mel_spectrogram = mel_spectrogram[:, start_frame:end_frame]

    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)

    return log_mel_spectrogram


def save_mel_spectrogram(mel_spectrogram, output_path):
    np.save(output_path, mel_spectrogram)
    print('Spectrogram saved to {}'.format(output_path))

def mel_spectrogram(mp3_file_path, output_path, cut=True, cut_length=1024):
    mel_spectrogram = mp3_to_mel_spectrogram(mp3_file_path, cut=cut, cut_length=cut_length)
    save_mel_spectrogram(mel_spectrogram, output_path)
