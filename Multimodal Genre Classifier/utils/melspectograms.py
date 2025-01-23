import essentia
import numpy as np
from essentia.standard import MonoLoader, Windowing, Spectrum, MelBands, UnaryOperator, FrameGenerator


def load_audio(filename, sampleRate=12000, segment_duration=None):
    audio = MonoLoader(filename=filename, sampleRate=sampleRate, resampleQuality=4)()

    if segment_duration:
        segment_duration = round(segment_duration * sampleRate)
        segment_start = (len(audio) - segment_duration) // 2
        segment_end = segment_start + segment_duration
    else:
        segment_start = 0
        segment_end = len(audio)

    if segment_start < 0 or segment_end > len(audio):
        raise ValueError('Segment duration is larger than the input audio duration')

    return audio[segment_start:segment_end]


def melspectrogram(audio,
                   sampleRate=12000, frameSize=512, hopSize=256,
                   window='hann', zeroPadding=0, center=True,
                   numberBands=96, lowFrequencyBound=0, highFrequencyBound=None,
                   weighting='linear', warpingFormula='slaneyMel',
                   normalize='unit_tri'):
    if highFrequencyBound is None:
        highFrequencyBound = sampleRate / 2

    windowing = Windowing(type=window, normalized=False, zeroPadding=zeroPadding)
    spectrum = Spectrum()
    melbands = MelBands(numberBands=numberBands,
                        sampleRate=sampleRate,
                        lowFrequencyBound=lowFrequencyBound,
                        highFrequencyBound=highFrequencyBound,
                        inputSize=(frameSize + zeroPadding) // 2 + 1,
                        weighting=weighting,
                        normalize=normalize,
                        warpingFormula=warpingFormula,
                        type='power')
    amp2db = UnaryOperator(type='lin2db', scale=2)

    pool = essentia.Pool()
    for frame in FrameGenerator(audio,
                                frameSize=frameSize, hopSize=hopSize,
                                startFromZero=not center):
        pool.add('mel', amp2db(melbands(spectrum(windowing(frame)))))

    return pool['mel'].T


def cut_mel_spectrogram_to_30s(mel_spectrogram, output_path,
                               cut_length=1376):  # 1376 is shorter than 30 second but easily dividable by 2
    """
    Cut a mel spectrogram to 30 seconds, selecting the middle portion of the spectrogram.

    Args:
        mel_spectrogram (str): mel spectrogram in .npy format.
        output_path (str): Path to save the cut mel spectrogram.
        cut_length (int): Number of frames to keep, default is 1406 frames (30 seconds).

    Returns:
        None
    """
    try:

        # Ensure the spectrogram has enough frames
        num_frames = mel_spectrogram.shape[1]
        if num_frames < cut_length:
            raise ValueError(f"The spectrogram is shorter than the required {cut_length} frames.")
        else:

            # Calculate the start and end frames to cut the middle part
            start_frame = (num_frames - cut_length) // 2
            end_frame = start_frame + cut_length

            # Slice the spectrogram to get the 30-second segment
            mel_spectrogram_cut = mel_spectrogram[:, start_frame:end_frame]

            # Save the cut mel spectrogram to spectograms_data folder with the same name
            np.save(output_path, mel_spectrogram_cut)

            print('Spectrogram cut and saved to {}'.format(output_path))


    except ValueError:
        print(f"The spectrogram for is shorter than the required {cut_length} frames.")


def create_mel_spectogram_save_npy_file(input_path, output_path):
    audio_path = input_path
    audio = load_audio(audio_path, segment_duration=None)
    mel_essentia = melspectrogram(audio)
    cut_mel_spectrogram_to_30s(mel_essentia, output_path)