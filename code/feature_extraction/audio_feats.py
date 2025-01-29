#import sys
#sys.path.append('/home/mb2554/.local/lib/python3.8/site-packages')
import librosa
import numpy as np
import os


def extract_audio_spectrogram(audio_path, fps=30):
    # Load audio at 60kHz
    y, sr = librosa.load(audio_path, sr=48000)

    # Calculate window size and hop length to match video fps
    samples_per_frame = int(sr/fps)  # Samples per frame
    print(f'Samples per frame: {samples_per_frame}')
    #hop_length = window_size   # Non-overlapping windows

    # Split audio into frame-aligned segments
    n_frames = len(y) // samples_per_frame
    print(f'Extracting {n_frames} frames from audio...')
    audio_frames = y[:n_frames*samples_per_frame].reshape(-1, samples_per_frame)
    
    print(f'Extracted {len(audio_frames)} frames.')
    
    spectrograms = []
    for frame_audio in audio_frames:
        mel_spect = librosa.feature.melspectrogram(
            y=frame_audio,
            sr=sr,
            n_mels=128,
            n_fft=samples_per_frame,
            hop_length=samples_per_frame//128
        )
        mel_spect = mel_spect[:, :128]
        #print(mel_spect.shape)
        mel_spect_db = librosa.power_to_db(mel_spect, ref=np.max)
        #print(mel_spect_db.shape)
        spectrograms.append(mel_spect_db)
    
    return np.array(spectrograms)  # Shape: (n_frames, 128, 128)


# Example usage
if __name__ == '__main__':
    audio_path = '../../data/audio/'
    audio_files = os.listdir(audio_path)
    for audio_file in audio_files:
        #see if this already exists
        if os.path.exists(f'../../data/audio_features/{audio_file}.npy'):
            print(f'File {audio_file} already exists. Skipping...')
            continue
        print(f'Processing {audio_file}...')
        audio_path_full = os.path.join(audio_path, audio_file)
        mel_spect = extract_audio_spectrogram(audio_path_full)
        print(mel_spect.shape)
        #save the mel_spect to a file
        np.save(f'../../data/audio_features/{audio_file}.npy', mel_spect)


   