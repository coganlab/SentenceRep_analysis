#%%
import os
import wave
import numpy as np

def get_wav_duration(file_path):
    with wave.open(file_path, 'rb') as wav_file:
        # Get the number of frames in the wav file
        num_frames = wav_file.getnframes()
        # Get the frame rate (number of frames per second)
        frame_rate = wav_file.getframerate()
        # Calculate the duration in seconds
        duration = num_frames / float(frame_rate)
        return duration

def get_audio_durations(directory):
    durations = {}
    for filename in os.listdir(directory):
        if filename.endswith(".wav"):
            file_path = os.path.join(directory, filename)
            try:
                duration = get_wav_duration(file_path)
                durations[filename] = duration
            except Exception as e:
                print(f"Could not process {filename}: {e}")
    return durations


HOME = os.path.expanduser("~")
directory_path = os.path.join(HOME, "Box", "CoganLab", "BIDS-1.4_Phoneme_sequencing", "BIDS", "stimuli")
durations = get_audio_durations(directory_path)

duration_vals = np.array(list(durations.values()))
mean = np.mean(duration_vals)
sd = np.std(duration_vals)
print(f'mean={mean} \n sd={sd}')