import numpy as np
from scipy.signal import spectrogram
from scipy.signal import find_peaks
import os
import wave


import numpy as np
from scipy.signal import spectrogram
from scipy.signal import find_peaks
import os
import wave

# A
def get_spectral_peaks(X):
    num_peaks = 20
    spectralPeaks = np.zeros((num_peaks, X.shape[1]))

    for i in range(X.shape[1]):
        # Find the spectral peaks for the current time frame
        peaks, _ = find_peaks(X[:, i], height=0)

        # Sort the peaks by magnitude and take the top 20
        sorted_peaks = np.argsort(X[peaks, i])[::-1][:num_peaks]

        # Store the top 20 spectral peak bins for this time frame
        spectralPeaks[:, i] = peaks[sorted_peaks]

    return spectralPeaks

def estimate_tuning_freq(x, blockSize, hopSize, fs):
    _, _, Sxx = spectrogram(x, fs=fs, nperseg=blockSize, noverlap=blockSize-hopSize)

    spectralPeaks = get_spectral_peaks(np.abs(Sxx))

    cents_scale = np.arange(0, 1200, 100)

    deviations_in_cents = np.zeros(spectralPeaks.shape)

    for i in range(spectralPeaks.shape[1]):
        for j in range(spectralPeaks.shape[0]):
            freq_in_hz = spectralPeaks[j, i] * fs / blockSize
            pitch_in_cents = 1200 * np.log2(freq_in_hz / 440.0)
            deviations_in_cents[j, i] = np.min(np.abs(cents_scale - pitch_in_cents))

    hist, bin_edges = np.histogram(deviations_in_cents, bins=np.arange(0, 600, 50))

    max_bin_index = np.argmax(hist)

    tfInHz = 440.0 * 2 ** (cents_scale[max_bin_index] / 1200)

    return tfInHz

# B
def extract_pitch_chroma(X, fs, tfInHz):
    num_octaves = 3
    num_semitones_per_octave = 12
    num_blocks = X.shape[1]
    pitchChroma = np.zeros((num_semitones_per_octave, num_blocks))

    semitone_step = 12 * np.log2(tfInHz / 440.0)

    for i in range(num_blocks):
        for j in range(num_semitones_per_octave):
            freq_bin = int(round(j * semitone_step))
            freq_bin %= X.shape[0]
            pitchChroma[j, i] = np.sum(X[freq_bin::num_semitones_per_octave, i])

        pitchChroma[:, i] /= np.linalg.norm(pitchChroma[:, i])

    return pitchChroma

def detect_key(x, blockSize, hopSize, fs, bTune):
    key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    t_pc_major = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    t_pc_minor = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

    _, _, Sxx = spectrogram(x, fs=fs, nperseg=blockSize, noverlap=blockSize-hopSize)

    tfInHz = 440.0
    if bTune:
        tfInHz = estimate_tuning_freq(x, blockSize, hopSize, fs)

    pitchChroma = extract_pitch_chroma(np.abs(Sxx), fs, tfInHz)

    distance_major = np.linalg.norm(pitchChroma - t_pc_major[:, np.newaxis], axis=0)
    distance_minor = np.linalg.norm(pitchChroma - t_pc_minor[:, np.newaxis], axis=0)

    if np.min(distance_major) < np.min(distance_minor):
        key_index = np.argmin(distance_major)
        keyEstimate = key_names[key_index]
    else:
        key_index = np.argmin(distance_minor)
        keyEstimate = key_names[key_index]

    return keyEstimate

# C
import os
import numpy as np
import wave

# Set the block size and hop size for spectrogram computation
blockSize = 4096
hopSize = 2048

# Set the number of spectral peaks for tuning frequency estimation
num_peaks = 20

# Set the cents scale for tuning frequency estimation
cents_scale = np.arange(0, 1200, 100)

# Set the tuning frequency correction flag
bTune = True  # True if you want to apply tuning frequency correction, False otherwise

# Set the directories for audio and ground truth data
mainDirectory = "/Users/shinhyunkyung/Downloads/key_tf"  # Replace with the path to your data directory
gt_tf_dir = os.path.join(mainDirectory, "tuning_eval/GT")  # Adjust the subdirectory accordingly
gt_key_dir = os.path.join(mainDirectory, "key_eval/GT")  # Adjust the subdirectory accordingly

def load_audio(file_path):
    with wave.open(file_path, 'rb') as audio_file:
        fs = audio_file.getframerate()
        num_frames = audio_file.getnframes()
        x = np.frombuffer(audio_file.readframes(num_frames), dtype=np.int16)
    return x, fs

def eval_tfe(pathToAudio, pathToGT):
    """
    Evaluates tuning frequency estimation for the audio files in the given directory.

    Parameters:
    pathToAudio (str): Path to the directory containing audio files.
    pathToGT (str): Path to the directory containing ground truth tuning frequency files.

    Returns:
    avgDeviation (float): Average absolute deviation of tuning frequency estimation in cents.
    """
    # Get a list of audio file names
    audio_files = os.listdir(pathToAudio)

    total_deviation = 0.0
    num_files = 0

    for audio_file in audio_files:
        if audio_file.endswith(".wav"):
            # Load the audio file
            x, fs = load_audio(os.path.join(pathToAudio, audio_file))

            # Compute the tuning frequency estimation
            tf_estimate = estimate_tuning_freq(x, blockSize=4096, hopSize=2048, fs=fs)

            # Load the ground truth tuning frequency
            gt_file = os.path.join(pathToGT, audio_file.replace(".wav", ".txt"))
            with open(gt_file, "r") as f:
                gt_tf = float(f.read().strip())

            # Calculate the absolute deviation in cents
            deviation = 1200 * np.log2(tf_estimate / gt_tf)

            total_deviation += np.abs(deviation)
            num_files += 1

    # Calculate the average absolute deviation in cents
    avgDeviation = total_deviation / num_files

    return avgDeviation

def eval_key_detection(pathToAudio, pathToGT):
    """
    Evaluates key detection for the audio files in the given directory.

    Parameters:
    pathToAudio (str): Path to the directory containing audio files.
    pathToGT (str): Path to the directory containing ground truth key label files.

    Returns:
    accuracy (numpy.ndarray): Array with accuracy values for key detection with and without tuning frequency correction.
    """
    # Get a list of audio file names
    audio_files = os.listdir(pathToAudio)

    accuracy_with_tune = 0
    accuracy_without_tune = 0
    num_files = 0

    for audio_file in audio_files:
        if audio_file.endswith(".wav"):
            # Load the audio file
            x, fs = load_audio(os.path.join(pathToAudio, audio_file))

            # Compute key detection with tuning frequency correction
            key_estimate_with_tune = detect_key(x, blockSize=4096, hopSize=2048, fs=fs, bTune=True)

            # Compute key detection without tuning frequency correction
            key_estimate_without_tune = detect_key(x, blockSize=4096, hopSize=2048, fs=fs, bTune=False)

            # Load the ground truth key label
            gt_file = os.path.join(pathToGT, audio_file.replace(".wav", ".txt"))
            with open(gt_file, "r") as f:
                gt_key = f.read().strip()

            # Compare key estimates with ground truth
            if key_estimate_with_tune == gt_key:
                accuracy_with_tune += 1
            if key_estimate_without_tune == gt_key:
                accuracy_without_tune += 1

            num_files += 1

    # Calculate accuracy for key detection with and without tuning frequency correction
    accuracy = np.array([accuracy_with_tune / num_files, accuracy_without_tune / num_files])

    return accuracy

def evaluate(mainDirectory):
    audio_key_dir = os.path.join(mainDirectory, "key_eval/audio")
    gt_key_dir = os.path.join(mainDirectory, "key_eval/GT")
    audio_tf_dir = os.path.join(mainDirectory, "tuning_eval/audio")
    gt_tf_dir = os.path.join(mainDirectory, "tuning_eval/GT")

    avg_deviationInCent = eval_tfe(audio_tf_dir, gt_tf_dir)

    avg_accuracy = eval_key_detection(audio_key_dir, gt_key_dir)

    return avg_accuracy, avg_deviationInCent

mainDirectory = "/Users/shinhyunkyung/Downloads/key_tf"

avg_accuracy, avg_deviationInCent = evaluate(mainDirectory)
print("Average Accuracy (with tune, without tune):", avg_accuracy)
print("Average Deviation in Cents:", avg_deviationInCent)
