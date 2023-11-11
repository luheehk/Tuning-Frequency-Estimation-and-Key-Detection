import numpy as np
from scipy.signal import spectrogram
from scipy.signal import find_peaks
import os
import wave


# part C

def load_audio(file_path):
    with wave.open(file_path, 'rb') as audio_file:
        fs = audio_file.getframerate()
        num_frames = audio_file.getnframes()
        x = np.frombuffer(audio_file.readframes(num_frames), dtype=np.int16)
    return x, fs

def eval_tfe(pathToAudio, pathToGT):
    audio_files = os.listdir(pathToAudio)

    total_deviation = 0.0
    num_files = 0

    for audio_file in audio_files:
        if audio_file.endswith(".wav"):
            x, fs = load_audio(os.path.join(pathToAudio, audio_file))
            
            tf_estimate = estimate_tuning_freq(x, blockSize=4096, hopSize=2048, fs=fs)

            gt_file = os.path.join(pathToGT, audio_file.replace(".wav", ".txt"))
            with open(gt_file, "r") as f:
                gt_tf = float(f.read().strip())

            deviation = 1200 * np.log2(tf_estimate / gt_tf)

            total_deviation += np.abs(deviation)
            num_files += 1

    avgDeviation = total_deviation / num_files

    return avgDeviation

def eval_key_detection(pathToAudio, pathToGT):
    audio_files = os.listdir(pathToAudio)

    accuracy_with_tune = 0
    accuracy_without_tune = 0
    num_files = 0

    for audio_file in audio_files:
        if audio_file.endswith(".wav"):
            x, fs = load_audio(os.path.join(pathToAudio, audio_file))

            key_estimate_with_tune = detect_key(x, blockSize=4096, hopSize=2048, fs=fs, bTune=True)
            key_estimate_without_tune = detect_key(x, blockSize=4096, hopSize=2048, fs=fs, bTune=False)

            gt_file = os.path.join(pathToGT, audio_file.replace(".wav", ".txt"))
            with open(gt_file, "r") as f:
                gt_key = f.read().strip()

            # Compare key estimates
            if key_estimate_with_tune == gt_key:
                accuracy_with_tune += 1
            if key_estimate_without_tune == gt_key:
                accuracy_without_tune += 1

            num_files += 1

    # Calculate accuracy
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

mainDirectory = "yout path to /key_tf"

# Evaluation
avg_accuracy, avg_deviationInCent = evaluate(mainDirectory)
print("Average Accuracy (with tune, without tune):", avg_accuracy)
print("Average Deviation in Cents:", avg_deviationInCent)
