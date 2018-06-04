from concurrent.futures import ProcessPoolExecutor
from functools import partial
import numpy as np
import os
from util import audio


def build_from_path(in_dir, out_dir, num_workers=1, tqdm=lambda x: x):
    '''Preprocesses the LJ Speech dataset from a given input path into a given output directory.

    Args:
      in_dir: The directory where you have downloaded the LJ Speech dataset
      out_dir: The directory to write the output into
      num_workers: Optional number of worker processes to parallelize across
      tqdm: You can optionally pass tqdm to get a nice progress bar

    Returns:
      A list of tuples describing the training examples. This should be written to train.txt
  '''

  # We use ProcessPoolExecutor to parallize across processes. This is just an optimization and you
  # can omit it and just call _process_utterance on each input if you want.
    executor = ProcessPoolExecutor(max_workers=num_workers)
    futures = []
    index = 1
    iteration=0;
    wav_path_neutral=""
    wav_path_happy=""

    for filename in os.listdir(in_dir):
        if(filename.endswith('neutral.wav')):
             wav_path_neutral = os.path.join(in_dir, filename)
        elif(filename.endswith('happy.wav')):
             wav_path_happy = os.path.join(in_dir, filename)
        iteration+=1
        if(iteration%2 == 0):
             task=partial(_process_utterance, out_dir, index, wav_path_neutral, wav_path_happy)
             futures.append(executor.submit(task))

             index += 1
    results = [future.result() for future in tqdm(futures)]
    return [r for r in results if r is not None]


def _process_utterance(out_dir, index, wav_path_neutral, wav_path_happy):
    '''Preprocesses a single utterance audio/text pair.

    This writes the mel and linear scale spectrograms to disk and returns a tuple to write
    to the train.txt file.

    Args:
    out_dir: The directory to write the spectrograms into
    index: The numeric index to use in the spectrogram filenames.
    wav_path: Path to the audio file containing the speech input
    text: The text spoken in the input audio file

    Returns:
    A (spectrogram_filename, mel_filename, n_frames, text) tuple to write to train.txt
    '''
    # Load the audio to a numpy array:
    wav1 = audio.load_wav(wav_path_neutral)
    wav2 = audio.load_wav(wav_path_happy)

    # Compute the neutral linear-scale spectrogram from the wav:
    spectrogram_neutral = audio.spectrogram(wav1).astype(np.float32)
    n_frames = spectrogram_neutral.shape[1]
    # Compute a neutral mel-scale spectrogram from the wav:
    mel_spectrogram_neutral = audio.melspectrogram(wav1).astype(np.float32)

    spectrogram_happy= audio.spectrogram(wav2).astype(np.float32)
    n_frames = spectrogram_happy.shape[1]
    mel_spectrogram_happy  =  audio.melspectrogram(wav2).astype(np.float32)

    # Write the spectrograms to disk:
    spectrogram_neutral_filename = 'neutral-spec-%05d.npy' % index
    mel_neutral_filename = 'neutral-mel-%05d.npy' % index
    np.save(os.path.join(out_dir, spectrogram_neutral_filename), spectrogram_neutral.T, allow_pickle=False)
    np.save(os.path.join(out_dir, mel_neutral_filename), mel_spectrogram_neutral.T, allow_pickle=False)

    spectrogram_happy_filename = 'happy-spec-%05d.npy' % index
    mel_happy_filename = 'happy-mel-%05d.npy' % index
    np.save(os.path.join(out_dir, spectrogram_happy_filename), spectrogram_happy.T, allow_pickle=False)
    np.save(os.path.join(out_dir, mel_happy_filename), mel_spectrogram_happy.T, allow_pickle=False)

    # Return a tuple describing this training example:
    return (spectrogram_neutral_filename, mel_neutral_filename, spectrogram_happy_filename, mel_happy_filename, n_frames)
