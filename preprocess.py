import argparse
import os
from multiprocessing import cpu_count
from tqdm import tqdm
# import load_data
import load_new_data
import hparams


def preprocess_data(args):
  in_dir = os.path.join(args.base_dir, 'Dataset/Training')
  out_dir = os.path.join(args.base_dir, args.output)
  os.makedirs(out_dir, exist_ok=True)
  metadata = load_new_data.build_from_path(in_dir, out_dir, args.num_workers, tqdm=tqdm)
  write_metadata(metadata, out_dir)


def write_metadata(metadata, out_dir):
  with open(os.path.join(out_dir, 'train.txt'), 'w', encoding='utf-8') as f:
    for m in metadata:
      f.write('|'.join([str(x) for x in m]) + '\n')
  # frames = sum([m[2] for m in metadata])
  # hours = frames * hparams.frame_shift_ms / (3600 * 1000)
  # print('Wrote %d utterances, %d frames (%.2f hours)' % (len(metadata), frames, hours))
  # print('Max input length:  %d' % max(len(m[3]) for m in metadata))
  # print('Max output length: %d' % max(m[2] for m in metadata))


def main():
   parser = argparse.ArgumentParser()
   parser.add_argument('--base_dir', default=os.path.expanduser('~/Documents/Neural Network Happy/'))
   parser.add_argument('--output', default='training-new')
  # parser.add_argument('--dataset', required=True, choices=['blizzard', 'ljspeech'])
   parser.add_argument('--num_workers', type=int, default=cpu_count())
   args = parser.parse_args()
   preprocess_data(args)


if __name__ == "__main__":
  main()
