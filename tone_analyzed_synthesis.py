from __future__ import print_function
from synthesizer import Synthesizer
from hparams import hparams, hparams_debug_string
import os
import json
from watson_developer_cloud import ToneAnalyzerV3
import sys
import argparse
import re

tone_analyzer = ToneAnalyzerV3(
    username='e8bed4a3-c201-407f-ad47-24762404a0be',
    password='fHgW6kuvRZsO',
    version='2017-09-26')

synthesizer = Synthesizer()

output_file = open('experiment_results','w')

def get_output_base_path(checkpoint_path):
  base_dir = os.path.dirname(checkpoint_path)
  m = re.compile(r'.*?\.ckpt\-([0-9]+)').match(checkpoint_path)
  name = 'eval-%d' % int(m.group(1)) if m else 'eval'
  return os.path.join(base_dir, name)

def run_eval(args, text):
  print(hparams_debug_string())
  synth = Synthesizer()
  synth.load(args.checkpoint)
  base_path = get_output_base_path(args.checkpoint)
  path = '%s.wav' % (text)
  print('Synthesizing: %s' % path)
  with open(path, 'wb') as f:
    f.write(synth.synthesize(text))

def analyze_tone(text):
    output_file.write(text)
    output_file.write(json.dumps(tone_analyzer.tone(tone_input=text,
                                        content_type="text/plain"), indent=2))

def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    # parser.add_argument('--hparams', default='',
    # help='Hyperparameter overrides as a comma-separated list of name=value pairs')
    # args = parser.parse_args()
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    # hparams.parse(args.hparams)
    text= [
        "I was trying to find you to warn you, I heard Malfoy saying he was going to catch you, he said you had a drag",
        "Harry shook his head violently to shut Neville up, but Professor McGonagall had seen", "She looked more likely to breathe fire than Norbert as she towered over the three of them",
        "I would never have believed it of any of you",
        "Mr. Filch says you were up in the astronomy tower.",
        "It's one o'clock in the morning.",
        "Explain yourselves.",
        "It was the first time Hermione had ever failed to answer a teacher's question.",
        "She was staring at her slippers, as still as a statue",
        "I think I've got a good idea of what's been going on,",
        "It doesn't take a genius to work it out.",
        "You fed Draco Malfoy some cock-and-bull story about a dragon, trying to get him out of bed and into trouble",
        "I've already caught him.",
        "I suppose you think it's funny that Longbottom here heard the story and believed it, too?",
        "Harry caught Neville's eye and tried to tell him without words that this wasn't true, because Neville was looking stunned and hurt",
        "Poor, blundering Neville -- Harry knew what it must have cost him to try and find them in the dark, to warn them.",
        "I am disgusted, said Professor McGonagall.",
        "Four students out of bed in one night!",
        "I have never heard of such a thing before! You, Miss Granger, I thought you had more sense.",
        "As for you, Mr. Potter, I thought Gryffindor meant more to you than this.",
        "All three of you will receive detentions -- yes, you too, Mr. Longbottom,",
        "nothing gives you the right to walk around school at night, especially these days, it's very dangerous -- and fifty points will be taken from Gryffindor." ,
        "Fifty? Harry gasped",
        "they would lose the lead, the lead he'd won in the last Quidditch match.",
        "Fifty points each,",
        "Professor -- please",
        "You can't --",
        "Don't tell me what I can and can't do, Potter",
        "Now get back to bed, all of you. I've never been more ashamed of Gryffindor students.",
        "They'll all forget this in a few weeks. Fred and George have lost loads of points in all the time they've been here, and people still like them.",
        "They've never lost a hundred and fifty points in one go, though, have they? said Harry miserably.",
        "Well -- no, Ron admitted",
        "It was a bit late to repair the damage, but Harry swore to himself not to meddle in things that weren't his business from now on.",
        "He felt so ashamed of himself that he went to Wood and offered to resign from the Quidditch team",
        "Resign?  Wood thundered." ,
        "What good'll that do? How are we going to get any points back if we can't win at Quidditch?",
        "But even Quidditch had lost its fun.",
        "The rest of the team wouldn't speak to Harry during practice, and if they had to speak about him, they called him the Seeker.",
        "No -- no -- not again, please --",
        "Thanks for the fudge and the sweater, Mrs. Weasley.",
        "Oh, it was nothing, dear",
        "Ready, are you?",
        "Behind him stood Aunt Petunia and Dudley, looking terrified at the very sight of Harry.",
        "You must be Harry's family!",
        "Hurry up, boy, we haven't got all day",
        "See you over the summer, then.",
        "Hope you have -- er -- a good holiday"
        "Oh, I will",
        "They don't know we're not allowed to use magic at home. I'm going to have a lot of fun with Dudley this summer",
        "Still famous, said Ron, grinning at him"

    ]

    for t in text:
        analyze_tone(t)
    print('done')
    #run_eval(args,text)

if __name__ == '__main__':
  main()
