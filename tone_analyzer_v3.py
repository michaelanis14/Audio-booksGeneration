from __future__ import print_function
import json
#from os.path import join, dirname
from watson_developer_cloud import ToneAnalyzerV3
import sys

tone_analyzer = ToneAnalyzerV3(
    username='e8bed4a3-c201-407f-ad47-24762404a0be',
    password='fHgW6kuvRZsO',
    version='2017-09-26')

#print("\ntone_chat() example 1:\n")
#utterances = [{'text': 'I am very happy.', 'user': 'glenn'},
#              {'text': 'It is a good day.', 'user': 'glenn'}]
while(1):
    utterances= input("Enter Text:")
    print(utterances)
    print("Tone analysis: \n")
    #print("\ntone() example 1:\n")
    print(json.dumps(tone_analyzer.tone(tone_input=utterances,
                                        content_type="text/plain"), indent=2))

#print("\ntone() example 2:\n")
#with open(join(dirname(__file__),
#                '../resources/tone-example.json')) as tone_json:
#     tone = tone_analyzer.tone(json.load(tone_json)['text'], "text/plain")
# print(json.dumps(tone, indent=2))
#
# print("\ntone() example 3:\n")
# with open(join(dirname(__file__),
#                '../resources/tone-example.json')) as tone_json:
#     tone = tone_analyzer.tone(tone_input=json.load(tone_json)['text'],
#                               content_type='text/plain', sentences=True)
# print(json.dumps(tone, indent=2))
#
# print("\ntone() example 4:\n")
# with open(join(dirname(__file__),
#                '../resources/tone-example.json')) as tone_json:
#     tone = tone_analyzer.tone(tone_input=json.load(tone_json),
#                               content_type='application/json')
# print(json.dumps(tone, indent=2))
#
# print("\ntone() example 5:\n")
# with open(join(dirname(__file__),
#                '../resources/tone-example-html.json')) as tone_html:
#     tone = tone_analyzer.tone(json.load(tone_html)['text'],
#                               content_type='text/html')
# print(json.dumps(tone, indent=2))
