
import io
import os
from os.path import join

from promo import f0_morph
from praatio import audioio
from praatio import dataio
from praatio import praat_scripts
from praatio import pitch_and_intensity
from praatio import tgio

# We need to use absolute paths when using praatEXE
inputPath = os.path.abspath(join('..', 'emotional_dataset'))
outputPath= os.path.abspath(join(inputPath, "pitch_tutorial_output"))


if not os.path.exists(outputPath):
    os.mkdir(outputPath)

########
# STEP 1: Load all the pre-defined parameters

# Pick your poison
praatEXE = "/home/youmna/Downloads/praat6037_linux64/praat"
stepList = [0.0, 0.33, 0.66, 1.0]
minPitch = 75
maxPitch = 350
########
# # STEP 2: Load pitch data
fromPitchTier = pitch_and_intensity.extractPitchTier(join(inputPath, "You are crazy.wav"),
                                                     join(outputPath, "You are crazy.PitchTier"),
                                                     praatEXE, minPitch,
                                                     maxPitch, forceRegenerate=True)
toPitchTier = pitch_and_intensity.extractPitchTier(join(inputPath, "angrier rocket.wav"),
                                                   join(outputPath, "angrier rocket.PitchTier"),
                                                   praatEXE, minPitch,
                                                   maxPitch, forceRegenerate=True)



##Create TextGrid
# fromUtteranceTier = tgio.IntervalTier('utterance', [], 0, pairedWav= join(inputPath, "The weather is hot.wav"))
# tg = tgio.Textgrid()
# tg.addTier(fromUtteranceTier)
# tg.save(join(outputPath, "The weather is hot.TextGrid"))


# toUtteranceTier = tgio.IntervalTier('utterance', [], 0, pairedWav= join(inputPath, "OAF_yearn_happy.wav"))
# tg = tgio.Textgrid()
# tg.addTier(toUtteranceTier)
# tg.save(join(outputPath, "OAF_yearn_happy.TextGrid"))




########
# # STEP 3: Prepare the pitch regions
# # (*NEW*) -- using getPitchForIntervals()
tierName = "utterance"
fromTGFN = join(outputPath, "You_are_crazy.TextGrid")
toTGFN = join(outputPath, "angrier_rocket.TextGrid")
fromPitchRegions = f0_morph.getPitchForIntervals(fromPitchTier.pointList, fromTGFN, tierName)
toPitchRegions = f0_morph.getPitchForIntervals(toPitchTier.pointList, toTGFN, tierName)

print(fromPitchRegions)
print(toPitchRegions)
# # ########
# # # STEP 4: Run the morph process
f0_morph.f0Morph(fromWavFN=join(inputPath, "You are crazy.wav"),
                 pitchPath=outputPath,
                 stepList=stepList,
                 outputName="CRAZY ROCKET ANGRY",
                 doPlotPitchSteps=False,
                 fromPitchData=fromPitchRegions,
                 toPitchData=toPitchRegions,
                 outputMinPitch=minPitch,
                 outputMaxPitch=maxPitch,
                 praatEXE=praatEXE,
                 keepPitchRange=False,
                 keepAveragePitch=False)
#
pitch_and_intensity.extractPitchTier(join(outputPath, "f0_resynthesized_wavs", "neutral_to_happy_with_morph_0.wav"),
                                     join(outputPath, "neutral_to_happy_with_morph_0.PitchTier"),
                                     praatEXE, minPitch,
                                     maxPitch, forceRegenerate=True)

pitch_and_intensity.extractPitchTier(join(outputPath, "f0_resynthesized_wavs", "neutral_to_happy_with_morph_0.33.wav"),
                                     join(outputPath, "neutral_to_happy_with_morph_0.33.PitchTier"),
                                     praatEXE, minPitch,
                                     maxPitch, forceRegenerate=True)

pitch_and_intensity.extractPitchTier(join(outputPath, "f0_resynthesized_wavs", "neutral_to_happy_with_morph_0.66.wav"),
                                     join(outputPath, "neutral_to_happy_with_morph_0.66.PitchTier"),
                                     praatEXE, minPitch,
                                     maxPitch, forceRegenerate=True)

pitch_and_intensity.extractPitchTier(join(outputPath, "f0_resynthesized_wavs", "neutral_to_happy_with_morph_1.wav"),
                                     join(outputPath, "neutral_to_happy_with_morph.PitchTier"),
                                     praatEXE, minPitch,
                                     maxPitch, forceRegenerate=True)




print("Let's take a look at the output:")
