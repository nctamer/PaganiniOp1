import os
import glob

import numpy as np
import pandas as pd
from audio2align import Audio2Align
from midi2align import MIDI2Align
from joblib import Parallel, delayed


obd_mode =  None #'ob'  # no downbeats
feature_rate = 44100/256


f0_dir = '/mnt/d/violin_etudes_aligned/pitch/finetuned_instrument_model_100_005'
midi_dir = '/mnt/d/violin_etudes_aligned/reference'
audio_dir = '/mnt/d/violin_etudes_aligned/audio_trim'
basic_pitch_dir = '/mnt/d/violin_etudes_aligned/pitch/basic_pitch'
aligned_midi_dir = '/mnt/d/violin_etudes_aligned/midi'



for no in ['Op01-13-b', 'Op01-19-b', 'Op01-19-c', 'Op01-20-b', 'Op01-23-b', 'Op01-23-c']:
    midi_names =  '*'+no+'.mid*'
    midi_path = glob.glob(os.path.join(midi_dir, '*', midi_names))[0]
    audio_names = '*'+no+'*.mp3'
    audio_paths = glob.glob(os.path.join(audio_dir, '*', '*', audio_names))
    midi_feats = MIDI2Align(midi_path, feature_rate=feature_rate,
                            obd_mode=obd_mode)
    for audio_path in audio_paths:
        audio_feats = Audio2Align(audio_path, feature_rate=feature_rate, obd_mode=obd_mode)
        f0_path = glob.glob(os.path.join(f0_dir, '*', '*', '*'+os.path.basename(audio_path).split('_')[-1][:-4]+'*'))[0]
        f0_content = pd.read_csv(f0_path)
        # align
        aligned_midi, aligned_df, wp_mono = midi_feats.align_with_f0_anchors(audio_feats,
                                                                             f0_content,
                                                                             f0_threshold=0.7,
                                                                             debug=True,
                                                                             debug_name=os.path.basename(audio_path)[:-4])

        #basic_out_path = glob.glob(os.path.join(basic_pitch_dir, '*', '*', '*'+os.path.basename(audio_path).split('_')[-1][:-4]+'*.mid*'))[0]

