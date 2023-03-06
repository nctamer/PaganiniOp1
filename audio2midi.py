import os
import glob

import numpy as np
import pandas as pd
from audio2align import Audio2Align
from midi2align import MIDI2Align
from joblib import Parallel, delayed


f0_dir = '/mnt/d/violin_etudes_aligned/pitch/finetuned_instrument_model_100_005'
midi_dir = '/mnt/d/violin_etudes_aligned/reference'
audio_dir = '/mnt/d/violin_etudes_aligned/audio_trim'
basic_pitch_dir = '/mnt/d/violin_etudes_aligned/pitch/basic_pitch'
aligned_midi_dir = '/mnt/d/violin_etudes_aligned/midi'

def split_name(name):
    composer, n, player, key_start_end = name.split('_', 3)
    key, start, end = key_start_end.rsplit('-', 2)

    start = int(start[:4])
    end = int(end[:4])

    return player, key, start, end

'''
base_path = 'violin_etudes_aligned/'
all_files_with_curly = glob.glob(os.path.join(base_path,'*', '*', '*', '*_v={*'))
all_files_with_curly += glob.glob(os.path.join(base_path,'*', '*', '*', '*', '*_v={*'))
for stupid_file in all_files_with_curly:
    path, idiot = stupid_file.rsplit('/',1)
    logical, idiot = idiot.split('v={',1)
    key, start_end = idiot.rsplit('}_s=', 1)
    start, end = start_end.split('_e=',1)
    less_idiot = '-'.join([key, start, end])
    less_idiot = logical + less_idiot
    less_idiot = os.path.join(path, less_idiot)
    os.rename(stupid_file, less_idiot)

'''



f0s = [os.path.relpath(_, f0_dir) for _ in sorted(glob.glob(os.path.join(f0_dir,'*','*','*.csv')))]
df = pd.DataFrame(f0s, columns=['f0Path'])
df = df.join(df['f0Path'].str.split('/',expand=True))
df.columns = ['f0Path', 'Composer', 'No', 'rest']
df = df.join(pd.DataFrame(df['rest'].apply(split_name).to_list(), columns=['PlayerID', 'YouTubeKey', 'Start', 'End']))
df['Duration'] = df['End'] - df['Start']
del df['rest']
print('Total duration:', df.Duration.sum() / 3600, 'hours')

# o: onset, b: beat, d: downbeat
obd_mode =  'obdt'  # no downbeats
feature_rate = 44100/256


def align(row):
    f0_path = os.path.join(f0_dir, row.f0Path)
    player, key = row.PlayerID, row.YouTubeKey
    audio_path = os.path.join(audio_dir, row.f0Path[:-6] + 'mp3')

    offset = 0  # no audio offset for this dataset
    duration = None  # audio ends at its usual length

    if 't' in obd_mode:
        transcription_path = os.path.join(basic_pitch_dir, row.f0Path[:-7] + '_basic_pitch.npz')
    else:
        transcription_path = None

    try:
        # construct audio features
        audio_feats = Audio2Align(audio_path, feature_rate=feature_rate,
                                  Fs=22050, offset=offset, duration=duration, transcription_path=transcription_path,
                                  obd_mode=obd_mode)
    except FileNotFoundError:
        print('ERROR IN', audio_path)
        print('SKIPPING')
        pass

    else:

        # load f0 content (for deterring the audio anchors and debug visualization)
        f0_content = pd.read_csv(f0_path)

        # align
        aligned_midi, aligned_df, wp_mono = midi_feats.align_with_f0_anchors(audio_feats,
                                                                             f0_content,
                                                                             f0_threshold=0.7,
                                                                             debug=True,
                                                                             debug_name=audio_path)



        # save the aligned MIDI
        out_dir = os.path.join(aligned_midi_dir, row.Composer, no)
        os.makedirs(out_dir, exist_ok=True)
        aligned_midi_path = os.path.join(aligned_midi_dir, row.f0Path[:-7] + '.mid')  # .f0.csv
        aligned_midi.write(aligned_midi_path)
    return


for no in sorted(df['No'].unique()):

    midi_names =  '*'+no+'.mid*'
    midi_paths = glob.glob(os.path.join(midi_dir, '*', midi_names))
    assert len(midi_paths)==1
    midi_path = midi_paths[0]
    # construct MIDI features
    midi_feats = MIDI2Align(midi_paths[0], feature_rate=feature_rate,
                            obd_mode=obd_mode)

    recordings = df[df['No'] == no]

    #for i, df_row in recordings.iterrows():
    #    align(df_row)
    Parallel(n_jobs=5)(delayed(align)(df_row[1]) for df_row in recordings.iterrows())


