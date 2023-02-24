import os
import glob
import pandas as pd
from audio2align import Audio2Align
from midi2align import MIDI2Align
from joblib import Parallel, delayed


f0_dir = 'pitch/finetuned_instrument_model_100_005'
midi_dir = 'midi'
audio_dir = '/mnt/d/paganini_audio'
basic_pitch_dir = 'pitch/basic_pitch'
aligned_midi_dir = 'aligned_midi'

def split_name(name):
    player, rest = name.rsplit('_[',1)
    key, rest = rest.split(']_', 1)
    dur = int(rest[:4])
    return player, key, dur

f0s = [os.path.relpath(_, f0_dir) for _ in sorted(glob.glob(os.path.join(f0_dir,'*','*.csv')))]
df = pd.DataFrame(f0s, columns=['f0Path'])
df = df.join(df['f0Path'].str.split('/',expand=True))
df.columns = ['f0Path', 'No', 'rest']
df = df.join(pd.DataFrame(df['rest'].apply(split_name).to_list(), columns=['PlayerID', 'YouTubeKey', 'Duration']))
del df['rest']
print('Total duration:', df.Duration.sum() / 3600, 'hours')

# o: onset, b: beat, d: downbeat
obd_mode = 'obd'  # no downbeats
feature_rate = 150


def align(row):
    f0_path = os.path.join(f0_dir, row.f0Path)
    player, key = row.PlayerID, row.YouTubeKey
    audio_path = os.path.join(audio_dir, row.f0Path[:-6] + 'wav')

    offset = 0  # no audio offset for this dataset
    duration = None  # audio ends at its usual length

    try:
        # construct audio features
        audio_feats = Audio2Align(audio_path, feature_rate=feature_rate,
                                  Fs=22050, offset=offset, duration=duration,
                                  obd_mode=obd_mode)
    except FileNotFoundError:
        print('ERROR IN', audio_path)
        print('SKIPPING')
        pass

    else:
        # construct MIDI features
        midi_feats = MIDI2Align(midi_path, feature_rate=feature_rate,
                                obd_mode=obd_mode)

        # load f0 content (for deterring the audio anchors and debug visualization)
        f0_content = pd.read_csv(f0_path)

        # align
        aligned_midi, aligned_df, wp_mono = midi_feats.align_with_f0_anchors(audio_feats,
                                                                             f0_content,
                                                                             f0_threshold=0.7,
                                                                             debug=False,)
                                                                             #debug_name=audio_path)

        # save the aligned MIDI
        out_dir = os.path.join(aligned_midi_dir, no)
        os.makedirs(out_dir, exist_ok=True)
        aligned_midi_path = os.path.join(aligned_midi_dir, row.f0Path[:-6] + 'mid')  # .f0.csv
        aligned_midi.write(aligned_midi_path)
    return


for no in df['No'].unique():
    midi_path = os.path.join(midi_dir, no + '.mid')
    recordings = df[df['No'] == no]
    Parallel(n_jobs=7)(delayed(align)(df_row[1]) for df_row in recordings.iterrows())
