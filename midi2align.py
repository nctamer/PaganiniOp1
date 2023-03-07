import pretty_midi
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import medfilt
import copy
from libfmp.b import list_to_pitch_activations
import libfmp.c2
from synctoolbox.dtw.mrmsdtw import sync_via_mrmsdtw_with_anchors
from synctoolbox.dtw.utils import make_path_strictly_monotonic
from synctoolbox.feature.csv_tools import df_to_pitch_features
from synctoolbox.feature.chroma import pitch_to_chroma, quantize_chroma
from scoreinformed import convert_ann_to_constraint_region, hz_to_cents, cents_to_hz, visualize_salience_traj_constraints
from scipy.stats import norm


class MIDI2Align:
    def __init__(self, midi_file, feature_rate=150, pad_gimmick=0.1, obd_mode='ob'):
        self.midi = pretty_midi.PrettyMIDI(midi_file)
        self.pad_gimmick = pad_gimmick
        self.feature_rate = feature_rate
        self.df, self.df_order = self.midi2df()
        self.df['start'] += pad_gimmick
        self.df['end'] = self.df['start'] + self.df['duration']

        self.f_chroma = self.get_features_from_df()
        self.f_chroma = np.pad(self.f_chroma,
                               ((0, 0),
                                (0, int(np.ceil(pad_gimmick * feature_rate)))),
                               'constant')  # GIMMICK! zero pad the end

        self.feature_len = self.f_chroma.shape[1]
        self.feature_times = np.arange(0,
                                       self.feature_len / self.feature_rate,
                                       1 / self.feature_rate)
        self.feature_times = self.feature_times[:self.feature_len]
        if obd_mode:
            self.use_transcription_onset = 't' in obd_mode
            self.use_onset = 'o' in obd_mode
            self.use_beat = 'b' in obd_mode
            self.use_downbeat = 'd' in obd_mode
            self.f_onset = self.midi2sparse()
        else:
            self.f_onset = None

    def midi2df(self, midi=None):
        if not midi:
            midi = self.midi
        midi_list = []

        for instrument in midi.instruments:
            for note in instrument.notes:
                start = note.start
                end = note.end
                duration = end - start
                pitch = note.pitch
                velocity = note.velocity / 128
                midi_list.append([start, duration, pitch, velocity, 'violin'])

        sort_index, midi_list = zip(
            *[(i, x) for i, x in sorted(enumerate(midi_list), key=lambda x: (x[1][0], x[1][2]))])

        df = pd.DataFrame(midi_list, columns=['start', 'duration', 'pitch', 'velocity', 'instrument'])
        return df, list(sort_index)

    def get_features_from_df(self):
        f_pitch = df_to_pitch_features(self.df, feature_rate=self.feature_rate)
        f_chroma = pitch_to_chroma(f_pitch=f_pitch)
        f_chroma_quantized = quantize_chroma(f_chroma=f_chroma)
        return f_chroma_quantized

    def times_to_sparse_features(self, times):
        times += self.pad_gimmick
        ind = np.unique((self.feature_rate * times).astype(int))
        feats = np.zeros(self.feature_len)
        try:
            feats[ind] = 1
        except IndexError:  # the last downbeat can create error!
            feats[ind[:-1]] = 1
        return np.convolve(feats, [0.3, 1, 0.3], mode='same')

    def df_start_to_onset(self):

        onset_roll = np.zeros((12, len(self.feature_times)))
        for i, row in self.df.iterrows():
            onset = row.start*self.feature_rate
            start = max(0, int(onset)-3)
            end = min(self.feature_len-1, int(onset)+3)

            vals = norm.pdf(np.linspace(start-onset, end-onset, end-start+1)/0.7)
            # if you increase 0.7 you smooth the peak
            # if you decrease it, e.g., 0.1, it becomes too peaky! around 0.5-0.7 seems ok
            vals = vals/vals.max()
            pitch = row.pitch % 12
            onset_roll[pitch,start:end+1] += vals

        onset_roll -= onset_roll.min()  # normalize using the same trick
        onset_roll /= onset_roll.max()
        return onset_roll

    def midi2sparse(self):
        feats = []
        assert any([self.use_onset, self.use_beat, self.use_downbeat])
        if self.use_onset:
            feats.append(self.times_to_sparse_features(self.midi.get_onsets()))
        if self.use_beat:
            feats.append(self.times_to_sparse_features(self.midi.get_beats()))
        if self.use_downbeat:
            feats.append(self.times_to_sparse_features(self.midi.get_downbeats()))
        feats = np.vstack(feats)
        if self.use_transcription_onset:
            feats = np.vstack((feats, self.df_start_to_onset()))

        return feats

    def aligned2midi(self, aligned_df):
        new_midi = copy.deepcopy(self.midi)
        i = 0
        for instrument in new_midi.instruments:
            for note in instrument.notes:
                df_index = self.df_order[i]
                note.start = aligned_df.loc[df_index, 'start']
                note.end = aligned_df.loc[df_index, 'end']
                i += 1
        return new_midi

    def align(self, audio_feats, audio_start_anchor, audio_end_anchor,
              step_weights=np.array([1.5, 1.5, 2.0]),
              threshold_rec=10 ** 6):
        assert self.feature_rate == audio_feats.feature_rate
        assert self.f_chroma.shape[0] == audio_feats.f_chroma.shape[0]
        if hasattr(self, 'f_onset'):
            assert self.f_onset.shape[0] == audio_feats.f_onset.shape[0]
        wp = sync_via_mrmsdtw_with_anchors(f_chroma1=self.f_chroma,
                                           f_onset1=self.f_onset,
                                           f_chroma2=audio_feats.f_chroma,
                                           f_onset2=audio_feats.f_onset,
                                           input_feature_rate=self.feature_rate,
                                           step_weights=step_weights,
                                           threshold_rec=threshold_rec,
                                           verbose=False,
                                           anchor_pairs=[(self.pad_gimmick,
                                                          audio_start_anchor),
                                                         (self.df['end'].max(),
                                                          audio_end_anchor)])


        wp_mono = make_path_strictly_monotonic(wp)

        df_aligned = copy.deepcopy(self.df)
        df_aligned.start = np.interp(self.df.start,
                                     wp_mono[0] / self.feature_rate,
                                     wp_mono[1] / self.feature_rate)
        df_aligned.end = np.interp(self.df.end,
                                   wp_mono[0] / self.feature_rate,
                                   wp_mono[1] / self.feature_rate)
        df_aligned.duration = df_aligned.end - df_aligned.start

        return self.aligned2midi(df_aligned), df_aligned, wp_mono

    def align_with_f0_anchors(self, audio_feats, f0_df, f0_threshold=0.7,
                              step_weights=np.array([1.5, 1.5, 2.0]),
                              threshold_rec=10 ** 6, debug=True, debug_name=None):
        audio_start, audio_end = self.estimate_audio_anchors_from_f0(f0_df,
                                                                     f0_threshold=0.5)
        midi_aligned, df_aligned, wp_mono = self.align(audio_feats, audio_start, audio_end)
        if debug:
            self.visualize_midi_transcription(f0_df, f0_threshold=f0_threshold,
                                              df_aligned=df_aligned,
                                              title=debug_name)

        return midi_aligned, df_aligned, wp_mono

    def visualize_midi_transcription(self, f0_df, f0_threshold=0.7,
                                     df_aligned=None, title=None):
        if not hasattr(df_aligned, '__len__'): # if it is none
            assert self.pad_gimmick == 0  # no need to align
            df_aligned = self.df  # then use them as they are
        # Visualization
        figsize = (25, 5)
        cmap = libfmp.b.compressed_gray_cmap(alpha=5)
        df_aligned['end'] = df_aligned['start'] + df_aligned['duration']
        ann_score = df_aligned[['start', 'end', 'pitch']].values
        constraint_region = convert_ann_to_constraint_region(ann_score, tol_freq_cents=100)

        f0_df.frequency = f0_df.frequency[f0_df.confidence > f0_threshold]

        T_coef = f0_df.time.values
        F_min, F_max = constraint_region[:, 2:].min(), constraint_region[:, 2:].max()

        F_coef_cents = np.arange(hz_to_cents(F_min) - 10, hz_to_cents(F_max) + 10, 10)
        Z = np.zeros((len(F_coef_cents), len(T_coef)))  # gimmick! we can use the stft instead. this is empty!!

        traj = f0_df[['time', 'frequency']].values

        # Visualization
        visualize_salience_traj_constraints(Z, T_coef, F_coef_cents,
                                            F_ref=cents_to_hz(F_coef_cents[0]),
                                            figsize=figsize, cmap=cmap,
                                            constraint_region=constraint_region,
                                            colorbar=False, traj=traj, ax=None)
        if title:
            plt.title(title)
        plt.show()
        return

    def estimate_audio_anchors_from_f0(self, f0_content, f0_threshold):
        smoothened_confidences = medfilt(f0_content.confidence, kernel_size=15)
        anchor_candidates = f0_content[np.logical_and(f0_content.confidence > f0_threshold,
                                                      smoothened_confidences > 0.5)]

        # now start searching for an end anchor candidate in the last 100 rows
        end_anchor_candidates = anchor_candidates.iloc[-100:]
        end = end_anchor_candidates.time.iloc[0]

        # search pitch track for the last note events
        end_freqs = self.df[self.df.end == self.df.end.max()].pitch.apply(pretty_midi.note_number_to_hz).values
        end_cents = np.tile(hz_to_cents(end_freqs), (2, 1))
        end_cents[0, :] = end_cents[0, :] - 50  # lower bound
        end_cents[1, :] = end_cents[1, :] + 50  # upper bound
        end_freqs = cents_to_hz(end_cents)

        for i in range(end_freqs.shape[1]):
            end_min = end_freqs[0, i]
            end_max = end_freqs[1, i]
            is_cand = end_anchor_candidates.frequency.between(end_min, end_max)
            if is_cand.any():
                cand = end_anchor_candidates[is_cand].iloc[-1].time
                if cand > end:
                    end = cand

        # now the same for the first 100 rows
        start_anchor_candidates = anchor_candidates.iloc[:100]
        start = start_anchor_candidates.time.iloc[-1]

        # search within the first midi event
        start_freqs = self.df[self.df.start == self.df.start.min()].pitch.apply(pretty_midi.note_number_to_hz).values
        start_cents = np.tile(hz_to_cents(start_freqs), (2, 1))
        start_cents[0, :] = start_cents[0, :] - 50
        start_cents[1, :] = start_cents[1, :] + 50
        start_freqs = cents_to_hz(start_cents)

        for i in range(start_freqs.shape[1]):
            start_min = start_freqs[0, i]
            start_max = start_freqs[1, i]
            is_cand = start_anchor_candidates.frequency.between(start_min, start_max)
            if is_cand.any():
                cand = start_anchor_candidates[is_cand].iloc[0].time
                if cand < start:
                    start = cand
        if start == f0_content.iloc[0,0]:  # TODO: gimmick! anchor points have to be positive
            start = f0_content.iloc[1,0]
        if end == f0_content.iloc[-1,0]:
            end = f0_content.iloc[-2,0]
        return start, end
