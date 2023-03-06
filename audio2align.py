import librosa
import librosa.effects
import librosa.display
import numpy as np
from scipy import ndimage
from scipy.interpolate import interp1d
from synctoolbox.feature.chroma import pitch_to_chroma, quantize_chroma
from synctoolbox.feature.pitch import audio_to_pitch_features
from synctoolbox.feature.novelty import spectral_flux
from synctoolbox.feature.dlnco import pitch_onset_features_to_DLNCO
from synctoolbox.feature.pitch_onset import audio_to_pitch_onset_features
from synctoolbox.feature.utils import estimate_tuning
from madmom.features.beats import RNNBeatProcessor
from madmom.features.downbeats import RNNDownBeatProcessor
from madmom.features.onsets import CNNOnsetProcessor


midi_min = 42   # lowest violin note
midi_max = 108  # highest violin

class Audio2Align:
    def __init__(self, audio_path, Fs=22050, offset=0, duration=None,
                 feature_rate=150, obd_mode='ob', Fs4dl=44100,
                 transcription_path=None):
        self.audio, _ = librosa.load(audio_path, sr=Fs,
                                     offset=offset, duration=duration)
        self.feature_rate = feature_rate
        self.audio4dl, _ = librosa.load(audio_path, sr=Fs4dl,
                                        offset=offset, duration=duration)
        self.Fs, self.Fs4dl = Fs, Fs4dl
        self.tuning_offset = estimate_tuning(self.audio, Fs)

        self.f_chroma = self.get_chroma_features_from_audio()

        self.feature_len = self.f_chroma.shape[1]
        self.feature_times = np.arange(0,
                                       self.feature_len / self.feature_rate,
                                       1 / self.feature_rate)

        self.transcription_path = transcription_path

        if obd_mode:
            self.use_transcription_onset = ('t' in obd_mode) and transcription_path
            self.use_onset = 'o' in obd_mode
            self.use_beat = 'b' in obd_mode
            self.use_downbeat = 'd' in obd_mode
            self.f_onset = self.get_merged_act_function()
        else:
            self.f_onset = None

    def get_chroma_features_from_audio(self, verbose=False):
        f_pitch = audio_to_pitch_features(f_audio=self.audio,
                                          Fs=self.Fs,
                                          tuning_offset=self.tuning_offset,
                                          feature_rate=self.feature_rate,
                                          verbose=verbose)
        f_chroma = pitch_to_chroma(f_pitch=f_pitch)
        f_chroma_quantized = quantize_chroma(f_chroma=f_chroma)
        return f_chroma_quantized


    def get_basic_pitch_onsets_from_audio(self):
        FFT_HOP = 256
        AUDIO_SAMPLE_RATE = 22050
        ANNOTATIONS_FPS = AUDIO_SAMPLE_RATE // FFT_HOP
        AUDIO_WINDOW_LENGTH = 2  # duration in seconds of training examples - original 1
        AUDIO_N_SAMPLES = AUDIO_SAMPLE_RATE * AUDIO_WINDOW_LENGTH - FFT_HOP
        ANNOT_N_FRAMES = ANNOTATIONS_FPS * AUDIO_WINDOW_LENGTH
        def basicpitch_frames_to_time(n_frames: int) -> np.ndarray:
            original_times = librosa.core.frames_to_time(
                np.arange(n_frames),
                sr=AUDIO_SAMPLE_RATE,
                hop_length=FFT_HOP,
            )
            window_numbers = np.floor(np.arange(n_frames) / ANNOT_N_FRAMES)
            window_offset = (FFT_HOP / AUDIO_SAMPLE_RATE) * (
                ANNOT_N_FRAMES - (AUDIO_N_SAMPLES / FFT_HOP)
            ) + 0.0018  # this is a magic number, but it's needed for this to align properly
            times = original_times - (window_offset * window_numbers)
            return times
        basic_feats = np.load(self.transcription_path, allow_pickle=True)
        basic_feats = basic_feats['basic_pitch_model_output'].tolist()
        basic_onset = basic_feats['onset'].T
        basic_times = basicpitch_frames_to_time(basic_onset.shape[1])
        onset_roll = np.zeros((12, len(self.feature_times)))
        for piano_key, onset in enumerate(basic_onset):
            midi_number = piano_key + 21
            onset = np.interp(x=self.feature_times, xp=basic_times, fp=onset)
            onset_roll[midi_number%12] += onset
        onset_roll -= onset_roll.min()  # this thing is quite noisy!
        onset_roll /= onset_roll.max()
        return onset_roll


    def get_DLNCO_features_from_audio(self, verbose=False):
        f_pitch_onset = audio_to_pitch_onset_features(f_audio=self.audio,
                                                      Fs=self.Fs,
                                                      tuning_offset=self.tuning_offset,
                                                      verbose=verbose)

        f_DLNCO = pitch_onset_features_to_DLNCO(f_peaks=f_pitch_onset,
                                                feature_rate=self.feature_rate,
                                                feature_sequence_length=self.feature_len,
                                                visualize=verbose)

        return f_DLNCO

    def get_spectral_flux_from_audio(self, gamma=10):  # log compression param):
        f_novelty = spectral_flux(self.audio,
                                  Fs=self.Fs,
                                  feature_rate=self.feature_rate,
                                  gamma=gamma)
        if f_novelty.size < self.feature_len:
            # The feature sequence length of the chroma features are not same as the novelty curve
            # due to the padding while the computation of STFT for chroma features and
            # the differentiation in spectral flux
            diff = self.feature_len - f_novelty.size
            pad = int(diff / 2)
            f_novelty = np.concatenate((np.zeros(pad), f_novelty, np.zeros(pad)))

            if diff % 2 == 1:
                f_novelty = np.concatenate((f_novelty, np.zeros(1)))

        return f_novelty.reshape(1, -1)

    def CNN_act_function(self, orig_feature_rate=100):

        cnn_onset_processor = CNNOnsetProcessor(fps=orig_feature_rate, sr=self.Fs4dl)
        act_function_CNN = cnn_onset_processor(self.audio4dl)
        act_function_CNN, Fs_out = resample_signal(act_function_CNN,
                                                   Fs_in=orig_feature_rate,
                                                   Fs_out=self.feature_rate)

        return act_function_CNN

    def get_CNN_act_function_from_audio(self):

        f_novelty = self.CNN_act_function()

        if f_novelty.size < self.feature_len:
            # The feature sequence length of the chroma features are not same as the novelty curve
            # due to the padding while the computation of STFT for chroma features and
            # the differentiation in spectral flux
            print('CNN_act_fun size before: ' + str((f_novelty.size)))
            diff = self.feature_len - f_novelty.size
            if diff % 2 == 0:
                pad = int(diff / 2)
                f_novelty = np.concatenate((np.zeros(pad), f_novelty, np.zeros(pad)))
            else:
                pad = int(diff / 2)
                f_novelty = np.concatenate((np.zeros(pad), f_novelty, np.zeros(pad)))
                null_to_append = np.array([0])
                f_novelty = np.append(f_novelty, null_to_append)

        print('CNN_act_fun size: ' + str(np.size(f_novelty)))
        return f_novelty.reshape(1, -1)

    def RNN_act_function(self, orig_feature_rate=100):

        act_function_RNN = RNNBeatProcessor(fps=orig_feature_rate)(self.audio4dl)
        act_function_RNN, Fs_out = resample_signal(act_function_RNN,
                                                   Fs_in=orig_feature_rate,
                                                   Fs_out=self.feature_rate)

        return act_function_RNN

    def get_RNN_act_function_from_audio(self):

        f_novelty = self.RNN_act_function()

        if f_novelty.size < self.feature_len:
            # The feature sequence length of the chroma features are not same as the novelty curve
            # due to the padding while the computation of STFT for chroma features and
            # the differentiation in spectral flux
            print('RNN_act_fun size before: ' + str((f_novelty.size)))
            diff = self.feature_len - f_novelty.size
            if diff % 2 == 0:
                pad = int(diff / 2)
                f_novelty = np.concatenate((np.zeros(pad), f_novelty, np.zeros(pad)))
            else:
                pad = int(diff / 2)
                f_novelty = np.concatenate((np.zeros(pad), f_novelty, np.zeros(pad)))
                null_to_append = np.array([0])
                f_novelty = np.append(f_novelty, null_to_append)

        print('RNN_act_fun size: ' + str(np.size(f_novelty)))
        return f_novelty.reshape(1, -1)

    def RNN_downbeat_act_function(self, orig_feature_rate=100):

        act_function_RNN_downbeat = RNNDownBeatProcessor(fps=orig_feature_rate)(self.audio4dl)
        act_function_RNN_downbeat, Fs_out = resample_signal(act_function_RNN_downbeat[:, 1],
                                                            Fs_in=orig_feature_rate,
                                                            Fs_out=self.feature_rate)
        return act_function_RNN_downbeat

    def get_RNN_downbeat_act_function_from_audio(self):

        f_novelty = self.RNN_downbeat_act_function()

        if f_novelty.size < self.feature_len:
            # The feature sequence length of the chroma features are not same as the novelty curve
            # due to the padding while the computation of STFT for chroma features and
            # the differentiation in spectral flux
            print('RNN_act_fun size before: ' + str((f_novelty.size)))
            diff = self.feature_len - f_novelty.size
            if diff % 2 == 0:
                pad = int(diff / 2)
                f_novelty = np.concatenate((np.zeros(pad), f_novelty, np.zeros(pad)))
            else:
                pad = int(diff / 2)
                f_novelty = np.concatenate((np.zeros(pad), f_novelty, np.zeros(pad)))
                null_to_append = np.array([0])
                f_novelty = np.append(f_novelty, null_to_append)

        print('RNN_act_fun size: ' + str(np.size(f_novelty)))
        return f_novelty.reshape(1, -1)

    def get_merged_act_function(self):
        afs = []
        if self.use_onset:
            af = self.get_CNN_act_function_from_audio()
            afs.append(np.reshape(af, af.shape[1]))
        if self.use_beat:
            af = self.get_RNN_act_function_from_audio()
            afs.append(np.reshape(af, af.shape[1]))
        if self.use_downbeat:
            af = self.get_RNN_downbeat_act_function_from_audio()
            afs.append(np.reshape(af, af.shape[1]))
        f_novelty = np.array(afs)

        if self.use_transcription_onset:
            af = self.get_basic_pitch_onsets_from_audio()
            f_novelty = np.vstack((f_novelty, af))

        print('Merge_act_fun size: ' + str(f_novelty.shape[1]))
        return f_novelty


def resample_signal(x_in, Fs_in, Fs_out=100, norm=True, time_max_sec=None, sigma=None):
    if sigma is not None:
        x_in = ndimage.gaussian_filter(x_in, sigma=sigma)
    T_coef_in = np.arange(x_in.shape[0]) / Fs_in
    time_in_max_sec = T_coef_in[-1]
    if time_max_sec is None:
        time_max_sec = time_in_max_sec
    N_out = int(np.ceil(time_max_sec * Fs_out))
    T_coef_out = np.arange(N_out) / Fs_out
    if T_coef_out[-1] > time_in_max_sec:
        x_in = np.append(x_in, [0])
        T_coef_in = np.append(T_coef_in, [T_coef_out[-1]])
    x_out = interp1d(T_coef_in, x_in, kind='linear')(T_coef_out)
    if norm:
        x_max = max(x_out)
        if x_max > 0:
            x_out = x_out / max(x_out)
    return x_out, Fs_out