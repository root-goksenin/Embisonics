#
# Data generator for training the SELDnet
#
# Dasheng integration: when params['mono_encoder'] starts with 'dasheng', a
# second stream of cached Dasheng mel features (W-channel, shape (1,64,T) on
# disk) is loaded in lockstep with the 7-channel spatial feature and yielded
# as the middle element of the batch tuple: (feat, dash, label).
# For GRAM (or any non-dasheng encoder) behaviour is unchanged: (feat, label).
#

import os
import numpy as np
import cls_feature_class
from IPython import embed
from collections import deque
import random


class DataGenerator(object):
    def __init__(
            self, params, split=1, shuffle=True, per_file=False, is_eval=False
    ):
        self._per_file = per_file
        self._is_eval = is_eval
        self._splits = np.array(split)
        self._batch_size = params['batch_size']
        self._feature_seq_len = params['feature_sequence_length']
        self._label_seq_len = params['label_sequence_length']
        self._shuffle = shuffle
        self._feat_cls = cls_feature_class.FeatureClass(params=params, is_eval=self._is_eval)
        self._label_dir = self._feat_cls.get_label_dir()
        self._feat_dir = self._feat_cls.get_normalized_feat_dir()
        self._multi_accdoa = params['multi_accdoa']

        # --- DASHENG ---
        # Second (mono) stream: cached Dasheng mel, loaded in parallel with feat.
        self._use_dasheng = str(params.get('mono_encoder', '')).startswith('dasheng')
        self._dasheng_dir = self._feat_cls.get_dasheng_feat_dir() if self._use_dasheng else None
        self._dasheng_mel_bins = 64
        self._circ_buf_dasheng = None
        # ---------------

        # --- SPEAR ---
        # Second (mono) stream: cached raw 16 kHz W-channel waveform, laid out on
        # the feature-frame grid (spf samples per frame), loaded in parallel with
        # feat and flattened back to a contiguous waveform per sequence.
        self._use_spear = False
        self._spear_dir = self._feat_cls.get_spear_feat_dir() if self._use_spear else None
        self._spear_samples_per_frame = int(round(params['hop_len_s'] * 16000))
        self._circ_buf_spear = None
        # -------------

        self._filenames_list = list()
        self._nb_frames_file = 0     # Using a fixed number of frames in feat files. Updated in _get_label_filenames_sizes()
        self._nb_mel_bins = self._feat_cls.get_nb_mel_bins()
        self._nb_ch = None
        self._label_len = None  # total length of label - DOA + SED
        self._doa_len = None    # DOA label length
        self._nb_classes = self._feat_cls.get_nb_classes()

        self._circ_buf_feat = None
        self._circ_buf_label = None

        self._get_filenames_list_and_feat_label_sizes()

        print(
            '\tDatagen_mode: {}, nb_files: {}, nb_classes:{}\n'
            '\tnb_frames_file: {}, feat_len: {}, nb_ch: {}, label_len:{}\n'.format(
                'eval' if self._is_eval else 'dev', len(self._filenames_list),  self._nb_classes,
                self._nb_frames_file, self._nb_mel_bins, self._nb_ch, self._label_len
                )
        )

        print(
            '\tDataset: {}, split: {}\n'
            '\tbatch_size: {}, feat_seq_len: {}, label_seq_len: {}, shuffle: {}\n'
            '\tTotal batches in dataset: {}\n'
            '\tlabel_dir: {}\n '
            '\tfeat_dir: {}\n'
            '\tuse_dasheng: {}, dasheng_dir: {}\n'.format(
                params['dataset'], split,
                self._batch_size, self._feature_seq_len, self._label_seq_len, self._shuffle,
                self._nb_total_batches,
                self._label_dir, self._feat_dir,
                self._use_dasheng, self._dasheng_dir
            )
        )

    def get_data_sizes(self):
        feat_shape = (self._batch_size, self._nb_ch, self._feature_seq_len, self._nb_mel_bins)
        if self._is_eval:
            label_shape = None
        else:
            if self._multi_accdoa is True:
                label_shape = (self._batch_size, self._label_seq_len, self._nb_classes*3*3)
            else:
                label_shape = (self._batch_size, self._label_seq_len, self._nb_classes*3)
        return feat_shape, label_shape

    def get_total_batches_in_data(self):
        return self._nb_total_batches

    def _get_filenames_list_and_feat_label_sizes(self):
        print('Computing some stats about the dataset')
        max_frames, total_frames, temp_feat = -1, 0, []
        for filename in os.listdir(self._feat_dir):
            if int(filename[4]) in self._splits: # check which split the file belongs to
                self._filenames_list.append(filename)
                    
                temp_feat = np.load(os.path.join(self._feat_dir, filename))
                total_frames += (temp_feat.shape[0] - (temp_feat.shape[0] % self._feature_seq_len))
                if temp_feat.shape[0]>max_frames:
                    max_frames = temp_feat.shape[0]
  
        if len(temp_feat)!=0:
            self._nb_frames_file = max_frames if self._per_file else temp_feat.shape[0]
            self._nb_ch = temp_feat.shape[1] // self._nb_mel_bins
        else:
            print('Loading features failed')
            exit()

        if not self._is_eval:
            temp_label = np.load(os.path.join(self._label_dir, self._filenames_list[0]))
            if self._multi_accdoa is True:
                self._num_track_dummy = temp_label.shape[-3]
                self._num_axis = temp_label.shape[-2]
                self._num_class = temp_label.shape[-1]
            else:
                self._label_len = temp_label.shape[-1]
            self._doa_len = 3 # Cartesian

        if self._per_file:
            self._batch_size = int(np.ceil(max_frames/float(self._feature_seq_len)))
            print('\tWARNING: Resetting batch size to {}. To accommodate the inference of longest file of {} frames in a single batch'.format(self._batch_size, max_frames))
            self._nb_total_batches = len(self._filenames_list)
        else:
            self._nb_total_batches = int(np.floor(total_frames / (self._batch_size*self._feature_seq_len)))

        self._feature_batch_seq_len = self._batch_size*self._feature_seq_len
        self._label_batch_seq_len = self._batch_size*self._label_seq_len
        return

    # --- DASHENG ---
    def _load_dasheng(self, filename, target_len):
        """Load cached Dasheng mel for `filename`, return (T, 64) aligned to
        `target_len` frames (crop or edge-pad). On-disk shape is (1, 64, T)."""
        d = np.load(os.path.join(self._dasheng_dir, filename))
        if d.ndim == 3:                 # (1, 64, T)
            d = d[0]
        if d.shape[0] == self._dasheng_mel_bins:   # (64, T) -> (T, 64)
            d = d.T
        if d.shape[0] >= target_len:
            d = d[:target_len]
        else:
            d = np.pad(d, ((0, target_len - d.shape[0]), (0, 0)), mode='edge')
        return d
    # ---------------

    # --- SPEAR ---
    def _load_spear(self, filename, target_len):
        """Load cached SPEAR waveform-rows for `filename`, return (T, spf) aligned
        to `target_len` frames (crop or edge-pad). On-disk shape is (1, spf, T)."""
        d = np.load(os.path.join(self._spear_dir, filename))
        if d.ndim == 3:                 # (1, spf, T)
            d = d[0]
        if d.shape[0] == self._spear_samples_per_frame:   # (spf, T) -> (T, spf)
            d = d.T
        if d.shape[0] >= target_len:
            d = d[:target_len]
        else:
            d = np.pad(d, ((0, target_len - d.shape[0]), (0, 0)), mode='edge')
        return d
    # -------------
    def generate(self):
        """
        Generates batches of samples
        :return: 
        """
        if self._shuffle:
            random.shuffle(self._filenames_list)

        # Ideally this should have been outside the while loop. But while generating the test data we want the data
        # to be the same exactly for all epoch's hence we keep it here.
        self._circ_buf_feat = deque()
        self._circ_buf_label = deque()
        self._circ_buf_dasheng = deque()   # --- DASHENG ---    
        self._circ_buf_spear = deque()     # --- SPEAR ---
        file_cnt = 0
        if self._is_eval:
            for i in range(self._nb_total_batches):
                # load feat and label to circular buffer. Always maintain atleast one batch worth feat and label in the
                # circular buffer. If not keep refilling it.
                while len(self._circ_buf_feat) < self._feature_batch_seq_len:
                    temp_feat = np.load(os.path.join(self._feat_dir, self._filenames_list[file_cnt]))

                    for row_cnt, row in enumerate(temp_feat):
                        self._circ_buf_feat.append(row)

                    # --- DASHENG --- aligned to (uncropped) temp_feat length
                    if self._use_dasheng:
                        temp_dash = self._load_dasheng(self._filenames_list[file_cnt], temp_feat.shape[0])
                        for d_row in temp_dash:
                            self._circ_buf_dasheng.append(d_row)
                    # ---------------
                    # --- SPEAR --- aligned to the same temp_feat length
                    if self._use_spear:
                        temp_spear = self._load_spear(self._filenames_list[file_cnt], temp_feat.shape[0])
                        for s_row in temp_spear:
                            self._circ_buf_spear.append(s_row)
                    # -------------
                    # If self._per_file is True, this returns the sequences belonging to a single audio recording
                    if self._per_file:
                        extra_frames = self._feature_batch_seq_len - temp_feat.shape[0]
                        extra_feat = np.ones((extra_frames, temp_feat.shape[1])) * 1e-6

                        for row_cnt, row in enumerate(extra_feat):
                            self._circ_buf_feat.append(row)

                        # --- DASHENG --- mirror the feat zero-ish padding
                        if self._use_dasheng:
                            for _ in range(extra_frames):
                                self._circ_buf_dasheng.append(
                                    np.ones(self._dasheng_mel_bins, dtype=np.float32) * 1e-6)
                        # ---------------
                        # --- SPEAR --- same count as feat padding
                        if self._use_spear:
                            for _ in range(extra_frames):       # train branch: feat_extra_frames
                                self._circ_buf_spear.append(
                                    np.zeros(self._spear_samples_per_frame, dtype=np.float32))
                        # -------------
                    file_cnt = file_cnt + 1

                # Read one batch size from the circular buffer
                feat = np.zeros((self._feature_batch_seq_len, self._nb_mel_bins * self._nb_ch))
                for j in range(self._feature_batch_seq_len):
                    feat[j, :] = self._circ_buf_feat.popleft()
                feat = np.reshape(feat, (self._feature_batch_seq_len, self._nb_ch, self._nb_mel_bins))

                # Split to sequences
                feat = self._split_in_seqs(feat, self._feature_seq_len)
                feat = np.transpose(feat, (0, 2, 1, 3))

                # --- mono stream ---
                if self._use_dasheng:
                    dash = np.zeros((self._feature_batch_seq_len, self._dasheng_mel_bins))
                    for j in range(self._feature_batch_seq_len):
                        dash[j, :] = self._circ_buf_dasheng.popleft()
                    dash = self._split_in_seqs(dash, self._feature_seq_len)   # (S, 200, 64)
                    dash = np.transpose(dash, (0, 2, 1))                      # (S, 64, 200)
                    yield feat, dash
                elif self._use_spear:
                    spear = np.zeros((self._feature_batch_seq_len, self._spear_samples_per_frame))
                    for j in range(self._feature_batch_seq_len):
                        spear[j, :] = self._circ_buf_spear.popleft()
                    spear = self._split_in_seqs(spear, self._feature_seq_len) # (S, 200, spf)
                    spear = spear.reshape(spear.shape[0], -1)                 # (S, 200*spf) waveform
                    yield feat, spear
                else:
                    yield feat
                # -------------

        else:
            for i in range(self._nb_total_batches):

                # load feat and label to circular buffer. Always maintain atleast one batch worth feat and label in the
                # circular buffer. If not keep refilling it.
                while len(self._circ_buf_feat) < self._feature_batch_seq_len:
                    temp_feat = np.load(os.path.join(self._feat_dir, self._filenames_list[file_cnt]))
                    temp_label = np.load(os.path.join(self._label_dir, self._filenames_list[file_cnt]))
                    if not self._per_file: 
                        # Inorder to support variable length features, and labels of different resolution. 
                        # We remove all frames in features and labels matrix that are outside 
                        # the multiple of self._label_seq_len and self._feature_seq_len. Further we do this only in training.
                        temp_label = temp_label[:temp_label.shape[0] - (temp_label.shape[0] % self._label_seq_len)]
                        temp_mul = temp_label.shape[0]//self._label_seq_len
                        temp_feat = temp_feat[:temp_mul*self._feature_seq_len, :]

                    for f_row in temp_feat:
                        self._circ_buf_feat.append(f_row)
                    for l_row in temp_label:
                        self._circ_buf_label.append(l_row)

                    # --- DASHENG --- aligned to the (possibly cropped) temp_feat
                    if self._use_dasheng:
                        temp_dash = self._load_dasheng(self._filenames_list[file_cnt], temp_feat.shape[0])
                        for d_row in temp_dash:
                            self._circ_buf_dasheng.append(d_row)
                    #   --- SPEAR --- aligned to the (possibly cropped) temp_feat
                    if self._use_spear:
                        temp_spear = self._load_spear(self._filenames_list[file_cnt], temp_feat.shape[0])
                        for s_row in temp_spear:
                            self._circ_buf_spear.append(s_row)
                    # ---------------

                    # If self._per_file is True, this returns the sequences belonging to a single audio recording
                    if self._per_file:
                        feat_extra_frames = self._feature_batch_seq_len - temp_feat.shape[0]
                        extra_feat = np.ones((feat_extra_frames, temp_feat.shape[1])) * 1e-6

                        label_extra_frames = self._label_batch_seq_len - temp_label.shape[0]
                        if self._multi_accdoa is True:
                            extra_labels = np.zeros((label_extra_frames, self._num_track_dummy, self._num_axis, self._num_class))
                        else:
                            extra_labels = np.zeros((label_extra_frames, temp_label.shape[1]))

                        for f_row in extra_feat:
                            self._circ_buf_feat.append(f_row)
                        for l_row in extra_labels:
                            self._circ_buf_label.append(l_row)

                        # --- DASHENG --- same count as feat padding
                        if self._use_dasheng:
                            for _ in range(feat_extra_frames):
                                self._circ_buf_dasheng.append(
                                    np.ones(self._dasheng_mel_bins, dtype=np.float32) * 1e-6)
                        # --- SPEAR --- same count as feat padding
                        if self._use_spear:
                            for _ in range(feat_extra_frames):
                                self._circ_buf_spear.append(
                                    np.zeros(self._spear_samples_per_frame, dtype=np.float32))
                        # -------------
                        # ---------------

                    file_cnt = file_cnt + 1

                # Read one batch size from the circular buffer
                feat = np.zeros((self._feature_batch_seq_len, self._nb_mel_bins * self._nb_ch))
                for j in range(self._feature_batch_seq_len):
                    feat[j, :] = self._circ_buf_feat.popleft()
                feat = np.reshape(feat, (self._feature_batch_seq_len, self._nb_ch, self._nb_mel_bins))

                if self._multi_accdoa is True:
                    label = np.zeros((self._label_batch_seq_len, self._num_track_dummy, self._num_axis, self._num_class))
                    for j in range(self._label_batch_seq_len):
                        label[j, :, :, :] = self._circ_buf_label.popleft()
                else:
                    label = np.zeros((self._label_batch_seq_len, self._label_len))
                    for j in range(self._label_batch_seq_len):
                        label[j, :] = self._circ_buf_label.popleft()
                # Split to sequences
                feat = self._split_in_seqs(feat, self._feature_seq_len)
                feat = np.transpose(feat, (0, 2, 1, 3))
                
                label = self._split_in_seqs(label, self._label_seq_len)
                if self._multi_accdoa is True:
                    pass
                else:
                    mask = label[:, :, :self._nb_classes]
                    mask = np.tile(mask, 3)
                    label = mask * label[:, :, self._nb_classes:]

                # --- DASHENG ---
                if self._use_dasheng:
                    dash = np.zeros((self._feature_batch_seq_len, self._dasheng_mel_bins))
                    for j in range(self._feature_batch_seq_len):
                        dash[j, :] = self._circ_buf_dasheng.popleft()
                    dash = self._split_in_seqs(dash, self._feature_seq_len)   # (S, 200, 64)
                    dash = np.transpose(dash, (0, 2, 1))                      # (S, 64, 200)
                    yield feat, dash, label
                elif self._use_spear:
                    spear = np.zeros((self._feature_batch_seq_len, self._spear_samples_per_frame))
                    for j in range(self._feature_batch_seq_len):
                        spear[j, :] = self._circ_buf_spear.popleft()
                    spear = self._split_in_seqs(spear, self._feature_seq_len) # (S, 200, spf)
                    spear = spear.reshape(spear.shape[0], -1)                 # (S, 200*spf) waveform
                    yield feat, spear, label
                else:
                    yield feat, label
                # -------------
                # ---------------

    def _split_in_seqs(self, data, _seq_len):
        if len(data.shape) == 1:
            if data.shape[0] % _seq_len:
                data = data[:-(data.shape[0] % _seq_len), :]
            data = data.reshape((data.shape[0] // _seq_len, _seq_len, 1))
        elif len(data.shape) == 2:
            if data.shape[0] % _seq_len:
                data = data[:-(data.shape[0] % _seq_len), :]
            data = data.reshape((data.shape[0] // _seq_len, _seq_len, data.shape[1]))
        elif len(data.shape) == 3:
            if data.shape[0] % _seq_len:
                data = data[:-(data.shape[0] % _seq_len), :, :]
            data = data.reshape((data.shape[0] // _seq_len, _seq_len, data.shape[1], data.shape[2]))
        elif len(data.shape) == 4:  # for multi-ACCDOA with ADPIT
            if data.shape[0] % _seq_len:
                data = data[:-(data.shape[0] % _seq_len), :, :, :]
            data = data.reshape((data.shape[0] // _seq_len, _seq_len, data.shape[1], data.shape[2], data.shape[3]))
        else:
            print('ERROR: Unknown data dimensions: {}'.format(data.shape))
            exit()
        return data

    @staticmethod
    def split_multi_channels(data, num_channels):
        tmp = None
        in_shape = data.shape
        if len(in_shape) == 3:
            hop = in_shape[2] / num_channels
            tmp = np.zeros((in_shape[0], num_channels, in_shape[1], hop))
            for i in range(num_channels):
                tmp[:, i, :, :] = data[:, :, i * hop:(i + 1) * hop]
        elif len(in_shape) == 4 and num_channels == 1:
            tmp = np.zeros((in_shape[0], 1, in_shape[1], in_shape[2], in_shape[3]))
            tmp[:, 0, :, :, :] = data
        else:
            print('ERROR: The input should be a 3D matrix but it seems to have dimensions: {}'.format(in_shape))
            exit()
        return tmp

    def get_nb_classes(self):
        return self._nb_classes

    def nb_frames_1s(self):
        return self._feat_cls.nb_frames_1s()

    def get_hop_len_sec(self):
        return self._feat_cls.get_hop_len_sec()

    def get_filelist(self):
        return self._filenames_list

    def get_frame_per_file(self):
        return self._label_batch_seq_len

    def get_nb_frames(self):
        return self._feat_cls.get_nb_frames()
    
    def get_data_gen_mode(self):
        return self._is_eval

    def write_output_format_file(self, _out_file, _out_dict):
        return self._feat_cls.write_output_format_file(_out_file, _out_dict)
