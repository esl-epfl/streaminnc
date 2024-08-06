###
# This one is for moving to 18 channels for all patients
# Also updates the scheme to be compatible with Memory mamangeer

import os
import copy
import re
import pickle

import mne  # import hdf5storage as h5
from torch.utils.data import Dataset
import numpy as np
from scipy import signal


class CHB_Store_Test:
    """
    This script is designed for loading the CHBMIT dataset for testing
    """

    def __init__(
        self,
        data_folder,
        patient,
        window_size_sec,
        ratio,
        train_ratio,
        chunks=5,
        last_seizure=False,
        only_last_chunk=False,
        first_last_seizure=False,
        repeating_ns=False,
        saving_seizure_all=False,
        dataset_dir="",
    ):
        # Not removing hte some flags to keep the interface
        assert not repeating_ns, "In this Obj, you can not use this"
        assert not saving_seizure_all, "In this Obj, you can not use this"
        assert last_seizure, "This is only base loader"
        assert window_size_sec == 1, "This is only test loader"
        # Note that the last seizure option is remained for repeating
        # the experiments. first_last_seizure option can be used in both
        # Scenarios
        # if both of the options are on(last + first_last) it will go to
        # the default mode
        assert os.path.exists(data_folder), "Data folder does not exist"
        self.stride_n = window_size_sec
        self.ratio = ratio
        # for now all of strides are the same.
        # oversampling is handled in the dataset
        self.stride_s = window_size_sec
        self.fs = 256

        # datasets that could be used outside of the class
        # self.test_datasets = []
        self.test_datasets = []

        self.data_folder = os.path.abspath(data_folder)
        self.win_sec = window_size_sec
        self.data = None
        self.labels = None

        self.channels_base = [
            "FP1-F7",
            "F7-T7",
            "T7-P7",
            "P7-O1",
            "FP1-F3",
            "F3-C3",
            "C3-P3",
            "P3-O1",
            "FP2-F4",
            "F4-C4",
            "C4-P4",
            "P4-O2",
            "FP2-F8",
            "F8-T8",
            "T8-P8",
            "P8-O2",
            "FZ-CZ",
            "CZ-PZ",
        ]

        patient_directory = os.path.join(self.data_folder, patient)
        summary_file = open(
            os.path.join(patient_directory, patient + "-summary.txt"), "r"
        )
        lines = summary_file.readlines()

        data_file_list = []
        matches_list = []
        data_tmax_list = []
        iter_lines = iter(lines)
        self.all_Seg = []
        for line in iter_lines:
            if "Channels" in line:
                print(patient, ": detected channels")
                next(iter_lines)  # skipping *** line
                channels = []
                in_line = next(iter_lines)
                while "Channel" in in_line:
                    channels.append(in_line.split(" ")[-1].strip())
                    in_line = next(iter_lines)

                match_list = self.match_ind(self.channels_base, channels)
                assert match_list, "Match list is empty"
            if "File Name" in line:
                filename = line[11:-1]
                ### by defaul raw data are not loaded
                data_file = mne.io.read_raw_edf(
                    os.path.join(patient_directory, filename), verbose="error"
                )
                data_file_list.append(data_file)
                data_tmax_list.append(data_file.tmax)
                ## per each data_file we should have a match_list
                matches_list.append(match_list)

                while "Number of Seizures" not in line:
                    # Skiping File Start and End Time
                    line = next(iter_lines)

                seizure_num = int(re.split(r"[/\\ :,%_]", line)[-1].strip())

                seizure_list = []
                for i in range(seizure_num):
                    start = int(re.split(r"[/\\ :,%_]", next(iter_lines))[-2])
                    end = int(re.split(r"[/\\ :,%_]", next(iter_lines))[-2])
                    seizure_list.append((start, end))

                segments = self.get_all_segments(
                    len(data_file_list) - 1, data_file.tmax, seizure_list
                )
                self.all_Seg.extend(segments)

        self.all_Seg = np.array(self.all_Seg)
        # This part is for compatibility with the previous version
        self.all_S = self.all_Seg[self.all_Seg[:, -1] == 1]
        self.all_NS = self.all_Seg[self.all_Seg[:, -1] == 0]

        ## Splitting to chunks
        if chunks > 1 or chunks == -1:
            if first_last_seizure:
                ## Finding which files have seizure
                sei_ind = np.where(self.all_Seg[:, -1] == 1)[0]
                sei_files = np.unique(self.all_Seg[sei_ind, 0])

                ## Moving the the last and first seizure section
                data_file_ind = list(range(len(data_file_list)))

                first_sei = int(sei_files[0])
                last_sei = int(sei_files[-1])

                data_file_ind.insert(0, data_file_ind.pop(first_sei))
                data_file_ind.append(data_file_ind.pop(last_sei))

            else:
                data_file_ind = list(range(len(data_file_list)))

            if chunks == -1:
                files_ind = [[i] for i in data_file_ind]
                seg_NS_list, files_ind = self.split_func_v2(
                    self.all_NS, files_ind
                )
                seg_S_list, files_ind = self.split_func_v2_seizure(
                    self.all_S, files_ind
                )

            else:
                # for NS we have data from all of the files
                ind_part = np.array(
                    self.partition_list(data_tmax_list, chunks)
                )
                files_ind = np.split(data_file_ind, ind_part[:-1])

                seg_NS_list, _ = self.split_func(self.all_NS, files_ind)
                seg_S_list, _ = self.split_func(
                    self.all_S, files_ind, saving_seizure_all
                )

            if only_last_chunk:
                seg_NS_list = [seg_NS_list[-1]]
                seg_S_list = [seg_S_list[-1]]
                files_ind = [files_ind[-1]]
        else:
            seg_S_list = [self.all_S]
            seg_NS_list = [self.all_NS]
            files_ind = [list(range(len(data_file_list)))]

        for i, [seg_S, seg_NS, file_ind] in enumerate(
            zip(seg_S_list, seg_NS_list, files_ind)
        ):
            ## CHBMIT module only works with sorted ind
            file_ind.sort()
            matches_list_ch = [matches_list[i] for i in file_ind]
            data_file_list_ch = [data_file_list[i] for i in file_ind]

            self.test_datasets.append(
                self.CHBMITest.from_rec(
                    segments=np.append(seg_S, seg_NS, 0),
                    matches_list=matches_list_ch,
                    data_file_list=data_file_list_ch,
                    file_ind=file_ind,
                )
            )

            print(f"Saving Files for {patient} t = {i}")
            save_dir = os.path.join(dataset_dir, patient, str(i))

            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            with open(os.path.join(save_dir, "test.pkl"), "wb") as file:
                pickle.dump(self.test_datasets[-1], file)

    def get_all_segments(self, file_id, data_file_tmax, seizure_list):
        segments = []
        n_segs = int(np.floor(np.floor(data_file_tmax)))
        seg_start = np.arange(0, n_segs)
        seg_stop = seg_start + 1
        segments.extend(
            np.column_stack(
                (
                    np.ones(n_segs) * file_id,
                    seg_start,
                    seg_stop,
                    np.zeros(n_segs),
                )
            )
        )

        ## handling seizuers
        for ev in seizure_list:
            for i in range(ev[0], ev[1]):
                segments[i][-1] = 1

        return segments

    def get_segments(self, file_id, data_file_tmax, seizure_list):
        segments_S = []
        segments_NS = []
        if len(seizure_list) == 0:
            n_segs = int(
                np.floor(
                    (np.floor(data_file_tmax) - self.win_sec) / self.stride_n
                )
            )
            seg_start = np.arange(0, n_segs) * self.stride_n
            seg_stop = seg_start + self.win_sec
            segments_NS.extend(
                np.column_stack(
                    (
                        np.ones(n_segs) * file_id,
                        seg_start,
                        seg_stop,
                        np.zeros(n_segs),
                    )
                )
            )

        else:
            for i, ev in enumerate(seizure_list):
                ## handling seizuers
                n_segs = int(
                    np.floor((ev[1] - ev[0] - self.win_sec) / self.stride_s)
                    + 1
                )
                seg_start = np.arange(0, n_segs) * self.stride_s + ev[0]
                seg_stop = seg_start + self.win_sec
                segments_S.extend(
                    np.column_stack(
                        (
                            np.ones(n_segs) * file_id,
                            seg_start,
                            seg_stop,
                            np.ones(n_segs),
                        )
                    )
                )

                ## before the first seizure
                if i == 0:
                    n_segs = int(
                        np.floor((ev[0] - self.win_sec) / self.stride_n) + 1
                    )
                    seg_start = np.arange(0, n_segs) * self.stride_n
                    seg_stop = seg_start + self.win_sec
                    segments_NS.extend(
                        np.column_stack(
                            (
                                np.ones(n_segs) * file_id,
                                seg_start,
                                seg_stop,
                                np.zeros(n_segs),
                            )
                        )
                    )
                ## between seizure
                else:
                    n_segs = int(
                        np.floor(
                            (ev[0] - seizure_list[i - 1][1] - self.win_sec)
                            / self.stride_n
                        )
                        + 1
                    )
                    seg_start = (
                        np.arange(0, n_segs) * self.stride_n
                        + seizure_list[i - 1][1]
                    )
                    seg_stop = seg_start + self.win_sec
                    segments_NS.extend(
                        np.column_stack(
                            (
                                np.ones(n_segs) * file_id,
                                seg_start,
                                seg_stop,
                                np.zeros(n_segs),
                            )
                        )
                    )
                if i == len(seizure_list) - 1:
                    n_segs = int(
                        np.floor(
                            (np.floor(data_file_tmax) - ev[1] - self.win_sec)
                            / self.stride_n
                        )
                        + 1
                    )
                    seg_start = np.arange(0, n_segs) * self.stride_n + ev[1]
                    seg_stop = seg_start + self.win_sec
                    if n_segs > 1:
                        segments_NS.extend(
                            np.column_stack(
                                (
                                    np.ones(n_segs) * file_id,
                                    seg_start,
                                    seg_stop,
                                    np.zeros(n_segs),
                                )
                            )
                        )
        return segments_S, segments_NS

    @staticmethod
    def match_ind(channel_base, channels_in):
        match_list = []
        for ch in channel_base:
            if ch in channels_in:
                match_list.append(channels_in.index(ch))
            else:
                match_list.append(-1)
        if not match_list or match_list.count(match_list[0]) == len(
            match_list
        ):
            match_list = []
            print("Different Data")
            ## This section is specialized for the Patient 12
            if "-" in channels_in[0]:
                channels_in = [ch.split("-")[0] for ch in channels_in]
            for ch in channel_base:
                ch_sub = ch.split("-")
                if ch_sub[0] in channels_in and ch_sub[1] in channels_in:
                    match_list.append(
                        [
                            channels_in.index(ch_sub[0]),
                            channels_in.index(ch_sub[1]),
                        ]
                    )
                else:
                    match_list.append(-1)
        return match_list

    @staticmethod
    def tr_val_indices(dataset_size, ratio):
        indices = list(range(dataset_size))
        split = int(np.floor(ratio * dataset_size))
        np.random.shuffle(indices)
        return indices[:split], indices[split:]

    @staticmethod
    def partition_list(a, k):
        if k <= 1:
            return [a]
        # if k <= 1: return list(range(1, len(a)+1))
        if k >= len(a):
            return [[x] for x in a]
        partition_between = [(i + 1) * len(a) // k for i in range(k - 1)]
        average_height = float(sum(a)) / k
        best_score = None
        count = 0
        best_ends = None

        while True:
            starts = [0] + partition_between
            ends = partition_between + [len(a)]
            partitions = [a[starts[i] : ends[i]] for i in range(k)]
            heights = list(map(sum, partitions))

            abs_height_diffs = list(
                map(lambda x: abs(average_height - x), heights)
            )
            worst_partition_index = abs_height_diffs.index(
                max(abs_height_diffs)
            )
            worst_height_diff = average_height - heights[worst_partition_index]

            if best_score is None or abs(worst_height_diff) < best_score:
                best_score = abs(worst_height_diff)
                best_ends = ends
                no_improvements_count = 0
            else:
                no_improvements_count += 1

            if (
                worst_height_diff == 0
                or no_improvements_count > 5
                or count > 100
            ):
                return best_ends
            count += 1

            move = -1 if worst_height_diff < 0 else 1
            bound_to_move = (
                0
                if worst_partition_index == 0
                else (
                    k - 2
                    if worst_partition_index == k - 1
                    else (
                        worst_partition_index - 1
                        if (worst_height_diff < 0)
                        ^ (
                            heights[worst_partition_index - 1]
                            > heights[worst_partition_index + 1]
                        )
                        else worst_partition_index
                    )
                )
            )
            direction = -1 if bound_to_move < worst_partition_index else 1
            partition_between[bound_to_move] += move * direction

    @staticmethod
    def split_func(
        all_seg, files_ind, fill_empty=False, saving_seizure_all=False
    ):
        """
        Splits the segments to the files
        fill_empty is mainly for when we don't have seizure and we are
        going to load previous seizure in this mode, files_ind is also
        update to include the seizure files
        """
        assert not (
            fill_empty and saving_seizure_all
        ), "These two modes should not be True togethor"
        file_ind_out = copy.deepcopy(files_ind)
        seg_list = np.split(
            all_seg, np.unique(all_seg[:, 0], return_index=True)[1][1:]
        )
        seg_f_ind_list = [int(seg_l[0][0]) for seg_l in seg_list]
        all_part = []
        i = 0
        all_sei_files = []
        # This part is not optimized and can be improved later
        last_sei_files = []
        for f_ind, group_ind in enumerate(files_ind):
            sei_files = []
            part = np.empty((0, 4))
            for i, ind in enumerate(seg_f_ind_list):
                if ind in group_ind:
                    part = np.append(part, seg_list[i], axis=0)
                    sei_files.append(ind)
            if part.size == 0 and fill_empty:
                ## loading last seizures
                part = all_part[-1]
                file_ind_out[f_ind] = np.append(
                    file_ind_out[f_ind], last_sei_files
                )
            else:
                last_sei_files = sei_files
            all_sei_files.append(sei_files)
            all_part.append(part)

        if saving_seizure_all:
            # Merging seizure files - Adding all previous seizures,
            # we don't add the seizures at the moment because it is
            # already there
            acc_sei_files = []
            for i in range(len(all_sei_files)):
                temp = []
                for j in range(i):
                    if all_sei_files[j]:
                        temp.extend(all_sei_files[j])
                acc_sei_files.append(temp)
            for i in range(1, len(all_part)):
                all_part[i] = np.concatenate((all_part[i - 1], all_part[i]))
                file_ind_out[i] = np.append(file_ind_out[i], acc_sei_files[i])

        return all_part, file_ind_out

    @staticmethod
    def split_func_v2(all_seg, files_ind):
        """
        Splits the segments to the files (1hr) - only for non-seizures
        the main difference is that here the bigger files are split to 1hr
        data, also some of the shorter ones are merged
        """
        file_ind_out = copy.deepcopy(files_ind)

        seg_list = np.split(
            all_seg, np.unique(all_seg[:, 0], return_index=True)[1][1:]
        )

        ## reordering the seg_list
        files_ind_base = [f_ind[0] for f_ind in files_ind]
        seg_list = [seg_list[i] for i in files_ind_base]

        # Making sure all of the subsegments are ordered
        for i, seg_l in enumerate(seg_list):
            assert all(
                seg_l[i][0] >= seg_l[i - 1][0] for i in range(1, len(seg_l))
            ), "First dimension is not in increasing order"

        # Spliting > 1 hr recordings
        def split_based_on_time(data, ideal_ch_dur=3600):
            """
            Splits a bigger chunk to smaller one like 1 hr.
            chunk = duratoin of the chunk (3600 seconds)
            """
            duration = data[-1][2]
            num_sub_segmets = round(duration / ideal_ch_dur)
            ch_dur = duration / num_sub_segmets

            segments = []
            current_segment = []
            time_cords = []
            i = 1

            for row in data:
                if not current_segment:
                    current_segment.append(row)
                    time_cords.append([row[1], 0])
                else:
                    if row[2] <= i * ch_dur:
                        current_segment.append(row)
                    else:
                        segments.append(current_segment)
                        time_cords[i - 1][1] = row[1]
                        time_cords.append([row[1], 0])
                        current_segment = [row]
                        i += 1

            if current_segment:
                segments.append(current_segment)
                time_cords[i - 1][1] = row[2]

            segments = [np.array(seg) for seg in segments]
            return segments, time_cords

        # Spliting the bigger segments
        i = 0
        while i < len(seg_list):
            if len(seg_list[i]) > 1350 * 4:  # 900 + 450
                segments, time_cords = split_based_on_time(seg_list[i])
                seg_list.pop(i)
                f_ind = file_ind_out.pop(i)[0]
                seg_list = seg_list[:i] + segments + seg_list[i:]

                for t_start, t_end in time_cords:
                    file_ind_out.insert(i, [[f_ind, t_start, t_end]])
                    i += 1
            else:
                i += 1

        # Merging less than 0.5 files with perivous ones.
        # Note: this apply for this dataset and except for one case,
        # it is better to merge with previous one. Better solution is
        # to do smarte spliting (just min(pre, next) is not enough)

        if len(seg_list[0]) < 450 * 4:
            seg_list[0] = np.concatenate(
                (seg_list[0], seg_list.pop(1)), axis=0
            )

            file_ind_out[0] += file_ind_out.pop(1)
        i = 1
        while i < len(seg_list):
            if len(seg_list[i]) < 450 * 4:
                seg_list[i - 1] = np.concatenate(
                    (seg_list[i - 1], seg_list.pop(i)), axis=0
                )

                file_ind_out[i - 1] += file_ind_out.pop(i)
            else:
                i += 1

        return seg_list, file_ind_out

    @staticmethod
    def split_func_v2_seizure(all_seg, files_ind):
        """
        Splits the segments to the files (1hr)
        the main difference is that here the bigger files are split to 1hr
        data, also some of the shorter ones are merged
        """
        file_ind_out = copy.deepcopy(files_ind)

        seg_list = np.split(
            all_seg, np.unique(all_seg[:, 0], return_index=True)[1][1:]
        )
        seg_f_ind_list = [int(seg_l[0][0]) for seg_l in seg_list]

        # Making sure all of the subsegments are ordered
        for i, seg_l in enumerate(seg_list):
            assert all(
                seg_l[i][0] >= seg_l[i - 1][0] for i in range(1, len(seg_l))
            ), "First dimension is not increasing"

        all_part = []
        i = 0
        all_sei_files = []
        for f_ind, group_ind in enumerate(files_ind):
            sei_files = []
            part = np.empty((0, 4))

            for ind in group_ind:
                if isinstance(ind, list):
                    if ind[0] in seg_f_ind_list:
                        file_ind = seg_f_ind_list.index(ind[0])
                        whole_seg = seg_list[file_ind]
                        selected_data = whole_seg[
                            (whole_seg[:, 1] >= ind[1])
                            & (whole_seg[:, 1] < ind[2])
                        ]
                        part = np.append(part, selected_data, axis=0)
                if isinstance(ind, int):
                    if ind in seg_f_ind_list:
                        file_ind = seg_f_ind_list.index(ind)
                        part = np.append(part, seg_list[file_ind], axis=0)

            all_sei_files.append(sei_files)
            all_part.append(part)

        ## reording the seizure - non-seizure segments and files
        len_list = [len(seg_l) for seg_l in all_part]
        first_sei = next(
            (index for index, num in enumerate(len_list) if num != 0), None
        )
        last_sei = next(
            (
                index
                for index, num in reversed(list(enumerate(len_list)))
                if num != 0
            ),
            None,
        )

        all_part.insert(0, all_part.pop(first_sei))
        all_part.append(all_part.pop(last_sei))

        file_ind_out.insert(0, file_ind_out.pop(first_sei))
        file_ind_out.append(file_ind_out.pop(last_sei))

        ## Removing unnecessary parts of the file_ind_out
        for i in range(len(file_ind_out)):
            for j in range(len(file_ind_out[i])):
                if isinstance(file_ind_out[i][j], list):
                    file_ind_out[i][j] = file_ind_out[i][j][0]

        return all_part, file_ind_out

    class CHBMITest(Dataset):
        """
        Args:
        ratio: ratio between the non-seizure-seizure data for undersampling.
                    -1 : default for validation -> no oversampling on seizure.
                    -1<: default for training -> oversampling for seizure.
                    0>: used for the undersampling the non-seizure.
        Note: the data file list should be sorted by id (could be
                        improved but I don't wanna spend time on this now :))
        """

        def __init__(self, data, labels, seg_list) -> None:
            self.win_sec = 4
            self.label_choice = "2nd"
            ## Checking for the seg_list to be sorted by file_id and start time
            assert np.all(
                seg_list[:-1, 0] <= seg_list[1:, 0]
            ), "File_id is not in increasing order"
            unique_file_ids = np.unique(seg_list[:, 0])
            for file_id in unique_file_ids:
                rows = seg_list[seg_list[:, 0] == file_id]
                assert np.all(
                    rows[:-1, 1] <= rows[1:, 1]
                ), "Start time is not in increasing order"
            self.data = data
            self.labels = labels
            self.seg_list_base = seg_list
            self.valid_indices = self._compute_valid_indices()
            self.seg_list = self._compute_new_seg_list()

        def _compute_valid_indices(self):
            valid_indices = []
            for i in range(len(self.seg_list_base) - 3):
                if self.seg_list_base[i][0] == self.seg_list_base[i + 3][0]:
                    end_time = self.seg_list_base[i][2]
                    next_end_time = self.seg_list_base[i + 3][2]
                    if next_end_time == end_time + 3:
                        valid_indices.append(i)

            return valid_indices

        def _compute_new_seg_list(self):
            seg_list = self.seg_list_base[self.valid_indices]
            for i, ind in enumerate(self.valid_indices):
                ## Updating finish time
                seg_list[i][2] = self.seg_list_base[ind + 3][2]
                ## Updating label
                if self.label_choice == "2nd":
                    seg_list[i][3] = self.labels[ind + 1]
                elif self.label_choice == "3rd":
                    seg_list[i][3] = self.labels[ind + 2]

            return seg_list

        @classmethod
        def from_rec(cls, segments, matches_list, data_file_list, file_ind):
            assert len(matches_list) == len(
                data_file_list
            ), "matches file length is different from file list"
            fs = 256
            data = []
            labels = []

            ## sorting by recordings' id
            segments = segments[segments[:, 0].astype(int).argsort()]
            seg_list = np.split(
                segments[:, :],
                np.unique(segments[:, 0].astype(int), return_index=True)[1][
                    1:
                ],
            )

            ## Sorting the seg_l in seg_list
            seg_list = [seg_l[seg_l[:, 1].argsort()] for seg_l in seg_list]

            assert len(file_ind) == len(
                np.unique(segments[:, 0].astype(int))
            ), "file_ind is not equal to the number of files"
            if not len(seg_list) == len(matches_list) == len(data_file_list):
                print(len(seg_list), len(matches_list), len(data_file_list))
            assert (
                len(seg_list) == len(matches_list) == len(data_file_list)
            ), "The len is not equal between these lists"
            for seg_l, match_list, data_file in zip(
                seg_list, matches_list, data_file_list
            ):
                rec = cls.get_rec(match_list, data_file)

                rec = cls.pre_process_ch(rec, fs)

                for seg in seg_l:
                    start_seg = int(seg[1] * fs)
                    stop_seg = int(seg[2] * fs)
                    data.append(rec[:, start_seg:stop_seg])
                    labels.append(int(seg[3]))

            seg_list = np.concatenate(seg_list)
            indices = np.arange(seg_list.shape[0]).reshape(1, -1).T
            seg_list = np.concatenate([seg_list, indices], axis=1)
            data = np.array(data)
            labels = np.array(labels)
            assert (
                len(seg_list) == len(data) == len(labels)
            ), "different Length of seg and data labels"

            neg = len([x for x in labels if x == 0])
            pos = len([x for x in labels if x == 1])
            print(f"Tot number of windows: {int(len(labels))}")
            print(
                "Positive Windows: {} ({:.2f}%)".format(
                    pos, 100 * pos / len(labels)
                )
            )
            print(
                "Negative Windows: {} ({:.2f}%)\n".format(
                    neg, 100 * neg / len(labels)
                )
            )
            return cls(data, labels, seg_list)

        @classmethod
        def from_pkl(cls, pkl_obj):
            return cls(pkl_obj.data, pkl_obj.labels, pkl_obj.seg_list)

        @staticmethod
        def pre_process_ch(ch_data, fs):
            """Pre-process EEG data by applying a 0.5 Hz highpass filter,
            a 60  Hz lowpass filter and a 50 Hz notch filter, all
            4th order Butterworth filters. The data is resampled to 200 Hz.

            Args:
                ch_data: a list or numpy array containing the data of
                    an EEG channel
                fs: the sampling frequency of the data

            Returns:
                ch_data: a numpy array containing the processed EEG data
                fs_resamp: the sampling frequency of the processed EEG data
            """
            b, a = signal.butter(4, 0.5 / (fs / 2), "high")
            ch_data = signal.filtfilt(b, a, ch_data, axis=1)

            b, a = signal.butter(4, 60 / (fs / 2), "low")
            ch_data = signal.filtfilt(b, a, ch_data, axis=1)

            b, a = signal.butter(
                4, [49.5 / (fs / 2), 50.5 / (fs / 2)], "bandstop"
            )
            ch_data = signal.filtfilt(b, a, ch_data, axis=1)

            return ch_data

        @staticmethod
        def get_rec(match_list, data_file):
            """
            Prepares the recording files

            Assumptions based on the input data:
                    1. all of the missing data are labeled with -1 in the
                            match_list
                    2. missing channels are always at the end of the list
            """
            data = data_file.get_data()
            # using the assumption of all -1s are at the end
            match_list = list(filter(lambda a: a != -1, match_list))
            rec = np.zeros((len(match_list), data.shape[1]))
            if isinstance(match_list[0], list):  # case for patient 12
                for i, ch_list in enumerate(match_list):
                    rec[i] = data[ch_list[0]] - data[ch_list[1]]
            else:
                rec[: len(match_list)] = data[match_list]

            return rec

        def __len__(self):
            return len(self.valid_indices)

        def __getitem__(self, index):
            real_index = self.valid_indices[index]

            window_data = np.concatenate(
                self.data[real_index : real_index + 4], axis=1
            )

            # Select label based on label_choice hyperparameter
            if self.label_choice == "2nd":
                label = self.labels[real_index + 1]
            elif self.label_choice == "3rd":
                label = self.labels[real_index + 2]

            return window_data, label

        def extend_dataset(self, data, labels, seg_list):
            # Can still update this to keep the main seizure indices
            # Can update to overlap the seizures if possible
            if len(labels) == 0:
                return
            assert len(data) == len(
                labels
            ), "data length does not match labels"
            assert (
                len(self.seg_list) == int(np.max(self.seg_list[:, 4])) + 1
            ), "seg_list index and size do not match"
            new_indices = np.arange(
                len(self.seg_list), len(self.seg_list) + len(seg_list)
            )
            seg_list[:, 4] = new_indices

            self.seg_list = np.concatenate((self.seg_list, seg_list), axis=0)
            self.data = np.concatenate((self.data, data))
            self.labels = np.concatenate((self.labels, labels))

        def get_all_seizures(self):
            """
            Returns all of seizures
            """
            indices = np.where(self.labels == 1.0)[0]
            return (
                self.data[indices],
                self.labels[indices],
                self.seg_list[indices],
            )

        def reduce_NS(self, rate):
            """
            Reduces the number of Non-seizures for rate %.
            """
            indices = np.where(self.labels == 0)[0]
            del_cnt = int(np.round(rate * len(indices)))
            np.random.shuffle(indices)
            del_indices = indices[:del_cnt]
            self.data = np.delete(self.data, del_indices, axis=0)
            self.labels = np.delete(self.labels, del_indices)
            print(f"Removed {del_cnt} Non-Seizures")

        def get_all_Nonseizures(self):
            """
            Returns all of Non seizures
            """
            indices = np.where(self.labels == 0.0)[0]
            return (
                self.data[indices],
                self.labels[indices],
                self.seg_list[indices],
            )

        def get_all_Nonseizures_indices(self):
            """
            Returns all of Non seizures' indexes
            """
            return np.where(self.labels == 0.0)[0]

        def get_all_NonSeizures_Sf(self):
            """
            returning all of the nonseizures from the same file as the
            seizures has happened

            No valid any more seg_list structure has changed
            """
            indices = np.where(self.seg_list[:, 3] == 1.0)[0]
            seg_list_seiz = self.seg_list[indices]
            unq_files = np.unique(seg_list_seiz[:, 0])
            new_ns = np.where(np.isin(self.seg_list[:, 0], unq_files))[0]
            mask = ~np.isin(new_ns, indices)
            new_ns = new_ns[mask]

            return self.data[new_ns], self.labels[new_ns]

        def repeat_seizures(self):
            s_indices = np.where(self.labels == 1.0)[0]
            ns_indices = np.where(self.labels == 0.0)[0]
            if len(s_indices) > 0:
                rep_cnt = len(ns_indices) // len(s_indices)
                if rep_cnt:
                    repeat_counts = np.ones(len(self.labels), dtype=int)
                    repeat_counts[s_indices] = rep_cnt
                    self.labels = np.repeat(self.labels, repeat_counts, axis=0)
                    self.data = np.repeat(self.data, repeat_counts, axis=0)
