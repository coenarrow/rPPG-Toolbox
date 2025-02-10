"""The dataloader for the Neckflix dataset.

Details for the Neckflix Dataset see ####
If you use this dataset, please cite this paper:
C. Arrow, M. Ward, J. Eshraghian, G. Dwivedi.
"Neckflix:"
"""
import glob
import os
import numpy as np
import cv2
import numpy as np
from dataset.data_loader.BaseLoader import BaseLoader
import pandas as pd
from collections import defaultdict
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d
from typing import Dict, Tuple, List
import pickle
import av
import re
from tqdm import tqdm

# testing imports
import argparse
from neural_methods import trainer
from dataset import data_loader

class NeckflixLoader(BaseLoader):
    """The data loader for the Neckflix dataset."""

    def __init__(self, name, data_path, config_data):
        """Initializes an Neckflix dataloader.
            Args:
                data_path(str): path of a folder which stores raw video and bvp data.
                e.g. data_path should be "RawData" for below dataset structure:
                -----------------
                     RawData/
                     |   |-- P001_S01_R1_0_D/
                     |       |-- video_start_end_times.csv
                     |       |-- K1_Depth.mkv
                     |       |-- K1_IR.mkv
                     |       |-- K1_RGB.mkv
                     |       |-- K2_Depth.mkv
                     |       |-- K2_IR.mkv
                     |       |-- K2_RGB.mkv
                     |       |...
                     |       |-- trace_data.csv
                     |   |-- P001_S01_R2_0_N/
                     |       |-- video_start_end_times.csv
                     |       |-- K1_RGB.mkv
                     |       |-- K2_RGB.mkv
                     |       |...
                     |       |-- trace_data.csv
                     |...
                     |   |-- Px_Sy_Rz/
                     |       |-- video_start_end_times.csv
                     |       |-- K1_Depth.mkv
                     |       |-- K1_IR.mkv
                     |       |-- K1_RGB.mkv
                     |       |-- K2_Depth.mkv
                     |       |-- K2_IR.mkv
                     |       |-- K2_RGB.mkv
                     |       |...
                     |       |-- trace_data.csv
                -----------------
                name(string): name of the dataloader.
                config_data(CfgNode): data settings(ref:config.py).
        """
        self.inputs = list()
        self.labels = list()
        self.dataset_name = name
        self.raw_data_path = data_path
        self.cached_path = self.update_default_cached_path(config_data)
        self.file_list_path = self.update_default_file_list_path(config_data)
        self.preprocessed_data_len = 0
        self.data_format = config_data.DATA_FORMAT
        self.do_preprocess = config_data.DO_PREPROCESS
        self.config_data = config_data
        assert (config_data.BEGIN < config_data.END)
        assert (config_data.BEGIN > 0 or config_data.BEGIN == 0)
        assert (config_data.END < 1 or config_data.END == 1)
        recording_dict_list = self.get_recording_dicts(self.raw_data_path)
        self.raw_data_dirs = self.split_and_filter_recording_dicts(recording_dict_list, config_data)
        if config_data.DO_PREPROCESS:
            self.preprocess_dataset(self.raw_data_dirs, self.config_data)
        else:
            pass
            if not os.path.exists(self.cached_path):
                print('CACHED_PATH:', self.cached_path)
                raise ValueError(self.dataset_name,
                                 'Please set DO_PREPROCESS to True. Preprocessed directory does not exist!')
            if not os.path.exists(self.file_list_path):
                print('File list does not exist... generating now...')
                print('Need to Implement this function')
                self.build_file_list_retroactive(self.raw_data_dirs, config_data.BEGIN, config_data.END)
                print('File list generated.', end='\n\n')
            self.load_preprocessed_data()
        # print('Cached Data Path', self.cached_path, end='\n\n')
        # print('File List Path', self.file_list_path)
        # print(f" {self.dataset_name} Preprocessed Dataset Length: {self.preprocessed_data_len}", end='\n\n')

    def __getitem__(self,index):
        with open(self.inputs[index], 'rb') as handle:
            data = pickle.load(handle)
        for key, value in data.items():
            if self.data_format == 'NDCHW':
                data[key] = np.float32(np.transpose(value, (0, 3, 1, 2)))
            elif self.data_format == 'NCDHW':
                data[key] = np.float32(np.transpose(value, (3, 0, 1, 2)))
            elif self.data_format == 'NDHWC':
                pass
            else:
                raise ValueError('Unsupported Data Format!')

        # If we aren't using bigsmall, there will only be 1 key in the data dictionary
        # so just load the data from that key
        if len(list(data.keys()))==1:
            data = data[list(data.keys())[0]]
        else:
            pass
        
        label = np.load(self.labels[index])
        label = np.float32(label[:,1:].squeeze()) # we don't want to load the time label

        # item_path is the location of a specific clip in a preprocessing output folder
        # For example, an item path could be /home/data/PURE_SizeW72_...unsupervised/501_input0.npy
        item_path = self.inputs[index]
        # item_path_filename is simply the filename of the specific clip
        # For example, the preceding item_path's filename would be 501_input0.npy
        item_path_filename = item_path.split(os.sep)[-1]
        # split_idx represents the point in the previous filename where we want to split the string 
        # in order to retrieve a more precise filename (e.g., 501) preceding the chunk (e.g., input0)
        split_idx = item_path_filename.rindex('_')
        # Following the previous comments, the filename for example would be 501
        filename = item_path_filename[:split_idx]
        # chunk_id is the extracted, numeric chunk identifier. Following the previous comments, 
        # the chunk_id for example would be 0
        chunk_id = item_path_filename[split_idx + 6:].split('.')[0]
        
        return data, label, filename, chunk_id

    def update_default_cached_path(self, config_data):
        """
        If the default cached path is used, updates it to include current settings.
        """
        default_cached_path = "_".join(
        [   config_data.DATASET, 
            "SizeW{0}".format(str(config_data.PREPROCESS.RESIZE.W)), 
            "SizeH{0}".format(str(config_data.PREPROCESS.RESIZE.W)), 
            "ClipLength{0}".format(str(config_data.PREPROCESS.CHUNK_LENGTH)), 
            "DataType{0}".format("_".join(config_data.PREPROCESS.DATA_TYPE)),
            "DataAug{0}".format("_".join(config_data.PREPROCESS.DATA_AUG)),
            "LabelType{0}".format(config_data.PREPROCESS.LABEL_TYPE),
            "Crop_face{0}".format(config_data.PREPROCESS.CROP_FACE.DO_CROP_FACE),
            "Backend{0}".format(config_data.PREPROCESS.CROP_FACE.BACKEND),
            "Large_box{0}".format(config_data.PREPROCESS.CROP_FACE.USE_LARGE_FACE_BOX),
            "Large_size{0}".format(config_data.PREPROCESS.CROP_FACE.LARGE_BOX_COEF),
            "Dyamic_Det{0}".format(config_data.PREPROCESS.CROP_FACE.DETECTION.DO_DYNAMIC_DETECTION),
            "det_len{0}".format(config_data.PREPROCESS.CROP_FACE.DETECTION.DYNAMIC_DETECTION_FREQUENCY),
            "Median_face_box{0}".format(config_data.PREPROCESS.CROP_FACE.DETECTION.USE_MEDIAN_FACE_BOX)
        ])

        if default_cached_path in config_data.CACHED_PATH:
            # Keep current directory settings
            directory = config_data.CACHED_PATH.split(default_cached_path)[0]
            # Dataset
            dataset = config_data.DATASET
            # video setting
            video_setting = [f"CameraMode{config_data.PREPROCESS.NECKFLIX.CAMERA_MODE}",
                            f"DataAug{''.join(config_data.PREPROCESS.DATA_AUG)}",
                            f"DoChunk{config_data.PREPROCESS.DO_CHUNK}",
                            f"ChunkLength{config_data.PREPROCESS.CHUNK_LENGTH}"]
            video_setting_string = "-".join(video_setting)
            # label processing settings
            if (config_data.PREPROCESS.USE_PSUEDO_PPG_LABEL) and ('RGB' in config_data.PREPROCESS.NECKFLIX.CAMERA_MODE):
                labels = config_data.PREPROCESS.NECKFLIX.LABELS + ['POS_PPG']
            else:
                labels = config_data.PREPROCESS.NECKFLIX.LABELS
            label_string = f"Labels{'-'.join(labels)}"
            label_string = label_string + f"-LabelType{config_data.PREPROCESS.LABEL_TYPE}"
            resize_and_processing = [f"UseBigSmall{config_data.PREPROCESS.USE_BIGSMALL}"]
            # video processing settings
            if config_data.PREPROCESS.USE_BIGSMALL:
                resize_and_processing = resize_and_processing + [f"BigW{config_data.PREPROCESS.BIGSMALL.RESIZE.BIG_W}",
                                                                    f"BigH{config_data.PREPROCESS.BIGSMALL.RESIZE.BIG_H}",
                                                                    f"BigData{'-'.join(config_data.PREPROCESS.BIGSMALL.BIG_DATA_TYPE)}",
                                                                    f"SmallW{config_data.PREPROCESS.BIGSMALL.RESIZE.SMALL_W}",
                                                                    f"SmallH{config_data.PREPROCESS.BIGSMALL.RESIZE.SMALL_H}",
                                                                    f"SmallData{'-'.join(config_data.PREPROCESS.BIGSMALL.SMALL_DATA_TYPE)}"]
            else:
                resize_and_processing = resize_and_processing + [f"W{config_data.PREPROCESS.RESIZE.W}",
                                                                f"H{config_data.PREPROCESS.RESIZE.H}",
                                                                f"Video{'-'.join(config_data.PREPROCESS.DATA_TYPE)}"]
            resize_and_processing_string = "-".join(resize_and_processing)
            setting_strings = [dataset,video_setting_string,
                               label_string,
                               resize_and_processing_string,
                               #f"Begin{config_data.BEGIN}-End{config_data.END}",
                               ]
            new_cached_path = directory + "_".join(setting_strings)
            return new_cached_path
        else:
            return config_data.CACHED_PATH

    def update_default_file_list_path(self, config_data):
        file_list_path = config_data.FILE_LIST_PATH
        file_list_path_split = file_list_path.split(os.sep)
        file_list_path_split[-1] = self.cached_path +'.csv'
        new_file_list_path = os.path.join(*file_list_path_split)
        return new_file_list_path

    def get_recording_dicts(self, data_path) -> List[dict]:
        """
        Returns a list of dictionaries for the dataset prior to expansion and filtering.
        """
        # Find all video files in subdirectories
        video_filepaths = sorted(
            glob.glob(os.path.join(data_path, "*", "*.mkv")) +
            glob.glob(os.path.join(data_path, "*", "*.hdf5"))
        )
        # Get unique data directories
        data_dirs = np.unique([os.path.dirname(video_file) for video_file in video_filepaths])
        
        dict_list = []
        
        for data_dir in data_dirs:
            # Extract index, participant and position info from directory name
            index = os.path.split(data_dir)[-1]
            parts = index.split('_')
            participant = parts[0]
            pos = parts[-2]
            if pos == '0':
                position = 'supine'
            elif pos == '45':
                position = 'recumbent'
            elif pos == '90':
                position = 'sitting'
            else:
                raise ValueError(f"Position {pos} not recognized!")

            # Check for trace_data.csv to determine available signals
            labels_path = os.path.join(data_dir, 'trace_data.csv')
            if os.path.exists(labels_path):
                labels = pd.read_csv(labels_path, nrows=0).columns
                cvp = 'CVP (mmHg)' in labels
                ecg = 'ECG (mV)' in labels
                abp = 'ABP (mmHg)' in labels
            else:
                cvp, ecg, abp = False, False, False

            # Read start times from CSV
            start_times_path = os.path.join(data_dir, 'video_start_end_times.csv')
            start_times_df = pd.read_csv(start_times_path)


            # Build cameras dictionary by iterating over files in data_dir
            cameras = {}
            for file in os.listdir(data_dir):
                # If file is bytes, decode it to string
                if isinstance(file, bytes):
                    file = file.decode('utf-8')
                    
                # Only process video files
                if not (file.endswith('.mkv') or file.endswith('.hdf5')):
                    continue

                # Skip CSV files
                if file in ['trace_data.csv', 'video_start_end_times.csv']:
                    continue

                # Use regex to extract camera and modality (if present)
                # Expected filenames: "K1_RGB.mkv", "K2_IR.mkv", or "EV.hdf5"
                pattern = r'(?P<camera>[A-Za-z0-9]+)(?:_(?P<modality>[A-Za-z]+))?\.(?:mkv|hdf5)$'
                match = re.match(pattern, file)
                if not match:
                    continue

                camera = match.group('camera')
                modality = match.group('modality')  # This may be None
                file_path = os.path.join(data_dir, file)
                
                # Lookup the start time from the CSV
                matching_row = start_times_df[start_times_df['Video_Name'] == file]
                start_time_val = (matching_row['True_Start_Time'].values[0] / 1e6 
                                if not matching_row.empty else None)
                
                # Organize into the cameras dictionary
                if modality:
                    # If modality exists, group under the camera
                    if camera not in cameras:
                        cameras[camera] = {}
                    cameras[camera][modality] = {"path": file_path, "start_time": start_time_val}
                else:
                    # For files without a modality (e.g., EV.hdf5)
                    cameras[camera] = {"path": file_path, "start_time": start_time_val}

            # Assemble the directory's info dictionary
            data_dir_info = {
                "index": index,
                "participant": participant,
                "position": position,
                "labels_path": labels_path,
                "CVP": cvp,
                "ECG": ecg,
                "ABP": abp,
                "cameras": cameras,  # Contains keys like 'K1', 'K2', 'EV', etc.
            }
            dict_list.append(data_dir_info)
        
        return dict_list

    def filter_data_by_labels(self, dict_list, selected_labels):
        """ 
        Only keeps the dictionaries that have the desired traces (labels)
        """
        filtered_dicts = []
        for data_dict in dict_list:
            if np.all([data_dict[label] for label in selected_labels]):
                filtered_dicts.append(data_dict)
        return filtered_dicts

    def split_dicts_by_camera(self,dict_list):
        """
        Split a list of data dictionaries (each with a 'cameras' key containing multiple cameras)
        into a list of dictionaries where each dictionary corresponds to a single camera.

        Each resulting dictionary retains the common metadata from the original dictionary,
        but the 'cameras' key will contain only one camera's data. Additionally, a new key
        'camera' is added to specify which camera the dictionary represents.

        Args:
            dict_list (list): List of dictionaries. Each dictionary is expected to have a 'cameras'
                            key mapping camera names to their respective information.

        Returns:
            list: A new list of dictionaries, one per camera.
        """
        split_list = []
        for d in dict_list:
            cameras = d.get("cameras", {})
            for cam, cam_data in cameras.items():
                # Create a shallow copy of the original dictionary to preserve metadata.
                new_d = d.copy()
                new_d["cameras"] = {cam: cam_data}
                new_d["camera"] = cam  # Optional: directly include the camera name.
                split_list.append(new_d)
        return split_list

    def filter_by_camera_mode(self,dict_list, camera_mode):
        """
        Filters a list of data dictionaries (each for a single camera) based on the specified camera mode.
        Each dictionary is expected to have a 'camera' key indicating the camera name and a 'cameras' dict
        with modality keys mapping to a dictionary that contains a 'path' key.
        
        Args:
            dict_list (list): List of dictionaries (each already split for a single camera).
            camera_mode (str): One of 'RGB', 'RGBI', 'RGBD', 'RGBID', 'I', 'ID', 'D', or 'EV'.
            
        Returns:
            list: A filtered list containing only dictionaries where all required modalities have a valid path.
        """
        # Define the required modalities based on camera_mode.
        if camera_mode == 'RGB':
            required_modalities = {'RGB'}
        elif camera_mode == 'RGBI':
            required_modalities = {'RGB', 'IR'}
        elif camera_mode == 'RGBD':
            required_modalities = {'RGB', 'Depth'}
        elif camera_mode == 'RGBID':
            required_modalities = {'RGB', 'IR', 'Depth'}
        elif camera_mode == 'I':
            required_modalities = {'IR'}
        elif camera_mode == 'ID':
            required_modalities = {'IR', 'Depth'}
        elif camera_mode == 'D':
            required_modalities = {'Depth'}
        elif camera_mode == 'EV':
            required_modalities = {'EV'}
        else:
            raise ValueError(f"Camera mode {camera_mode} not recognized!")

        filtered_list = []
        if camera_mode !='EV':
            for d in dict_list:
                try:
                    # Expect that each dictionary has a 'camera' key with the camera name.
                    cam = d['camera']
                    # Check that for each required modality, the modality exists and its 'path' is not None.
                    if all(d['cameras'][cam][modality]['path'] is not None for modality in required_modalities):
                        filtered_list.append(d)
                except KeyError:
                    # If any key is missing (e.g. modality not present), skip this dictionary.
                    continue
            return filtered_list
        else:
            for d in dict_list:
                cam = d['camera']
                if cam == 'EV':
                    filtered_list.append(d)
            return filtered_list

    def split_and_filter_recording_dicts(self, recording_dicts:list, config_data):
        """
        Splits the recording dictionaries based on stereo, camera mode and trace labels.
        """
        print(f"Number of Recordings Found: {len(recording_dicts)}")
        filtered_dicts = recording_dicts

        # Split the dataset by camera if we're not using stereo mode
        if config_data.PREPROCESS.NECKFLIX.STEREO_MODE == False:
            filtered_dicts = self.split_dicts_by_camera(filtered_dicts)
            print(f"Num files after separating by cameras: {len(filtered_dicts)}")
            filtered_dicts = self.filter_by_camera_mode(filtered_dicts, config_data.PREPROCESS.NECKFLIX.CAMERA_MODE)
            print(f"Num files after filtering by camera mode: {len(filtered_dicts)}")
        else:
            raise ValueError("Stereo Mode not yet supported, set NECKFLIX.STEREO_MODE to False")

        # First filter the dataset by labels
        filtered_dicts = self.filter_data_by_labels(filtered_dicts, self.config_data.PREPROCESS.NECKFLIX.LABELS)
        print(f"Num Files after filtering by label: {len(filtered_dicts)}")
        for recording_dict in filtered_dicts:
            new_index = recording_dict['index'] + '_' +  ''.join(recording_dict['cameras'].keys())
            recording_dict['index'] = new_index
        return filtered_dicts

    def split_by_participant(self,data_dirs, begin, end):
        """Returns a subset of data dirs, split with begin and end values, 
        and ensures no overlapping subjects between splits"""

        # Return the full directory if range covers the entire dataset
        if begin == 0 and end == 1:
            return data_dirs

        # Group data by subject
        data_info = defaultdict(list)
        for data in data_dirs:
            data_info[data['participant']].append(data)

        # Sort subjects and calculate range of interest
        subj_list = sorted(data_info.keys())
        num_subjs = len(subj_list)
        subj_range = subj_list[int(begin * num_subjs):int(end * num_subjs)]

        # Compile the subset of data
        subset = [data for subj in subj_range for data in data_info[subj]]
        return subset
    
    def preprocess_dataset(self, dict_list, config_data):
        print('Starting Preprocessing...')
        

        # Split the dataset by participants
        split_dict_list  = self.split_by_participant(dict_list, self.config_data.BEGIN, self.config_data.END)
        print(f"Num Files after splitting by participant: {len(split_dict_list)}")

        # REMOVE ALREADY PREPROCESSED SUBJECTS
        split_dict_list = self.adjust_data_dirs(split_dict_list)
        print(f"Num Files after removing already processed: {len(split_dict_list)}")

        # CREATE CACHED DATA PATH
        if not os.path.exists(self.cached_path):
            os.makedirs(self.cached_path, exist_ok=True)

        # # READ RAW DATA, PREPROCESS, AND SAVE PROCESSED DATA FILES
        file_list_dict = self.multi_process_manager(split_dict_list, config_data)

        self.build_file_list_retroactive(dict_list, config_data.BEGIN, config_data.END)
        
        self.load()  # load all data and corresponding labels (sorted for consistency)
        print("Total Number of raw files preprocessed:", len(dict_list), end='\n\n')
        print("Num loaded files", self.preprocessed_data_len)

    def load_and_align_frames(self, video_info_dict, config_data, alignment_tolerance=0.025) -> Tuple[List, np.ndarray]:
        """
        Loads all the videos from the dictionary depending on the camera mode and aligns the frames.
        Returns a tuple of numpy arrays.
        First array is the frames, with shape (N,H,W,C) (N: number of frames, H: height, W: width, C: channels)
        Second array is the times (in seconds), with shape (N,) (N: number of frames)
        """
        camera_mode = config_data.PREPROCESS.NECKFLIX.CAMERA_MODE
        if camera_mode == 'RGB':
            formats = ['RGB']
        elif camera_mode == 'RGBI':
            formats = ['RGB', 'IR']
        elif camera_mode == 'RGBD':
            formats = ['RGB', 'Depth']
        elif camera_mode == 'RGBID':
            formats = ['RGB', 'IR', 'Depth']
        elif camera_mode == 'I':
            formats = ['IR']
        elif camera_mode == 'ID':
            formats = ['IR', 'Depth']
        elif camera_mode == 'D':
            formats = ['Depth']
        elif camera_mode == 'EV':
            formats = ['EV']
        else:
            raise ValueError(f"Camera mode {camera_mode} not recognized!")

        all_frames = []
        all_frame_times = []

        for format in formats:
            if format == 'RGB':
                video_path = video_info_dict['cameras'][video_info_dict['camera']][format]['path']
                start_time = video_info_dict['cameras'][video_info_dict['camera']][format]['start_time']
                frames = self.read_video(video_path).astype(np.float32)
                frame_times = np.array([start_time+n/config_data.FS for n in range(len(frames))])
                all_frames.append(frames)
                all_frame_times.append(frame_times)
            elif format == 'Depth' or format == 'IR':
                video_path = video_info_dict['cameras'][video_info_dict['camera']][format]['path']
                start_time = video_info_dict['cameras'][video_info_dict['camera']][format]['start_time']
                frames = self.read_16bit_video(video_path).astype(np.float32)
                frame_times = np.array([start_time+n/config_data.FS for n in range(len(frames))])
                all_frames.append(frames)
                all_frame_times.append(frame_times)
            elif format == 'EV':
                raise ValueError(f'EV Video not yet supported')
            else:
                raise ValueError(f"Video type {format} not recognized!")
            

        # Align all the frames
        frames, times = self.align_frames(all_frames, all_frame_times, alignment_tolerance)
        return frames, times

    def load_traces(self,trace_filepath,frame_times,apply_filter=True):
        """
        Loads the trace data from a file and applies a low-pass filter if desired.
        """
        trace_df = pd.read_csv(trace_filepath)
        trace_times =  trace_df['Time (s)'].values
        sample_rate = int(1/(trace_times[1] - trace_times[0]))
        
        trace_dict = dict()
        for col in trace_df.columns:
            if col == 'Time (s)':
                continue
            sig_name = col.split(' ')[0]
            if sig_name not in self.config_data.PREPROCESS.NECKFLIX.LABELS:
                continue
            if apply_filter:
                trace_dict[sig_name] = self.lowpass_filter(np.array(trace_df[col].values),fs=sample_rate)
            else:
                trace_dict[sig_name] = trace_df[col].values

        # Interpolate the signals to the video frame times
        interpolated_dict = dict()
        for sig_name, signal in trace_dict.items():
            interpolator = interp1d(trace_times, signal, kind='cubic', bounds_error=False,fill_value=np.nan)
            interpolated_signal = interpolator(frame_times)
            interpolated_dict[sig_name] = interpolated_signal
        
        return interpolated_dict

    def construct_data_dict(self, video_info_dict, config_data) -> Dict:

        frames, frame_times = self.load_and_align_frames(video_info_dict, config_data)

        # Load the labels
        trace_dict = self.load_traces(video_info_dict['labels_path'],frame_times,apply_filter=True)

        # Get the index values where the signals are not nan
        trace_df = pd.DataFrame(trace_dict).dropna()
        selected_index = trace_df.index
        selected_frames = []
        for stream in frames:
            frames = stream[selected_index,:,:,:]
            selected_frames.append(frames)
        selected_times = frame_times[selected_index]

        # Add the selected traces and frames to the original dictionary
        data_dict = trace_df.to_dict(orient='list')
        data_dict['FRAMES'] = selected_frames
        data_dict['TIMES'] = selected_times

        return data_dict

    def preprocess_frames(self, data_dict, config_data, use_bigsmall:bool) -> List:
        frames = data_dict['FRAMES']
        f_c = frames.copy()
        if use_bigsmall:
            big_resized_frames = [self.resize_frames(stream, config_data.PREPROCESS.BIGSMALL.RESIZE.BIG_H, config_data.PREPROCESS.BIGSMALL.RESIZE.BIG_W) for stream in f_c]
            ### Big Frames
            for data_type in config_data.PREPROCESS.BIGSMALL.BIG_DATA_TYPE: # The loop allows multiple processing methods
                if data_type == "Raw":
                    big_data = big_resized_frames
                elif data_type == "DiffNormalized":
                    big_data = [self.diff_normalize_data(stream, exclude_mask=True) for stream in big_resized_frames]
                elif data_type == "Standardized":
                    big_data = [self.standardized_data(stream, exclude_mask=True) for stream in big_resized_frames]
                else:
                    raise ValueError("Unsupported data type!")
            big_data = np.concatenate(big_data, axis=-1)  # concatenate all channels
            ### Small Frames
            small_resized_frames = [self.resize_frames(stream, config_data.PREPROCESS.BIGSMALL.RESIZE.SMALL_H, config_data.PREPROCESS.BIGSMALL.RESIZE.SMALL_W) for stream in f_c]
            for data_type in config_data.PREPROCESS.BIGSMALL.SMALL_DATA_TYPE: # The loop allows multiple processing methods
                if data_type == "Raw":
                    small_data = small_resized_frames
                elif data_type == "DiffNormalized":
                    small_data = [self.diff_normalize_data(stream, exclude_mask=True) for stream in small_resized_frames]
                elif data_type == "Standardized":
                    small_data = [self.standardized_data(stream, exclude_mask=True) for stream in small_resized_frames]
                else:
                    raise ValueError("Unsupported data type!")
            small_data = np.concatenate(small_data, axis=-1)  # concatenate all channels
            preprocessed_data = [big_data, small_data]
        else:
            resized_frames = [self.resize_frames(stream, config_data.PREPROCESS.RESIZE.H, config_data.PREPROCESS.RESIZE.W) for stream in f_c]
            for data_type in config_data.PREPROCESS.DATA_TYPE:
                if data_type == "Raw":
                    data = resized_frames
                elif data_type == "DiffNormalized":
                    data = [self.diff_normalize_data(stream, exclude_mask=True) for stream in resized_frames]
                elif data_type == "Standardized":
                    data = [self.standardized_data(stream, exclude_mask=True) for stream in resized_frames]
                else:
                    raise ValueError("Unsupported data type!")
            data = np.concatenate(data, axis=-1)  # concatenate all channels
            preprocessed_data = [data]
        return preprocessed_data
    
    def preprocess_labels(self,data_dict, config_data) -> np.ndarray:
        processed_labels = list()
        ecg, cvp, abp, pos_ppg = None, None, None, None
        for key, value in data_dict.items():
            if key in ['FRAMES', 'TIMES']:
                continue
            if key == 'CVP':
                if config_data.PREPROCESS.NECKFLIX.CVP_NORMALIZATION is None:
                    cvp = value
                else:
                    min,max = config_data.PREPROCESS.NECKFLIX.CVP_NORMALIZATION
                    processed_label = self.normalize_pressure_label(np.array(value), min, max)
                    cvp = processed_label
            elif key == 'ABP':
                if config_data.PREPROCESS.NECKFLIX.ABP_NORMALIZATION is None:
                    abp = value
                else:
                    min,max = config_data.PREPROCESS.NECKFLIX.ABP_NORMALIZATION
                    processed_label = self.normalize_pressure_label(np.array(value), min, max)
                    abp = processed_label
            elif key == 'ECG':
                if config_data.PREPROCESS.LABEL_TYPE == "Raw":
                    ecg = np.array(value)
                elif config_data.PREPROCESS.LABEL_TYPE == "Standardized":
                    processed_label = self.standardized_label(np.array(value))
                    ecg = processed_label
                elif config_data.PREPROCESS.LABEL_TYPE == "DiffNormalized":
                    processed_label = self.diff_normalize_label(np.array(value))
                    ecg = processed_label
            elif key == 'POS_PPG':
                if config_data.PREPROCESS.LABEL_TYPE == "Raw":
                    pos_ppg = np.array(value)
                elif config_data.PREPROCESS.LABEL_TYPE == "Standardized":
                    processed_label = self.standardized_label(np.array(value))
                    pos_ppg = processed_label
                elif config_data.PREPROCESS.LABEL_TYPE == "DiffNormalized":
                    processed_label = self.diff_normalize_label(np.array(value))
                    pos_ppg = processed_label
            else:
                raise ValueError(f"Label {key} not recognized!")

        times = np.array(data_dict['TIMES'])
        processed_labels.append(times)
        for label in [ecg, cvp, abp,pos_ppg]:
            if label is not None:
                processed_labels.append(label)
        processed_labels = np.stack(processed_labels, axis=-1)
        return processed_labels

    def chunk(self, preprocessed_frames:List, labels:np.ndarray, chunk_length:int) -> Tuple[List, np.ndarray]:
        clip_num = labels.shape[0] // chunk_length
        label_clips = [labels[i * chunk_length:(i + 1) * chunk_length] for i in range(clip_num)]
        label_clips = np.array(label_clips)
        frame_clips = []
        for data in preprocessed_frames:
            clipped_frames = [data[i * chunk_length:(i + 1) * chunk_length] for i in range(clip_num)]
            frame_clips.append(np.array(clipped_frames))
        return frame_clips, label_clips

    def preprocess(self, data_dict, config_data):
        ###############      Process Frames      ###############
        preprocessed_frames = self.preprocess_frames(data_dict, config_data, config_data.PREPROCESS.USE_BIGSMALL)
        ###############      Process Labels      ###############
        preprocessed_labels = self.preprocess_labels(data_dict, config_data)
        ###############      Chunk Frames/Labels      ###############
        chunked_frames, chunked_labels = self.chunk(preprocessed_frames, preprocessed_labels, config_data.PREPROCESS.CHUNK_LENGTH)
        return chunked_frames, chunked_labels
    
    def save_multi_process(self, frame_chunks, label_chunks, filename):
        """Saves the preprocessing data."""
        if not os.path.exists(self.cached_path):
            os.makedirs(self.cached_path, exist_ok=True)
        count = 0
        input_path_name_list = []
        label_path_name_list = []

        for i in range(len(label_chunks)):
            input_path_name = os.path.join(self.cached_path,f"{filename}_input{count}.pickle")
            label_path_name = os.path.join(self.cached_path,f"{filename}_label{count}.npy")
            frames_dict = dict()
            for j in range(len(frame_chunks)):
                frames_dict[j] = frame_chunks[j][i]
                continue

            input_path_name_list.append(input_path_name)
            label_path_name_list.append(label_path_name)

            np.save(label_path_name, label_chunks[i]) # save out labels npy file
            with open(input_path_name, 'wb') as handle: # save out frame dict pickle file
                pickle.dump(frames_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            count += 1 # count of processed clips
            continue
        return count, input_path_name_list, label_path_name_list

    def preprocess_dataset_subprocess(self, data_dirs, config_data, i, file_list_dict):
        """ invoked by preprocess_dataset for multi_process """
        video_info_dict = data_dirs[i] # get data raw data file path 

        # # CONSTRUCT DATA DICTIONARY FOR VIDEO TRIAL
        data_dict = self.construct_data_dict(video_info_dict, config_data) # construct a dictionary of ALL labels and video frames (of equal length)
        if config_data.PREPROCESS.USE_PSUEDO_PPG_LABEL:
            rgb_frames = None
            # Only generate POS labels if rgb frames are available
            for frames in data_dict['FRAMES']:
                _,_,_,c = frames.shape
                if c == 3:
                    rgb_frames = frames
            if rgb_frames is not None:
                pos = self.generate_pos_psuedo_labels(rgb_frames, fs=self.config_data.FS)
                data_dict['POS_PPG'] = pos
            else:
                raise RuntimeWarning("No RGB frames: Pseudo PPG labels not generated")

        frame_chunks, label_chunks = self.preprocess(data_dict, self.config_data)
        saved_filename = video_info_dict['index']
        count, input_path_name_list, label_path_name_list = self.save_multi_process(frame_chunks, label_chunks, saved_filename)
        file_list_dict[i] = input_path_name_list

    def build_file_list_retroactive(self, data_dirs, begin, end):
        """ If a file list has not already been generated for a specific data split build a list of files 
        used by the dataloader for the data split. Eg. list of files used for 
        train / val / test. Also saves the list to a .csv file.

        Args:
            data_dirs(List[str]): a list of video_files.
            begin(float): index of begining during train/val split.
            end(float): index of ending during train/val split.
        Returns:
            None (this function does save a file-list .csv file to self.file_list_path)
        """

        # get data split based on begin and end indices.
        data_dirs_subset = self.split_by_participant(data_dirs, begin, end)

        # generate a list of unique raw-data file names
        filename_list = []
        for i in range(len(data_dirs_subset)):
            filename_list.append(data_dirs_subset[i]['index'])
        filename_list = list(set(filename_list))  # ensure all indexes are unique

        # generate a list of all preprocessed / chunked data files
        file_list = []
        for fname in filename_list:
            search_string = self.cached_path + os.sep + f"{fname}_input*.pickle"
            processed_file_data = list(glob.glob(search_string))
            file_list += processed_file_data

        if not file_list:
            raise ValueError(self.dataset_name,
                             'File list empty. Check preprocessed data folder exists and is not empty.')

        file_list_df = pd.DataFrame(file_list, columns=['input_files'])
        os.makedirs(os.path.dirname(self.file_list_path), exist_ok=True)
        file_list_df.to_csv(self.file_list_path)  # save file list to .csv

    def load(self):
        """ Loads the preprocessed data listed in the file list.

        Args:
            None
        Returns:
            None
        """
        file_list_path = self.file_list_path  # get list of files in
        file_list_df = pd.read_csv(file_list_path)
        inputs = file_list_df['input_files'].tolist()
        if not inputs:
            raise ValueError(self.dataset_name + ' dataset loading data error!')
        inputs = sorted(inputs)  # sort input file name list
        labels = [input_file.replace("input", "label").replace('.pickle', '.npy') for input_file in inputs]
        self.inputs = inputs
        self.labels = labels
        self.preprocessed_data_len = len(inputs)

    #### Video/Frame Functions ####

    @staticmethod
    def read_video(video_file:str) -> np.ndarray:
        """Reads a video file, returns frames(T,H,W,3) """
        VidObj = cv2.VideoCapture(video_file)
        VidObj.set(cv2.CAP_PROP_POS_MSEC, 0)
        success, frame = VidObj.read()
        frames = list()
        while success:
            frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)
            frame = np.asarray(frame)
            frames.append(frame)
            success, frame = VidObj.read()
        VidObj.release()
        return np.asarray(frames)

    @staticmethod
    def read_16bit_video(video_file:str) -> np.ndarray:
        """Reads a 16-bit grayscale FFV1 MKV file, returns frames (T, H, W, 1) as uint16"""
        container = av.open(video_file)
        video_stream = container.streams.video[0]
        frames = []
        num_frames = video_stream.frames
        for frame in container.decode(video_stream): # type: ignore
            # Convert to a NumPy array in 16-bit grayscale
            frame_np = frame.to_ndarray(format="gray16le")
            # PyAV returns grayscale frames with shape (H, W)
            frames.append(frame_np)

        # Return all frames as a single NumPy array of shape (T, H, W)
        frames = np.expand_dims(np.asarray(frames, dtype=np.uint16),axis=-1) # changes type and expands dimensions
        return frames
        
    @staticmethod
    def align_frames(all_frames:list, all_frame_times:list, tolerance=0.025) -> Tuple[List, np.ndarray]:
        """
        Align multiple video streams based on their frame timestamps.
        
        Parameters:
        all_frames: list of NumPy arrays, each of shape (T, H, W, C) (or (T,H,W,1) for grayscale)
        all_frame_times: list of 1D NumPy arrays with the corresponding frame timestamps.
        tolerance: maximum allowed difference (in seconds) between the frame times to be considered aligned.
        
        Returns:
        aligned_frames: list of NumPy arrays, each of shape (num_aligned_frames, H, W, C) (or similar)
        aligned_times: 1D NumPy array of the averaged timestamps for each aligned set.
        """
        num_streams = len(all_frames)
        # Initialize indices into each stream
        indices = [0] * num_streams
        # Lists to store the aligned frames from each stream
        aligned_frames = [[] for _ in range(num_streams)]
        aligned_times = []
        
        # Continue until at least one stream is exhausted.
        while all(indices[i] < len(all_frame_times[i]) for i in range(num_streams)):
            # Get the current timestamp from each stream.
            current_times = [all_frame_times[i][indices[i]] for i in range(num_streams)]
            t_min = min(current_times)
            t_max = max(current_times)
            
            # If the current timestamps are close enough, accept this set.
            if t_max - t_min <= tolerance:
                # You can choose to record the average or one of the times.
                aligned_times.append(np.mean(current_times))
                # Append the corresponding frames from each modality.
                for i in range(num_streams):
                    aligned_frames[i].append(all_frames[i][indices[i]])
                    indices[i] += 1  # Move to the next frame in each stream.
            else:
                # Otherwise, advance the stream that is the most behind.
                # Find the index (stream) with the minimum timestamp.
                min_index = current_times.index(t_min)
                indices[min_index] += 1
                
        # Convert lists to NumPy arrays.
        aligned_frames = [np.array(frames) for frames in aligned_frames]
        aligned_times = np.array(aligned_times)
        
        return aligned_frames, aligned_times
    
    @staticmethod
    def standardized_data(data:np.ndarray,exclude_mask:bool = True) -> np.ndarray:
        """Z-score standardization for video data."""
        if exclude_mask:
            selected_pixels = data[data != 0]
            mean_value = np.mean(selected_pixels)
            std_value = np.std(selected_pixels)
        else:
            mean_value = np.mean(data)
            std_value = np.std(data)
        data = (data - mean_value) / std_value
        data[np.isnan(data)] = 0
        return data
    
    @staticmethod
    def diff_normalize_data(data:np.ndarray,exclude_mask:bool=True,precision=np.float32) -> np.ndarray:
        """Calculate discrete difference in video data along the time-axis and normalize by its standard deviation."""
        if exclude_mask:
            data = data.astype(precision)
            data[data == 0] = np.nan
        n, h, w, c = data.shape
        diffnormalized_len = n - 1
        diffnormalized_data = np.zeros((diffnormalized_len, h, w, c), dtype=precision)
        diffnormalized_data_padding = np.zeros((1, h, w, c), dtype=precision)
        for j in range(diffnormalized_len):
            numerator = (data[j + 1, :, :, :] - data[j, :, :, :])
            denominator = (data[j + 1, :, :, :] + data[j, :, :, :] + 1e-7)
            diffnormalized_data[j, :, :, :] = numerator / denominator
        diffnormalized_data = diffnormalized_data / np.std(diffnormalized_data)
        diffnormalized_data = np.append(diffnormalized_data, diffnormalized_data_padding, axis=0)
        diffnormalized_data[np.isnan(diffnormalized_data)] = 0
        return diffnormalized_data
    
    @staticmethod
    def resize_frames(data: np.ndarray, new_height: int, new_width: int) -> np.ndarray:
        """
        Resize video frames to a new height and width using OpenCV.
        
        Parameters:
        data (np.ndarray): Video frames array of shape (N, H, W, C).
        new_height (int): The desired height.
        new_width (int): The desired width.
        
        Returns:
        np.ndarray: Resized video frames of shape (N, new_height, new_width, C) with the same dtype as the input.
        """
        N, H, W, C = data.shape
        resized_data = np.empty((N, new_height, new_width, C), dtype=data.dtype)
        for i in range(N):
            # cv2.resize expects the size in (width, height) order.
            frame_resized = cv2.resize(data[i], (new_width, new_height), interpolation=cv2.INTER_AREA)
            if C == 1 and frame_resized.ndim == 2:  # For grayscale images, cv2.resize returns a 2D array. Add a channel axis if needed.
                frame_resized = frame_resized[..., np.newaxis]
            resized_data[i] = frame_resized
        return resized_data
    
    #### Wave/Trace Functions ####
    @staticmethod
    def lowpass_filter(data: np.ndarray, fs:int, cutoff:float=10, order:int=4) -> np.ndarray:
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        filtered_data = filtfilt(b, a, data)
        return filtered_data
    
    @staticmethod
    def normalize_pressure_label(label:np.ndarray,min_val:float,max_val:float)->np.ndarray:
        """Normalize the pressure signal to a range of [-1,1]."""
        label = (label - min_val) / (max_val - min_val) * 2 - 1
        return label
     
    def adjust_data_dirs(self,data_dirs:List[dict]):
        """ Reads data folder and only preprocess files that have not already been preprocessed."""
        cached_path = self.cached_path
        file_list = glob.glob(os.path.join(cached_path, '*label*.npy'))
        trial_list = ['_'.join(os.path.split(f)[-1].split('_')[:-1]) for f in file_list]
        trial_list = list(set(trial_list))
        adjusted_data_dirs = []
        for d in data_dirs:
            idx = d['index']
            if not idx in trial_list: # if trial has already been processed
                adjusted_data_dirs.append(d)
        return adjusted_data_dirs