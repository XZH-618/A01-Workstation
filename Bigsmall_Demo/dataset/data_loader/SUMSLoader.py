import numpy as np
import pandas as pd
import cv2
import glob
import os
from scipy.interpolate import interp1d
from dataset.data_loader.BaseLoader import BaseLoader

class SUMSLoader(BaseLoader):
    def __init__(self, name, data_path, config_data, device=None):
        """Initializes a SUMS dataloader.
            Args:
                data_path (str): Path to a folder containing raw video and BVP data.
                For example, data_path should be "On-Road-rPPG" for the following structure:
                -----------------
                     On-Road-rPPG/
                    |   |-- 060200/
                    |       |-- v01
                    |           |-- BVP.csv
                    |           |-- frames_timestamp.csv
                    |           |-- HR.csv
                    |           |-- RR.csv
                    |           |-- video_ZIP_H264_face.avi
                    |           |-- video_ZIP_H264_finger.avi
                    |       |-- v02
                    |       |-- v03
                    |       |-- v04
                    |   |-- 060201/
                    |       |-- v01
                    |       |-- v02
                    |       |...
                    |...
                    |   |-- 0602mn/
                    |       |-- v01
                    |       |-- v02
                    |       |...
                -----------------
                name (str): Name of the dataloader.
                config_data (CfgNode): Data settings (ref: config.py).
        """
        self.info = config_data.INFO  
        print(data_path)
        super().__init__(name, data_path, config_data, device)

    def get_raw_data(self, data_path):
        """Returns data directories in the specified path (suitable for the THUSPO2 dataset)."""
        "Get all 060200, 060201... files; data_path needs to be changed"
        print(data_path)
        data_dirs = glob.glob(data_path + os.sep + '0602*')
        if not data_dirs:
            raise ValueError(self.dataset_name + ' Data path is empty!')
        dirs = list()
        # data_dirs absolute path
        for data_dir in data_dirs:
            
            subject = int(os.path.split(data_dir)[-1]) # File name directly 060200
            d_dirs = os.listdir(data_dir)
            print(d_dirs)
            # v01 v02 v03 v04
            for dir in d_dirs:    
                items_dirs = os.listdir(data_dir + os.sep + dir) # avi csv 
                for item in items_dirs:
                    if "avi" in item: # If returning all together here
                        dirs.append({'index': dir[1:],
                                    'path': data_dir + os.sep + dir + os.sep +item,
                            'subject': subject,
                            'type': item.split('_')[-1].split('.')[0]
                        })
                
        print(dirs)    
        return dirs
        

    def split_raw_data(self, data_dirs, begin, end):
        """Returns a subset of data dirs, split with begin and end values."""
        if begin == 0 and end == 1:  # return the full directory if begin == 0 and end == 1
            return data_dirs
        # Split according to tags v01, v02, v03, v04
        
        data_info = dict()
        for data in data_dirs:
            # index = data['index']
            # data_dir = data['path']
            subject = data['subject']
            # type = data['type'] # face or finger
            # Create a data directory dictionary indexed by subject number
            if subject not in data_info:
                data_info[subject] = list()
            data_info[subject].append(data)
        
        subj_list = list(data_info.keys())  # Get all subject numbers
        subj_list = sorted(subj_list)  # Sort subject numbers
        
        num_subjs = len(subj_list)  # Total number of subjects      
        
        # Get data set split (according to start/end ratio)
        subj_range = list(range(num_subjs))
        if begin != 0 or end != 1:
            subj_range = list(range(int(begin * num_subjs), int(end * num_subjs)))
        print('Subjects ID used for split:', [subj_list[i] for i in subj_range])

        # Add file paths that meet the split range to the new list
        data_dirs_new = list()
        for i in subj_range:
            subj_num = subj_list[i]
            data_dirs_new += data_info[subj_num]
        
        print(data_dirs_new)
        return data_dirs_new            


    def preprocess_dataset_subprocess(self, data_dirs, config_preprocess, i,  file_list_dict):
        
        # Read video frames
        video_file = data_dirs[i]['path']
        frames = self.read_video(video_file)

        # Get the directory of the current video
        video_dir = os.path.dirname(video_file)

        # Extract subject ID and experiment ID from the directory path
        subject_id = video_dir.split(os.sep)[-2]
        experiment_id = video_dir.split(os.sep)[-1]  # Assuming experiment ID follows subject ID
        print(f"subject_id: {subject_id}, experiment_id: {experiment_id}")
        # Get BVP, frame timestamps
        bvp_file = os.path.join(video_dir, "BVP.csv")
        timestamp_file = os.path.join(video_dir, "frames_timestamp.csv")

        # Read frame timestamps
        frame_timestamps = self.read_frame_timestamps(timestamp_file)

        # Read BVP data and timestamps
        bvp_timestamps, bvp_values = self.read_bvp(bvp_file)

        # Resample BVP data to match video frames
        resampled_bvp = self.synchronize_and_resample(bvp_timestamps, bvp_values, frame_timestamps)

        # Process frames, BVP signals, and SpO2 signals according to the configuration
        if config_preprocess.USE_PSUEDO_PPG_LABEL:
            bvps = self.generate_pos_psuedo_labels(frames, fs=self.config_data.FS)
        else:
            bvps = resampled_bvp

        # Label once here
        if "face" in video_file:
            frames_clips, bvps_clips = self.preprocess(frames, bvps, config_preprocess)
            filename = f"{subject_id}_{experiment_id}"
            input_name_list, label_name_list = self.save_multi_process(frames_clips, bvps_clips, filename)
            file_list_dict[i] = input_name_list
        

    def load_preprocessed_data(self):
        """Load preprocessed data listed in the file list."""

        file_list_path = self.file_list_path   # Get file list path
        file_list_df = pd.read_csv(file_list_path)  # Read file list
        inputs_temp = file_list_df['input_files'].tolist()  # Get input file list
        inputs_face = [] 
        
        # v01 v02 v03 v04 face configuration information
        for each_input in inputs_temp:
            inputs_face.append(each_input) 
       
        inputs_face = sorted(inputs_face)
        labels_bvp = [input_file.replace("input", "label") for input_file in inputs_face]
           
        self.inputs = inputs_face    
        self.labels = labels_bvp
        self.preprocessed_data_len = len(inputs_face)

    @staticmethod
    def read_bvp(bvp_file):
        """Reads a BVP signal file with timestamps."""
        data = pd.read_csv(bvp_file)
        timestamps = data['timestamp'].values
        bvp_values = data['bvp'].values
        return timestamps, bvp_values

    @staticmethod
    def read_frame_timestamps(timestamp_file):
        """Reads timestamps for each video frame."""
        data = pd.read_csv(timestamp_file)
        return data['timestamp'].values

    @staticmethod
    def synchronize_and_resample(timestamps_data, data_values, timestamps_frames):
        """Synchronize and resample data to match video frame timestamps."""
        interpolator = interp1d(timestamps_data, data_values, bounds_error=False, fill_value="extrapolate")
        resampled_data = interpolator(timestamps_frames)
        return resampled_data

    @staticmethod
    def read_video(video_file):
        """Reads a video file, returns frames."""
        VidObj = cv2.VideoCapture(video_file)
        VidObj.set(cv2.CAP_PROP_POS_MSEC, 0)
        success, frame = VidObj.read()
        frames = []
        while success:
            frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)
            frames.append(frame)
            success, frame = VidObj.read()
        return np.array(frames)