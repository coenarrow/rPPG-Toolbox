"""The dataloader for the Neckflix dataset.

Details for the Neckflix Dataset see ####
If you use this dataset, please cite this paper:
C. Arrow, M. Ward, J. Eshraghian, G. Dwivedi.
"Neckflix:"
"""
import numpy as np
from dataset.data_loader.BaseLoader import BaseLoader
from pathlib import Path

class NeckflixLoader(BaseLoader):
    """The data loader for the Neckflix dataset."""

    def __init__(self, name, data_path, config_data,device=None):
        """Initializes an Neckflix dataloader.
            Args:
        """
        self.inputs = list()
        self.labels = list()
        self.config_data = config_data
        self.cached_path = Path(self.config_data.CACHED_PATH)
        if self.cached_path.exists():
            self.load()
        else:
            raise ValueError("Neckflix Dataset must be preprocessed before loading")

    def __getitem__(self, index):
        if self.data_format == 'NDCHW':
            data = np.load(self.inputs[index]).transpose((0, 3, 1, 2)).astype(np.float32)
        elif self.data_format == 'NCDHW':
            data = np.load(self.inputs[index]).transpose((3, 0, 1, 2)).astype(np.float32)
        elif self.data_format == 'NDHWC':
            data = np.load(self.inputs[index])
        else:
            raise ValueError('Unsupported Data Format!')
        label = np.load(self.labels[index]).astype(np.float32)
        filename,chunk_id = Path(self.inputs[index]).stem.split('-input')
        return data, label, filename, chunk_id

    def load(self):
        """ Loads the preprocessed data listed in the file list.
        Args:
            None
        Returns:
            None
        """

        # Get the list of the participants based on the fold
        subject_path = Path(self.config_data.FOLD.FOLD_PATH)
        with open(subject_path, 'r') as f:
            participants = f.read().splitlines()

        # Get input paths, filtered by participants
        input_paths = [path for path in sorted(self.cached_path.glob("*input*")) if path.stem.split("_")[0] in participants]

        # load the inputs and labels
        inputs=[]
        labels=[]
        for input_path in input_paths:
            label_path = Path(input_path.as_posix().replace('input','label'))
            if not label_path.exists() and input_path.exists():
                continue
            inputs.append(input_path.as_posix())
            labels.append(label_path.as_posix())

        if len(inputs) == 0:
            raise ValueError("Could not load any inputs - check inputs exist")

        self.inputs = inputs
        self.labels = labels