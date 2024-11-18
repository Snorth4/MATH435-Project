"""Peptides Dataset."""

import tensorflow_datasets as tfds
import tensorflow as tf
import pandas as pd
import random

class Peptides(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for Peptides Dataset."""
    
    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {'1.0.0': 'Initial release.'}

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return tfds.core.DatasetInfo(
            builder=self,
            features=tfds.features.FeaturesDict({
                'ID': tfds.features.Text(doc='The id of the peptide.'),
                'sequence': tfds.features.Text(doc='The sequence of amino acids in the peptide.'),
                'sequence_length': tf.int32,
                'MIC': tf.float32,
                'is_hemolytic': tfds.features.ClassLabel(
                    names=['false', 'true'],
                    doc='Whether this peptide kills human cells or not.'
                ),
            }),
            supervised_keys=('MIC', 'sequence_length', 'sequence'),
            homepage='https://aps.unmc.edu/home',
        )
    
    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        # Download and extract the single CSV file
        path = dl_manager.download_and_extract('Peptide Dataset.csv')

        return {
            'train': self._generate_examples(path, split='train'),
            'val':   self._generate_examples(path, split='val'),
            'test':  self._generate_examples(path, split='test'),
        }

    def _generate_examples(self, path, split):
        """Yields examples based on the split."""
        # Load data from the CSV file
        df = pd.read_csv(path)
        
        randint = random.randint(0, 4294967295)

        # Shuffle the data for random splitting
        df = df.sample(frac=1, random_state=randint).reset_index(drop=True)
    
        # Calculate split indices
        train_end = int(0.70 * len(df))
        val_end = int(0.85 * len(df))
    
        if split == 'train':
            # Select the first 70% of data for training
            data = df[:train_end]
        elif split == 'val':
            # Select the next 15% for validation
            data = df[train_end:val_end]
        else:
            # Select the remaining 15% for testing
            data = df[val_end:]
    
        # Yield each row as an example
        for idx, row in data.iterrows():
            yield idx, {
                'ID'              : row['ID'],
                'sequence'        : row['sequence'],
                'sequence_length' : row['sequence_length'],
                'MIC'             : row['MIC'],
                'is_hemolytic'    : row['is_hemolytic']
            }