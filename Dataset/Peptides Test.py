"""Peptides Dataset."""

import tensorflow_datasets as tfds
import Peptides
import util

class MyDatasetTest(tfds.testing.DatasetBuilderTestCase):
    """Tests for my_dataset dataset."""

    DATASET_CLASS = "Peptides.py"
    SPLITS = {
        'train': 14,    # Number of fake train examples.
        'val'  : 3,     # Number of fake valid examples.
        'test' : 3,     # Number of fake test examples.
    }
    DL_EXTRACT_RESULT = 'Peptides.csv'

if __name__ == '__main__':
    tfds.testing.test_main()