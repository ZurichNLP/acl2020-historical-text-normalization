import csv
import gzip
import os
import sys
import time
import string
from collections import namedtuple
from pathlib import Path
from typing import Generator, Union, Tuple, Callable, List, Optional

from contextualized_transduction.vocabulary import VocabSet, Dataset, SentSample

import logging
log = logging.getLogger(__name__)

import numpy as np
import scipy.special


ENCODING = 'utf8'
DIRNAME = os.path.dirname(os.path.realpath(__file__))
DATADIR = os.path.join(DIRNAME, 'data')

LARGE_NEG_CONST = -float(10 ** 6)
LARGE_POS_CONST = 10 ** 8
TOL = 10 ** -10

EOS = '<#>'  # end of sequence symbol
ASCII_ALPHABET = string.ascii_letters + string.digits + string.punctuation
GERMAN_ALPHABET = string.ascii_letters + string.digits + string.punctuation + 'üäöÄÜÖß'
NT_DEFAULT_FEATURE = '#'
UNK = '࿋'  # '<unk>'

ComponentConfig = namedtuple('ComponentConfig', 'model params')


def expand_path(fn: Union[str, Path]):
    if str(fn).startswith("."):
        return fn
    else:
        return os.path.join(DATADIR, fn)


def read_counts(gz: str, dir_=DATADIR, encoding=ENCODING, gzipped=True, unprocessed=False, delimiter='\t') -> Generator[
    Union[str, Tuple[str, int]], None, None]:
    """
    Read n-gram count files.
    :param gz: "Gzip"-compressed file.
    :param dir_: Directory of the file.
    :param encoding: File encoding.
    :param gzipped: Whether file is gzipped.
    :param unprocessed: Whether to yield raw lines or word/count tuple.
    :param delimiter: In the case of yielding words/counts, which column delimiter to use.
    :return: Generator object.
    """
    opener = gzip.open if gzipped else open
    with opener((os.path.join(dir_, gz)), mode='rt', encoding=encoding) as f:
        if unprocessed:
            for line in f:
                yield line
        else:
            for word, count in csv.reader(f, delimiter=delimiter, quoting=csv.QUOTE_NONE):
                yield word, int(count)


def read_dataset(dataset_fn: str, dataset_name: Optional[str] = None, dir_: str = DATADIR,
                 data_cutoff: int = LARGE_POS_CONST, max_seg_len: int = LARGE_POS_CONST,
                 encoding: str = ENCODING) -> Tuple[Dataset, VocabSet, VocabSet, float]:
    """
    Read in a dataset in a fixed format. Data comes in a two-column tab-separated format. Segments / sentences are
    separated by empty lines. This also computes a COPY baseline.
    :param dataset_fn: Dataset file name.
    :dataset_name: Dataset name.
    :param dir_: File directory.
    :param data_cutoff: Return at most this amount of segments.
    :param max_seg_len: Segments must be at most this length.
    :param encoding: Encoding of the dataset file.
    :return: The dataset (a sequence of pairs of sequences of source and (possibly) target words), the set of source
        word characters, the set of target word characters, accuracy score of the COPY baseline.
    """
    DATASET = []
    source_segment = []
    target_segment = []
    SOURCE_CHARSET = VocabSet()
    TARGET_CHARSET = VocabSet()

    assert max_seg_len > 0

    if data_cutoff == 0:
        return Dataset(DATASET, dataset_name), SOURCE_CHARSET, TARGET_CHARSET, 1.

    # COMPUTE COPY BASELINE
    total = 0
    correct = 0
    seg_len = 0
    with open(os.path.join(dir_, dataset_fn), encoding=encoding) as f:
        for row in csv.reader(f, delimiter='\t', lineterminator='\n', quotechar='', quoting=csv.QUOTE_NONE):
            try:
                source, target, *_ = row
                if not (source or target) or seg_len == max_seg_len:
                    if len(DATASET) == data_cutoff - 1:
                        break
                    if not (source_segment or target_segment):
                        print('Skipping empty segment...')
                        continue
                    DATASET.append(SentSample(source_segment, target_segment))
                    source_segment = []
                    target_segment = []
                    seg_len = 0
                source_segment.append(source)
                target_segment.append(target)
                correct += (source == target)
                total += 1
                seg_len += 1
                SOURCE_CHARSET.update(source)
                TARGET_CHARSET.update(target)
            except ValueError:
                assert not row, row
                if len(DATASET) == data_cutoff - 1:
                    break
                if not (source_segment or target_segment):
                    print('Skipping empty segment...')
                    continue
                DATASET.append(SentSample(source_segment, target_segment))
                source_segment = []
                target_segment = []
                seg_len = 0
    if source_segment:
        DATASET.append(SentSample(source_segment, target_segment))
    copy_baseline = correct * 100 / total
    return Dataset(DATASET, dataset_name), SOURCE_CHARSET, TARGET_CHARSET, copy_baseline


def simple_preprocessor(dataset_fn: str, dir_: str = DATADIR,
                        tokenize: Callable[[str], List[str]] = lambda w: w.split(' '),
                        encoding=ENCODING) -> Generator[Tuple[List[str], List[str]], None, None]:
    """
    Read in tab-separated input (source word sequence, target word sequence) and tokenize them into words.
    :param dataset_fn: Dataset filename.
    :param dir_: File directory.
    :param tokenize: Tokenizer. By default, split on whitespace.
    :param encoding: Encoding.
    """
    with open(os.path.join(dir_, dataset_fn), encoding=encoding) as f:
        for source_segment, target_segment in csv.reader(f, delimiter='\t'):
            yield tokenize(source_segment), tokenize(target_segment)



# for neural transducer channel trained with IL
DEFAULT_PARAMS = {
    "--transducer": "haem",
    "--vanilla-lstm": False,
    "--sigm2017format": True,
    "--no-feat-format": True,  # False,
    "--pos-emb": False,  # True
    "--avm-feat-format": False,
    "--mlp": 0,
    "--nonlin": "ReLU",
    "--lucky-w": 55,
    "--tag-wraps": "both",
    "--align-dumb": True,
    "--mode": "il",
    "--try-reverse": False,
    "--iterations": 0,
    "--beam-width": 0,
    "--beam-widths": None,
    "--dropout": 0,
    "--pretrain-dropout": False,
    "--optimization": None,
    "--l2": 0,
    "--alpha": 0,
    "--beta": 0,
    "--no-baseline": False,
    "--epochs": 0,
    "--patience": 0,
    "--pick-loss": False,
    "--pretrain-epochs": 0,
    "--pretrain-until": 0,
    "--batch-size": 0,
    "--decbatch-size": 0,
    "--sample-size": 0,
    "--scale-negative": 0,
    "--il-decay": 0,
    "--il-k": 0,
    "--il-tforcing-epochs": 0,
    "--il-loss": "nll",
    "--il-bias-inserts": False,
    "--il-beta": 1,
    "--il-global-rollout": False,
    "--il-optimal-oracle": True,
    "--test-path": '',
    "--reload-path": ''
}

def logsumexp10_(a):
    a = np.asarray(a)
    a_max = a.max(axis=0)
    out = np.log10(np.sum(np.power(10., a - a_max), axis=0))
    out += a_max
    return out


def logsumexp10(a, axis=None):
    """Log sum exp for base 10 logarithms."""
    return scipy.special.logsumexp(np.asarray(a) / np.log10(np.e), axis=axis) / np.log(10)


def scheduler_queue(lockfile_name: str = "SCHEDULER_LOCKFILE" , sleep_seconds: int = 10) -> None:
    """
    Return as soon as we get a multiprocessing lock file
    """
    while True:
        if os.path.exists(f"/tmp/{lockfile_name}"):
            log.critical(f'Multiprocessing schedule file {lockfile_name} exists... waiting for {sleep_seconds}')
            time.sleep(sleep_seconds)
        else:
            try:
                with open(f"/tmp/{lockfile_name}",mode="w") as f:
                    print(sys.argv,file=f)
            except:
                continue
            log.critical(f'Multiprocessing schedule file {lockfile_name} created...')
            return


def scheduler_dequeue(lockfile_name: str = "SCHEDULER_LOCKFILE") -> None:
    """
    Release the lock file
    """
    try:
        log.critical(f'Releasing multiprocessing schedule file {lockfile_name} now...')
        os.unlink(f"/tmp/{lockfile_name}")
    except:
        pass
