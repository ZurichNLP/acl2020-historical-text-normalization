from collections import Counter
from pathlib import Path
import json
from typing import Any, List, Dict, Iterable, Union, TextIO, Optional

class Vocab:
    def __init__(self):
        self.idx2word: List[Any] = []
        self.word2idx: Dict[Any, int] = dict()

    def __getitem__(self, word: Any) -> int:
        # get index for a word
        if word in self.word2idx:
            idx = self.word2idx[word]
        else:
            idx = len(self.idx2word)
            self.word2idx[word] = idx
            self.idx2word.append(word)
        return idx

    def __iter__(self):
        return iter(self.idx2word)

    def get_idx_for_word(self, word: Any) -> int:
        return self.__getitem__(word)

    def get_word_for_idx(self, idx: int) -> Any:
        return self.idx2word[idx]

    def __len__(self):
        return len(self.idx2word)

    def __contains__(self, word: str):
        assert isinstance(word, str)
        return word in self.word2idx

    def __repr__(self):
        return '%r' % self.idx2word


CHAR_UPDATE = ['<UNK>', '⟪', '⟫']
FEAT_UPDATE = ['<UNK>']
ACT_UPDATE = ['<UNK>', '⟪', '⟫', '<DEL>', '<COPY>']


class VocabSet:
    def __init__(self, counter: Optional[Counter] = None):
        self.counter = Counter(counter) if counter else Counter()

    def add(self, item: Any):
        self.__update((item,))

    def update(self, other: Union[Iterable, "VocabSet"]):
        if hasattr(other, 'counter'):
            other = other.counter
        self.__update(other)

    def __update(self, other):
        self.counter.update(other)

    def __contains__(self, item: Any):
        return item in self.counter

    @property
    def set(self):
        return tuple(self.counter.keys())

    def serialize(self, update: List[str] = None) -> dict:
        """
        Create a data structure loadable by the neural transducer code as vocab.
        :param update: Add vocab-specific entities such as actions. Their codes are assumed to be
            0, ..., len(`update`) - 1.
        :return: Serializable representation of `self`.
        """
        w2i = {a: h for h, a in enumerate((update if update else []) + list(self.set))}
        i2w_freq = {h: self.counter.get(key, 0) for key, h in w2i.items()}
        return dict(w2i=w2i, encoding=None, freqs=i2w_freq)

    @classmethod
    def from_json(cls, fileobj: TextIO):
        tmp_dict = json.load(fileobj)
        counter = Counter(tmp_dict['freqs'])
        return cls(counter)

    def __repr__(self):
        return self.set.__repr__()


def write_vocabsets(path: Union[Path, str], act: VocabSet, feat: VocabSet, word: VocabSet,
                    char: Optional[VocabSet] = None, pos: Optional[VocabSet] = None,
                    feat_type: Optional[VocabSet] = None,
                    pos_emb: bool = False, avm_feat_format: bool = False, param_tying: bool = True,
                    encoding: str = 'utf8') -> None:
    """
    Write to file a vocabulary-box like object for loading into the neural transducer. One use case:
    load a vocabulary precomputed for the labeled and unlabeled datasets but train only on the labeled
    dataset only, which would also for reloading the pretrained transducer model and continuing training
    with all the data.
    :param path: Path to write to.
    :param act: Actions.
    :param feat: Features.
    :param word: Target-side words. NB in channel order, this is historical / corrupted words.
    :param char: Source-side characters.
    :param pos: POS features.
    :param feat_type: AVM feature types (?).
    :param pos_emb: Whether POS features are separate from the other features.
    :param avm_feat_format: Whether to use AMV feature format.
    :param param_tying: Whether actions and characters are from the same set.
    :param encoding: Encoding for the json file.
    """
    # @TODO move encoding to constants from utils
    d = dict(pos_emb=pos_emb, avm_feat_format=avm_feat_format, param_tying=param_tying, encoding=None,
             act=act.serialize(ACT_UPDATE), feat=feat.serialize(FEAT_UPDATE), word=word.serialize(),
             char=char, pos=pos, feat_type=feat_type, w2i_acts={a: h for h, a in enumerate(ACT_UPDATE)})
    assert not pos_emb or pos, (pos_emb, pos)
    assert not avm_feat_format or feat_type, (avm_feat_format, feat_type)
    assert param_tying or char, (param_tying, char)

    with Path(path).open(mode='w', encoding=encoding) as w:
        json.dump(d, w, indent=4, ensure_ascii=False)


class SentSample:
    def __init__(self, original_words: List[str], refs: List[str], words: Optional[List[str]] = None):
        """
        Holds a single sentence / segment of a dataset, comprising original sequence of words, possibly modified input
        sequence of words (e.g. partially UNK-ed), and the sequence of targets.
        :param original_words: Original, unchanged sequence of input words. Useful for decoding when some
            `words` are UNK-ed.
        :param words: Possibly modified sequence of input words (e.g. some words might be replaced by an UNK).
        :param refs: Sequence of target words.
        """
        self.original_words = original_words
        self.words = words if words else list(original_words)
        self.refs = refs

    def reset_word(self, word_id: int, reset: str):
        tmp = self.words[word_id]
        self.words[word_id] = reset
        assert tmp == reset or self.original_words[word_id] != reset, (tmp, reset, self.original_words[word_id])

class Dataset:
    def __init__(self, sents: List[SentSample], dataset_name: Optional[str] = None):
        """
        Holds a sequence of sentences.
        :param sents: Sentences.
        :param dataset_name: Dataset name.
        """
        self.sents = sents
        self.dataset_name = dataset_name

    def __len__(self):
        return len(self.sents)

    def __iter__(self):
        for s in self.sents:
            yield s.original_words, s.words, s.refs

    def __getitem__(self, sliced):
        return self.sents[sliced]
