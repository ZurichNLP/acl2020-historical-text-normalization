import re
import functools
from abc import ABC, abstractmethod
from collections import defaultdict, Counter
from typing import Set, List, Iterable, Any, Union, Dict, Pattern
from pathlib import Path
import csv

import editdistance

from contextualized_transduction.charspacer import CharSpacer
from contextualized_transduction.utils import read_counts, ENCODING, NT_DEFAULT_FEATURE, UNK


class CandidateGenerator(ABC):
    """
    This generates graphically close candidates for input words. Subclass this if you need to implement some crude
    filtering of candidates that Chanspacer returns.
    """

    @abstractmethod
    def generate(self, word: Any, num_candidates: int, **kwargs) -> Iterable:
        """
        Generate graphically close candidates for a source word.
        :param word: Word.
        :param num_candidates: Number of candidates to generate.
        :return: Candidates.
        """
        pass

    def generate_candidate_state_space(self, words: Iterable[Any], num_candidates: int) -> List[List[Any]]:
        """
        Generate full state space:
          [
            [word_1_candidate_1, word_1_candidate_2, ... , word_1_candidate_`num_candidates`],
            [word_2_candidate_1, word_2_candidate_2, ... , word_2_candidate_`num_candidates`],
            .... ,
            [word_`len(words)`_candidate_1, _`len(words)`candidate_2, ... , _`len(words)`candidate_`num_candidates`]
          ]
        This enumeration is necessary for training with Markov language models (we then would compute
        forward and backward probabilities over this space) and decoding with them using Viterbi.

        Note that you might need to add beginning-of-sequence and / or end-of-sequence symbols to the output of this.
        :param words: Source words.
        :param num_candidates: Number of candidates to generate.
        :return: Full candidate state space for the given sequence of source words.
        """
        candidate_states: List[List[Any]] = []
        for word in words:
            candidate_states.append(list(self.generate(word, num_candidates)))
        return candidate_states

    def compute_coverage(self, filename: str, num_candidates: int, channel_order: bool = True,
                         output_not_found: bool = False, **kwargs) -> None:
        """
        Compute coverage of TARGET words for a tab-separated development set file.
        :param filename: Path to the file for which we compute the coverage of its TARGET words.
        :param num_candidates: How many candidates to generate from a SOURCE word to see if its TARGET is among them.
        :param channel_order: If True, the format is

                SOURCE word <tab> TARGET word <tab> rest...

            If False, the format is

                TARGET word <tab> SOURCE word <tab> rest...
        :param output_not_found: Whether to output to stdout the SOURCE words for which the TARGET is not among the
            candidates.
        @TODO word is not Any.
        """
        target_not_generated_for = set()
        correct = 0
        total = 0
        with open(filename, "r", encoding=ENCODING) as w:
            for line in w:
                line = line.rstrip()
                total += 1
                if line:
                    if channel_order:
                        # MODERN WORDS <tab> HISTORICAL WORD, we care about the coverage of MODERN words (=TARGET words)
                        target, word, *rest = line.split('\t')
                    else:
                        # HISTORICAL WORDS <tab> MODERN WORD
                        word, target, *rest = line.split('\t')
                    if target in self.generate(word, num_candidates, **kwargs):
                        correct += 1
                    else:
                        target_not_generated_for.add(word)

        print('Coverage:\t%.3f' % (100 * correct / total))
        if output_not_found:
            print('Target not generated for: ')
            for word in target_not_generated_for:
                print(word)


class CrudeCandidateGenerator(CandidateGenerator):

    def __init__(self, language: str, candidates_alphabet: Set[str],
                 indomain_candidates: Union[str, List, None],
                 crudefilter_regex_candidate: Union[str, Pattern], crudefilter_maxlen: int,
                 crudefilter_maxedit: int, lowercase: bool, verbose: bool,
                 crudefilter_regex_word: Union[str, Pattern] = re.compile(r'^[^\d.,!?]'),
                 *args, **kwargs):
        """
        Candidate generator that implements rule-based filters for words and candidates.
        :param language: Language.
        :param candidates_alphabet: Reasonable alphabet over which candidates can be built.
        :param indomain_candidates: In-domain candidate list.
        :param crudefilter_regex_candidate: Regex restricting the shape of candidates.
        :param crudefilter_maxlen: Maximum permissible absolute length difference between a target and a candidate.
        :param crudefilter_maxedit: Maximum permissible edit distance between a target and a candidate.
        :param crudefilter_regex_word: Regex to e.g. never correct numbers or urls.
        :param lowercase: Using a lowercase language model?
        :param verbose: Verbose?
        """
        self.language = language
        self.candidates_alphabet = candidates_alphabet
        self.indomain_candidates = indomain_candidates
        self.crudefilter_regex_word = re.compile(crudefilter_regex_word)
        self.crudefilter_regex_candidate = re.compile(crudefilter_regex_candidate)
        self.crudefilter_maxlen = crudefilter_maxlen
        self.crudefilter_maxedit = crudefilter_maxedit
        self.lowercase = lowercase
        self.verbose = verbose

        print('Candidate generator produces {}case candidates.'.format('lower' if self.lowercase else 'mixed '))

        if self.indomain_candidates:
            if isinstance(self.indomain_candidates, (str, Path)):
                raise NotImplementedError('Do not support custom vocabularies from file yet.')
            elif isinstance(self.indomain_candidates, List):
                # a list of words and their ** obligatory ** candidates
                self.obligatory_dict = defaultdict(set)
                for word, candidate in self.indomain_candidates:
                    self.obligatory_dict[word].add(candidate)
                self.obligatory_dict: Dict[str, Set[str]] = dict(self.obligatory_dict)
                print('Loaded an obligatory dictionary of candidates for candidate generation '
                      f'(size {len(self.obligatory_dict)}): ', list(self.obligatory_dict.items())[:10], '...')
        else:
            self.obligatory_dict = None

        self.get_candidates = self._get_candidates

    @abstractmethod
    def _get_candidates(self, word: str, num_candidates: int):
        pass

    @functools.lru_cache(maxsize=2 ** 14)
    def generate(self, word: str, num_candidates: int, add_word: bool = False) -> List[str]:
        cands = {word} if add_word else set()
        if self.obligatory_dict and word in self.obligatory_dict:
            # returns obligatory candidates and decrease the number of candidates to generate
            obligatory_cands = self.obligatory_dict[word]
            num_candidates -= len(obligatory_cands)
            cands.update(obligatory_cands)
        if self.crudefilter_regex_word.search(word):
            len_word = len(word)
            # @TODO Return exactly ** num_candidates ** candidates that pass the crude filter ?
            for c in self.get_candidates(word, num_candidates):
                if (abs(len(c) - len_word) <= self.crudefilter_maxlen and
                        all(ch in self.candidates_alphabet for ch in c) and
                        self.crudefilter_regex_candidate.search(c) and
                        editdistance.eval(c, word) <= self.crudefilter_maxedit):
                    cands.add(c)
        elif self.verbose:
            print(f'** WARNING: Not generating any candidates for "{word}": '
                  f'Doesn\'t match word regex: {self.crudefilter_regex_word.pattern}')
        if not cands:
            return [UNK]
        return list(cands)


class CrudeCharNGramGenerator(CrudeCandidateGenerator):

    def __init__(self, spacer_model_fn: str, outdomain_candidates_fn: str,
                 outdomain_cutoff: int, subsample: float, *args, **kwargs):
        """
        Wrapper around Charspacer that implements candidate generation with all kinds of rule-based filters.
        :param spacer_model_fn: Pickle filename to load spacer from.
        :param outdomain_candidates_fn: Out-domain candidate list (e.g. Google unigrams).
        :param outdomain_cutoff: Minimum word type count that a candidate must have.
        :param subsample: How many char n-grams to consider in candidate retrieval? (Charspacer param)
        """
        super().__init__(*args, **kwargs)

        self.spacer_model_fn = spacer_model_fn
        self.outdomain_candidates_fn = outdomain_candidates_fn
        self.outdomain_cutoff = outdomain_cutoff
        self.subsample = subsample

        # @TODO limit words to those from `candidates_alphabet`, train a corresponding Charspacer model
        if self.lowercase:
            # @TODO a quick workaround for lowercasing with the same unigram set
            WORDSgeqCUTOFF = sorted({w.lower() for w, c in read_counts(self.outdomain_candidates_fn)
                                     if c > self.outdomain_cutoff})
        else:
            WORDSgeqCUTOFF = (w for w, c in read_counts(self.outdomain_candidates_fn) if c > self.outdomain_cutoff)

        self.VECTOR_SPACER = CharSpacer.from_pickle(spacer_model_fn, words=WORDSgeqCUTOFF)

    def _get_candidates(self, word: str, num_candidates: int):
        return self.VECTOR_SPACER.candidates(word, num_candidates, s=self.subsample)


class CrudeMEDSGenerator(CrudeCandidateGenerator):

    def __init__(self, meds_fns: List[Union[str, Path]],  word_separator: str = '÷', *args, **kwargs):
        """
        Wrapper for offline MEDS.
        :param meds_fns: A list of paths to meds files (e.g. one for train, one for dev, etc.).
        :param word_separator: A separator character that might be used to replace whitespace in multi-token words.
        """
        super().__init__(*args, **kwargs)

        self.meds_fns = meds_fns
        self.word_separator = word_separator
        self.big_table = defaultdict(list)
        seen = defaultdict(set)
        for meds_fn in meds_fns:
            with Path(meds_fn).open(encoding=ENCODING) as f:
                for l, line in enumerate(f):
                    try:
                        word, candidate, *_rest = line.rstrip().split('\t')
                    except:
                        print(f'Ignored not well-formed candidate line {line}')
                        continue
                    word = word.replace(self.word_separator, ' ')
                    candidate = candidate.replace(self.word_separator, ' ')
                    if candidate in seen[word]:
                        continue
                    self.big_table[word].append(candidate)
                    seen[word].add(candidate)
                print(f'Loaded MEDS tsv file "{meds_fn}" with {l} word-candidate pairs ...')
        self.big_table: Dict[str, List[str]] = dict(self.big_table)
        print(f'Loaded MEDS for {len(self.big_table)} source words. The most common number of candidates (count): ',
              Counter(len(v) for v in self.big_table.values()).most_common(1)[0])

    def _get_candidates(self, word: str, num_candidates: int):
        return self.big_table.get(word.replace(self.word_separator, ' '), [])[:num_candidates]


class CandidateGeneratorWriter(CandidateGenerator):

    def __init__(self, generated_cadidates_fn: str, candidate_generator: CandidateGenerator):
        """
        Supports writing generated candidates to file e.g. for off-line training of neural transducer (NT).
        :param generated_cadidates_fn: File to which to write candidates.
        """
        self.generated_cadidates_fn = generated_cadidates_fn
        self.candidate_generator = candidate_generator

    @functools.lru_cache(maxsize=2 ** 14)
    def generate(self, word: Any, num_candidates: int, **kwargs) -> Iterable:
        """
        In addition to candidate generation, write candidates to file for NT channel training.
        :param word: Word.
        :param num_candidates: Number of candidates to generate.
        :return: Candidates.
        """
        cands = self.candidate_generator.generate(word, num_candidates, **kwargs)
        if cands == [UNK]:
            word = UNK
        with open(self.generated_cadidates_fn, 'a', encoding=ENCODING) as w:
            csv.writer(w, delimiter='\t', lineterminator='\n', quotechar='', quoting=csv.QUOTE_NONE).writerows(
                (cand, word, NT_DEFAULT_FEATURE) for cand in cands)
        return cands
