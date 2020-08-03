import abc
from typing import Sequence, Tuple,  Any, Optional

import numpy as np

import kenlm

from contextualized_transduction.utils import expand_path



class LM(abc.ABC):
    """
    Language model decoder.
    """

    def __init__(self, order: Optional[int] = None):
        """
        :param order: Order of LM. "None" means LM makes no Markov assumption.
        """
        self.order = order
        super(LM, self).__init__()

    @abc.abstractmethod
    def initial_state(self):
        """
        Initial representation of context.
        :return:
        """
        pass

    @abc.abstractmethod
    def score(self, word: str, prefix: Sequence[str], state: Any) -> Tuple[Any, float]:
        """
        Given a word and some representation of the previous context, `state`, score the full sequence.
        :param word: Continuation of the sequence.
        :param prefix: Previous context.
        :param state: Object representing the previous context.
        :return: Updated state and log probability of the resulting sequence, possibly unnormalized.
        """
        pass

    @abc.abstractmethod
    def __contains__(self, word: str):
        pass



class KenLM(LM):

    def __init__(self, apra_fn: str, log10=True, char_lm_backoff: Optional[str] = None, lowercase=True, verbose=False,
                 config=None, *args, **kwargs):
        """
        Wrapper for stateful decoding with KenLM.
        :param apra_fn: Input text file in apra format or binary file in kenlm format that holds LM parameters.
        :param log10: Are the parameters in log10?
        :param char_lm_backoff: Input text file in apra format or binary file in kenlm format that holds character-level
            LM parameters for backoff on UNK tokens.
        :param lowercase: Does the model (and the character backoff model) assume lowercased input?
        :param verbose: Verbose.
        :param config: KenLM config (see KenLM documentation). @TODO not used at the moment.
        """
        self.model = kenlm.Model(expand_path(apra_fn))
        self.lowercase = lowercase
        self.verbose = verbose
        self.rebased = np.log10(np.e) if log10 else 1.
        super(KenLM, self).__init__(order=self.model.order)
        print(
            f'Loaded {self.model.order}-gram kenLM model.',
            '**NB** This contains only lower-cased word types!' if self.lowercase else '*NB* LM not lowercased!')
        if char_lm_backoff:
            # @TODO should not necessarily be kenLM
            self.char_model = kenlm.Model(expand_path(char_lm_backoff))
            print(
                f'Loaded {self.char_model.order}-gram kenLM character-level model for ad-hoc <UNK> back-off.'
                ' **NB** This contains only lower-cased word types!')
        else:
            self.char_model = None

    def initial_state(self) -> kenlm.State:
        # this add the beginning-of-sentence symbol
        init_state = kenlm.State()
        self.model.BeginSentenceWrite(init_state)  # declare state as initial
        return init_state

    def score(self, word: str, prefix: Sequence[str], state: kenlm.State) -> Tuple[kenlm.State, float]:
        word = word.lower() if self.lowercase else word
        out_state = kenlm.State()
        score = self.model.BaseScore(state, word, out_state)
        if word not in self.model:
            if self.char_model:
                # stand-alone char seq evaluation ? use prefix ?
                # https://github.com/scfrank/de_charlm/blob/master/query_lm.py
                # @TODO add lowercase flag for back off
                score += self.char_model.score(' '.join(list(word.lower())), bos=False, eos=False)
            if self.verbose:
                print('Candidate "%s" not in LM.' % word)
        return out_state, score / self.rebased

    def __contains__(self, word: str):
        # can it give any non-UNK score to word?
        return word in self.model
