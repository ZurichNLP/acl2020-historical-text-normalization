from collections import namedtuple
from dataclasses import dataclass
import numpy as np
from typing import Sequence, List, Any, Optional, Tuple, Union

from contextualized_transduction.lm import KenLM
from contextualized_transduction.utils import ComponentConfig, UNK
from contextualized_transduction.exact_model import Model, ExactModel, DecoderResult

from nn_lm.custom_lm import CharLanguageModel, WordLanguageModel, CustomLanguageModel


class InexactModel(Model):

    def __init__(self, language_model_config: ComponentConfig, channel_config: ComponentConfig,
                 candidate_generator_config: ComponentConfig, num_candidates: int, kbest: Optional[int] = None,
                 *args, **kwargs):
        """
        A model that performs inexact decoding. The set of permissible hypotheses are given by the candidate generator,
        which generates candidates for each source word.
        :param language_model_config: Language model class and a configuration dictionary.
        :param channel_config: Channel model class and a configuration dictionary.
        :param candidate_generator_config: Candidate generator model class and a configuration dictionary.
        :param num_candidates: Default number of candidates to generate using candidate generator. Used for training.
        :param kbest: Default number of predictions. Used for training.
        """

        self.candidate_generator_class, self.candidate_generator_params = candidate_generator_config
        self.candidate_generator = self.candidate_generator_class(**self.candidate_generator_params)

        super().__init__(language_model_config=language_model_config, channel_config=channel_config)

        self.num_candidates = num_candidates
        self.kbest = kbest  # when kbest is not passed to decode method, do we collect weights for top k predictions?

    def _compute_weights(self, paths: List[List[str]], path_scores: Union[List[float], np.ndarray],
                        source_words: Sequence[str]) -> Tuple[np.ndarray, List[List[str]]]:
        """
        Compute posterior weights and candidates from decoding paths (hard EM):
         * weights are a 2D matrix: One p(c, w, t) weight for each candidate position (candidate_number, timestep).
         * candidates is a list of lists of candidates per timestep.
        :param paths: Paths from decoding (k-best, Viterbi, beam, etc.)
        :param path_scores: Scores for each path in `paths`.
        :param source_words: Source words.
        :return: normalized weights and candidates.
        """
        weights = np.zeros((len(paths), len(source_words)))  # there can be at most len(paths) candidates if at this
        # position, each path has a different candidate
        candidates = []
        for t, word in enumerate(source_words):
            candidates_at_t = dict()
            candidates_at_t_ = []
            candidate_count = 0
            for path, path_weight in zip(paths, path_scores):
                candidate = path[t]
                if candidate in candidates_at_t:
                    # get candidate's id
                    k = candidates_at_t[candidate]
                else:
                    # add candidate and increment candidate count
                    k = candidate_count
                    candidates_at_t[candidate] = candidate_count
                    candidates_at_t_.append(candidate)
                    candidate_count += 1
                weights[k, t] += path_weight
            candidates.append(candidates_at_t_)
        d = weights.sum(axis=0)
        weights = weights / d
        return weights, candidates

    def _compute_weights_single_path(self, path: List[str]) -> Tuple[np.ndarray, List[List[str]]]:
        """
        Return posterior weights and candidates from a single decoding path (hard EM):
         * weights are a 2D matrix: One p(c, w, t) weight for each candidate position (candidate_number, timestep).
         * candidates is a list of lists of candidates per timestep.
        :param path: Path from decoding (Viterbi or greedy)
        :param source_words: Source words.
        :return: weights and candidates.
        """
        return np.ones((1, len(path))), [[c] for c in path]


class ViterbiEStepModel(InexactModel):

    def __init__(self, *args, **kwargs):
        """
        A model that performs inexact E-step using Viterbi or k-best Viterbi with an n-gram language model.
        """
        super().__init__(*args, **kwargs)
        assert issubclass(self.language_model_class, KenLM)

    def decode(self, source_words: Sequence[str], *args, **kwargs) -> DecoderResult:
        """
        Returns all candidates and their scores. Possibly, outputs the best path.
        :param source_words: Source words.
        :param args: Args.
        :param kwargs: Kwargs.
        :return: The results of the decoding.
        """
        origin_words = kwargs.get('origin_words')  # use original sequence of words to replace UNKs back

        num_candidates = kwargs.get('num_candidates', self.num_candidates)
        kbest = kwargs.get('kbest', self.kbest)

        if kbest:
            # get k-best predictions with k-best viterbi
            *_, k_best_paths, k_best_path_scores = self.language_model.k_best_viterbi_evaluate(
                words=source_words,
                channel_model=self.channel,
                candidate_generator=self.candidate_generator,
                num_candidates=num_candidates,
                topK=kbest,
                test=False)

            renormalized_path_scores = expc(k_best_path_scores / np.log10(np.e))  # rebase kenlm's log and normalize
            assert not np.allclose(renormalized_path_scores, np.zeros_like(renormalized_path_scores)), (source_words,
                k_best_path_scores, k_best_paths)
            weights, candidates = self._compute_weights(k_best_paths,
                                                        renormalized_path_scores,
                                                        source_words)

            k_best_paths = [self._deunker(k_best_path, origin_words) for k_best_path in k_best_paths]
            best_path = k_best_paths[0]
            best_path_score = k_best_path_scores[0]

        else:
            # get Viterbi path
            delta, _, best_path = self.language_model.viterbi_evaluate(
                words=source_words,
                channel_model=self.channel,
                candidate_generator=self.candidate_generator,
                num_candidates=num_candidates,
                test=False)

            best_path_score = delta[(*[0] * (delta.ndim - 1), -1)]  # @TODO replace hack to get (0, 0, -1) and (0, 1)

            weights, candidates = self._compute_weights_single_path(best_path)

            best_path = self._deunker(best_path, origin_words)

            k_best_paths = None
            k_best_path_scores = None

        return DecoderResult(weights=weights, candidates=candidates,
                             best_path=best_path, best_path_score=best_path_score,
                             k_best_paths=k_best_paths, k_best_path_scores=k_best_path_scores)


@dataclass
class Hypothesis:
    score: float
    individual_scores: List[Tuple[float, float]]
    prefix: List[str]
    state: Any

    def __repr__(self):
        return ('Hypothesis(score=%.3f, indscores=%r, prefix=%r, state=...)' %
                (self.score,
                 [(round(cm, 3) if cm else 0, round(lm, 3) if lm else 0) for cm, lm in self.individual_scores],
                 self.prefix))


class BeamEStepModel(InexactModel):

    def __init__(self, topk_by_channel: Optional[int] = None, *args, **kwargs):
        """
        A model that performs inexact E-step using beam search with an RNN LM.
        :param topk_by_channel: if LM scores are hard to compute, compute LM scores only for top-K candidates
            ranked by channel score.
        """
        super().__init__(*args, **kwargs)
        assert issubclass(self.language_model_class, CustomLanguageModel)
        self.topk_by_channel = topk_by_channel

    def decode(self, source_words: Sequence[str], topk_by_channel: Optional[int] = None, verbose: bool = False,
               *args, **kwargs) -> DecoderResult:
        """
        Returns all candidates and their scores. Possibly, outputs the best path.
        :param source_words: Source words.
        :param topk_by_channel: if LM scores are hard to compute, compute LM scores only for top-K candidates
            ranked by channel score.
        :param verbose: Verbose.
        :return: The results of the decoding.
        """
        origin_words = kwargs.get('origin_words')  # use original sequence of words to replace UNKs back

        num_candidates = kwargs.get('num_candidates', self.num_candidates)
        kbest = kwargs.get('kbest', self.kbest)
        unk_score = kwargs.get('unk_score', -18.0)
        topk_by_channel = topk_by_channel if topk_by_channel is None else self.topk_by_channel

        if kbest:
            # get approximate k-best predictions with beam search
            beam = [Hypothesis(0, [], [], self.language_model.initial_state())]
            len_source_words = len(source_words)
            for t in range(len_source_words + 1):
                if t < len_source_words:
                    word = source_words[t]
                    candidates = self.candidate_generator.generate(word, num_candidates)
                    if topk_by_channel:
                        canidate_scores4word = {candidate: channel_score for candidate, channel_score
                                                in sorted(self.channel.bulkscore(candidates, word),
                                                          key=lambda x: x[1])[-topk_by_channel:]}  # descending order
                        candidates = list(canidate_scores4word.keys())
                    else:
                        canidate_scores4word = {candidate: channel_score for candidate, channel_score
                                                in self.channel.bulkscore(candidates, word)}
                    if verbose:
                        print(word, '\t>>\t', len(candidates), len(set(candidates)), set(candidates))
                else:
                    # properly handle end of sequence
                    candidates = [self.language_model.eos]
                    canidate_scores4word = {self.language_model.eos : 0.}
                new_beam = []
                new_computed_scores = []
                # (1) extend all hypotheses with a new word
                for hypothesis in beam:
                    candidate_scores = []
                    computed_scores = []
                    candidate_states = []
                    # @TODO stateful batch scoring of words is only implemented for word lm
                    states_lm_scores = self.language_model.score_word_batch(candidates, state=hypothesis.state,
                                                                            unk_score=unk_score)
                    for (state, lm_score), candidate in zip(states_lm_scores, candidates):
                        channel_score = canidate_scores4word[candidate]
                        candidate_scores.append((channel_score, lm_score))
                        candidate_states.append(state)
                        # weighted sum of (possibly unnormalized) log probabilities
                        computed_scores.append(channel_score + lm_score + hypothesis.score)
                    # (2) at most `beam_size` candidates are relevant
                    k = min(len(computed_scores), kbest)
                    for idx in np.argpartition(computed_scores, -k)[-k:]:
                        cs = computed_scores[idx]
                        new_hypothesis = Hypothesis(cs,
                                                    hypothesis.individual_scores + candidate_scores[idx:idx + 1],
                                                    hypothesis.prefix + candidates[idx: idx + 1],
                                                    candidate_states[idx])
                        new_beam.append(new_hypothesis)
                        new_computed_scores.append(cs)
                # (3) promote overall-top `beam_size` hypotheses
                k = min(len(new_computed_scores), kbest)
                beam = [new_beam[idx] for idx in np.argpartition(new_computed_scores, -k)[-k:]]
                if verbose:
                    print('BEAM=', len(beam),
                          len(set([(h.score, tuple(h.individual_scores), tuple(h.prefix)) for h in beam])))
                    for h in beam:
                        print('\t', h)
            # (4) get paths objects and their scores
            k_best_paths = []
            k_best_path_scores = []
            # beam not sorted. @TODO why not use maxheap intead of argpartition ?
            for hypothesis in sorted(beam, key=lambda h: getattr(h, 'score'), reverse=True):
                k_best_paths.append(hypothesis.prefix[:-1])  # @TODO overall better handing of eos
                k_best_path_scores.append(hypothesis.score)

            # (5) compute weights, candidates
            renormalized_path_scores = expc(k_best_path_scores)
            assert not np.allclose(renormalized_path_scores, np.zeros_like(renormalized_path_scores)), (source_words,
                k_best_path_scores, k_best_paths)
            weights, candidates = self._compute_weights(k_best_paths,
                                                        renormalized_path_scores,
                                                        source_words)

            k_best_paths = [self._deunker(k_best_path, origin_words) for k_best_path in k_best_paths]
            best_path = k_best_paths[0]
            best_path_score = k_best_path_scores[0]

        else:
            # get greedy path
            raise NotImplementedError
            # best_path = None
            # weights, candidates = self._compute_weights_single_path(best_path)
            #
            # best_path = self.__deunker(best_path, origin_words)
            #
            # k_best_paths = None
            # k_best_path_scores = None

        return DecoderResult(weights=weights, candidates=candidates,
                             best_path=best_path, best_path_score=best_path_score,
                             k_best_paths=k_best_paths, k_best_path_scores=k_best_path_scores)


def expc(c: Union[np.ndarray, List[float]]) -> np.ndarray:
    d = np.exp(c - np.max(c))
    return d / d.sum()
