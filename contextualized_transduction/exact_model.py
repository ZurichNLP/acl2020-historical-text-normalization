import abc
import numpy as np
from typing import Sequence, List, Any, Optional

from contextualized_transduction.lm import KenLM
from contextualized_transduction.utils import ComponentConfig, UNK


class DecoderResult:

    def __init__(self, weights: np.ndarray, candidates: List[List[Any]], best_path: Optional[List[Any]] = None,
                 best_path_score: Optional[float] = None, k_best_paths: Optional[List[Any]] = None,
                 k_best_path_scores: Optional[List[float]] = None):
        """
        Container for decoding output.
        :param weights: 2D weight matrix: One p(c, w, t) weight for each candidate position (candidate_number, timestep)
        :param candidates: Candidates, one sequence of candidates per timestep.
        :param best_path: Viterbi path through the candidate state space.
        :param best_path_score: The score of `best_path`.
        :param k_best_paths: k-best paths (k-best Viterbi or beam paths) through the candidate state space.
        :param k_best_path_scores: The scores of `k_best_paths`.
        """
        self.weights = weights
        self.candidates = candidates
        self.best_path = best_path
        self.best_path_score = best_path_score
        self.k_best_paths = k_best_paths
        self.k_best_path_scores = k_best_path_scores


class Model(abc.ABC):

    def __init__(self, language_model_config: ComponentConfig, channel_config: ComponentConfig):
        """
        A base model defines a language model and a channel (noise) model.
        :param language_model_config: Language model class and a configuration dictionary.
        :param channel_config: Channel model class and a configuration dictionary.
        """

        self.language_model_class, self.language_model_params = language_model_config
        try:
            # @TODO remove this hack accomodating custom RNN LMs
            self.language_model = self.language_model_class.load_language_model(**self.language_model_params)
        except AttributeError:
            self.language_model = self.language_model_class(**self.language_model_params)

        self.channel_class, self.channel_params = channel_config
        self.channel = self.channel_class(**self.channel_params)

    @abc.abstractmethod
    def decode(self, source_words: Sequence[str], map_predict: bool, *args, **kwargs) -> DecoderResult:
        """
        Returns all candidates and their scores. Possibly, outputs the Viterbi path.
        :param source_words: Source words.
        :param map_predict: If applicable, return maximum a posteriori sequence of candidates and its score.
        :param args: Args.
        :param kwargs: Kwargs.
        :return: The results of the decoding.
        """
        pass

    def _deunker(self, hypothesis: List[str], origin_words: List[str]) -> List[str]:
        return [origin_words[i] if w == UNK else w for i, w in enumerate(hypothesis)]


class ExactModel(Model):

    def __init__(self, language_model_config: ComponentConfig, channel_config: ComponentConfig,
                 candidate_generator_config: ComponentConfig, num_candidates: int, *args, **kwargs):
        """
        A model that performs exact decoding using an n-gram language model. The set of permissible hypotheses are
        given by the candidate generator, which generates candidates for each source word.
        :param language_model_config: Language model class and a configuration dictionary.
        :param channel_config: Channel model class and a configuration dictionary.
        :param candidate_generator_config: Candidate generator model class and a configuration dictionary.
        :param num_candidates: Default number of candidates to generate using candidate generator. Used for training.
        """

        self.candidate_generator_class, self.candidate_generator_params = candidate_generator_config
        self.candidate_generator = self.candidate_generator_class(**self.candidate_generator_params)

        super().__init__(language_model_config=language_model_config, channel_config=channel_config)

        # @TODO extend the interface of KenLMs to any exact posterior model.
        assert issubclass(self.language_model_class, KenLM)

        self.num_candidates = num_candidates

    def decode(self, source_words: Sequence[str], map_predict: bool, *args, **kwargs) -> DecoderResult:
        """
        Returns all candidates and their scores. Possibly, outputs the Viterbi path.
        :param source_words: Source words.
        :param map_predict: Return maximum aposteriori sequence of candidates and its score.
        :param args: Args.
        :param kwargs: Kwargs.
        :return: The results of the decoding.
        """
        num_candidates = kwargs.get('num_candidates', self.num_candidates)

        # @TODO `compute_weight` should return `DecoderResult` directly
        weights, candidates, (best_path, best_path_score) = \
            self.language_model.compute_weight(words=source_words,
                                               channel=self.channel,
                                               candidate_generator=self.candidate_generator,
                                               num_candidates=num_candidates,
                                               predict=map_predict,
                                               test=False)

        origin_words = kwargs.get('origin_words')
        if origin_words:
            # use original sequence of words to replace UNKs back
            best_path = self._deunker(best_path, origin_words)

        kbest = kwargs.get('kbest')
        if kbest:
            # get k-best predictions with k-best viterbi
            *_, k_best_paths, k_best_path_scores = self.language_model.k_best_viterbi_evaluate(
                words=source_words,
                channel_model=self.channel,
                candidate_generator=self.candidate_generator,
                num_candidates=num_candidates,
                topK=kbest,
                test=False)

            if origin_words:
                k_best_paths = [self._deunker(k_best_path, origin_words) for k_best_path in k_best_paths]

            assert best_path == k_best_paths[0], (best_path, k_best_paths[0], source_words)
            assert np.isclose(best_path_score, k_best_path_scores[0]), \
                (best_path_score, k_best_path_scores[0], best_path, k_best_paths[0], source_words)
        else:
            k_best_paths = None
            k_best_path_scores = None

        return DecoderResult(weights=weights, candidates=candidates,
                             best_path=best_path, best_path_score=best_path_score,
                             k_best_paths=k_best_paths, k_best_path_scores=k_best_path_scores)