"""
Based on Ristad and Yianilos (1998) Learning String Edit Distance.
(https://www.researchgate.net/publication/3192848_Learning_String_Edit_Distance)
"""
import numbers
import time
import os
import pickle
import math
import multiprocessing
from abc import ABC, abstractmethod
from collections import namedtuple, defaultdict
from typing import Iterable, Sequence, Optional, Generator, Tuple, Any, Union, List

import numpy as np
import scipy.misc

from contextualized_transduction.utils import expand_path, LARGE_NEG_CONST, ASCII_ALPHABET, EOS, TOL

Sub = namedtuple('Sub', 'old new')
Del = namedtuple('Del', 'old')
Ins = namedtuple('Ins', 'new')


class Channel(ABC):
    """
    Channel is a model for p(source word | target word).
    """

    @abstractmethod
    def bulkscore(self, cands: Iterable, word: Any) -> Generator[Tuple[Any, float], None, None]:
        """
        Score candidates for a word.
        :param cands: Candidates.
        :param word: Word.
        :return: Candidates paired with scores.
        """
        pass

    def bulkscore_state_space(self, candidate_state_space: Sequence[Iterable], words: Sequence[Any]) -> \
            List[List[Tuple[Any, float]]]:
        """
        Score the generated candidates for all words.
        :param candidate_state_space: Generated candidates, one batch for each word.
        :param words: The sequence of words.
        :return: A sequence of candidates paired with scores, one for each word.
        """
        assert len(candidate_state_space) == len(words), \
            'Number of steps in `candidate_state_space` does not match the number of words!'
        channel_scores: List[List[Tuple[Any, float]]] = []
        for cands, word in zip(candidate_state_space, words):
            channel_scores.append(list(self.bulkscore(cands, word)))
        return channel_scores

    @abstractmethod
    def save_model(self, path2model: str) -> None:
        """
        If model is trainable, save parameters to file after training.
        :param path2model: Path to model file.
        """
        pass

    @abstractmethod
    def update_model(self, sources: Iterable, targets: Iterable, *args,
                     weights: Optional[Iterable[float]] = None, **kwargs) -> None:
        """
        Update channel's parameters by optimizing some criterion based on (weighted) pairs of source and target words.
        :param sources: Source words (candidates).
        :param targets: Target words.
        :param weights: Weights.
        """
        pass


class DummyChannel(Channel):
    """
    Channel that does not score candidates. Use in the first iteration of EM learning as a placeholder
    for a real channel.
    """
    def bulkscore(self, cands: Iterable, word: str) -> Generator[Tuple[str, float], None, None]:
        for cand in cands:
            yield cand, 0.  # log prob(word | cand)

    def save_model(self, path2model: str) -> None:
        pass

    def update_model(self, sources: Iterable, targets: Iterable, *args,
                 weights: Optional[Iterable[float]] = None, **kwargs) -> None:
        pass

    def reset_from_path(self, path: str):
        self.param_path = path

    def best_param_path(self, epoch: int, **kwargs) -> str:
        return self.param_path


class StochasticEditDistance_Channel(Channel):
    EOS = EOS

    def __init__(self, source_alphabet: Optional[Iterable] = ASCII_ALPHABET,
                 target_alphabet: Optional[Iterable] = ASCII_ALPHABET,
                 param_dicts: Optional[str] = None, smart_init: bool = False, copy_proba: float = 0.9,
                 discount: float = 10 ** -5, generated_candidates_fn: Optional[str] = None,
                 score_candidates_on_init: bool = False, em_data_dir: str = None, *args, **kwargs) -> None:
        """
        Implementation of the Stochastic Edit Distance model from Ristad & Yianilos 1998 "Learning String Edit
        Distance". The model is a memoryless probabilistic weighted finite-state transducer that by use of character
        edits (insertions, deletions, substitutions) maps one string to another. We use it to model channel:
        p(observed possibly erroneous word | candidate replacement). Edit weights are learned with
        Expectation-Maximization. The channel can be decoded either by computing forward probabilities (the correct way,
        but somewhat slow in log real) or by finding the highest probability (=Viterbi) sequence of edits
        (approximation: max_{edits} p(observed possibly erroneous word, edit | candidate replacement), which runs
        faster). For further details, see referenced paper.

        :param source_alphabet: Characters of all input strings.
        :param target_alphabet: Characters of all target strings.
        :param param_dicts: Dictionaries of learned parameters.
        :param smart_init: Initialization of parameters with bias towards copying.
        :param copy_proba: If smart_init, how much mass to give to copy edits.
        :param discount: Pseudocount for assigning non-zero probability to unknown edit (to meaningfully handle OOV
            words).
        :param generated_candidates_fn: To speed up prediction for a fixed dataset, score all (source, target) pairs
            from this tab-separated file.
        :param score_candidates_on_init: Whether to score the (source, target) pairs from `generated_candidates_fn` on
            initialization.
        :param em_data_dir: On EM learning of new parameters, in which directory (if at all) to save the latest weights.
        """
        self.param_dicts = param_dicts
        self.smart_init = smart_init
        self.copy_proba = copy_proba
        self.EOS = EOS
        if param_dicts:
            # load dicts from file
            path2pkl = expand_path(param_dicts)
            self.from_pickle(path2pkl)
        else:
            self.source_alphabet = list(set(source_alphabet))
            self.target_alphabet = list(set(target_alphabet))

            self.len_source_alphabet = len(self.source_alphabet)
            self.len_target_alphabet = len(self.target_alphabet)

            # all edits
            self.N = self.len_source_alphabet * self.len_target_alphabet + \
                     self.len_source_alphabet + self.len_target_alphabet + 1

            self.discount = discount
            # for simplicity, ignore discount in initialization
            if self.smart_init:
                # initialize weights with a strong bias to copying
                assert 0 < self.copy_proba < 1, f'0 < copy probility={self.copy_proba} < 1 doesn\'t hold.'
                num_copy_edits = len(set(self.target_alphabet) & set(self.source_alphabet))
                num_rest = self.N - num_copy_edits
                # params for computing word confusion probability
                alpha = np.log(copy_proba / num_copy_edits)  # copy log probability
                log_p = np.log((1 - copy_proba) / num_rest)  # log probability of substitution, deletion, and insertion
                self.delta_sub = {(s, t): alpha if s == t else log_p
                                  for s in self.source_alphabet for t in self.target_alphabet}
                self.delta_del = {s: log_p for s in self.source_alphabet}
                self.delta_ins = {t: log_p for t in self.target_alphabet}
                self.delta_eos = log_p
            else:
                # initialize weights uniformly
                uniform_weight = np.log(1 / self.N)
                self.delta_sub = {(s, t): uniform_weight for s in self.source_alphabet for t in self.target_alphabet}
                self.delta_del = {s: uniform_weight for s in self.source_alphabet}
                self.delta_ins = {t: uniform_weight for t in self.target_alphabet}
                self.delta_eos = uniform_weight

        self.default = np.log(self.discount / self.N)  # log probability of unseen edits

        assert len(self.delta_sub) + len(self.delta_del) + len(self.delta_ins) + 1 == self.N
        assert np.isclose(0., scipy.misc.logsumexp(list(self.delta_sub.values()) + list(self.delta_del.values()) +
                                                   list(self.delta_ins.values()) + [self.delta_eos]))

        self.generated_candidates_fn = generated_candidates_fn
        self.big_table: Optional[dict] = None
        if score_candidates_on_init:
            self._reset_big_table()
        self.em_data_dir = em_data_dir

    def from_pickle(self, path2pkl: str) -> None:
        """
        Load delta* parameters from pickle.
        :param path2pkl: Path to pickle.
        """
        try:
            print('Loading sed channel parameters from file: ', path2pkl)
            with open(path2pkl, 'rb') as w:
                self.delta_sub, self.delta_del, self.delta_ins, self.delta_eos, self.discount = pickle.load(w)

            # ignore input alphabets
            self.source_alphabet = list(self.delta_del.keys())
            self.target_alphabet = list(self.delta_ins.keys())

            self.len_source_alphabet = len(self.source_alphabet)
            self.len_target_alphabet = len(self.target_alphabet)

            # all edits
            self.N = self.len_source_alphabet * self.len_target_alphabet + \
                     self.len_source_alphabet + self.len_target_alphabet + 1

            # some random sanity checks
            assert all((s, t) in self.delta_sub for s in self.source_alphabet for t in self.target_alphabet)
            assert len(self.delta_sub) == len(self.source_alphabet) * len(self.target_alphabet)
            assert isinstance(self.delta_eos, numbers.Real)
        except (OSError, KeyError, AssertionError) as e:
            print(f'"{path2pkl}" exists and data in right format?', os.path.exists(path2pkl))
            raise e

    def forward_evaluate(self, source: Sequence, target: Sequence) -> np.ndarray:
        """
        Compute dynamic programming table (in log real) filled with forward log probabilities.
        :param source: Source string.
        :param target: Target string.
        :return: Dynamic programming table.
        """
        T, V = len(source), len(target)
        alpha = np.full((T + 1, V + 1), LARGE_NEG_CONST)
        alpha[0, 0] = 0.
        for t in range(T + 1):
            for v in range(V + 1):
                summands = [alpha[t, v]]
                if v > 0:
                    summands.append(self.delta_ins.get(target[v - 1], self.default) + alpha[t, v - 1])
                if t > 0:
                    summands.append(self.delta_del.get(source[t - 1], self.default) + alpha[t - 1, v])
                if v > 0 and t > 0:
                    summands.append(
                        self.delta_sub.get((source[t - 1], target[v - 1]), self.default) + alpha[t - 1, v - 1])
                alpha[t, v] = scipy.misc.logsumexp(summands)
        alpha[T, V] += self.delta_eos
        return alpha

    def backward_evaluate(self, source: Sequence, target: Sequence) -> np.ndarray:
        """
        Compute dynamic programming table (in log real) filled with backward log probabilities (the probabilities of
        the suffix, i.e. p(source[t:], target[v:]) e.g. p('', 'a') = p(ins(a))*p(#).
        :param source: Source string.
        :param target: Target string.
        :return: Dynamic programming table.
        """
        T, V = len(source), len(target)
        beta = np.full((T + 1, V + 1), LARGE_NEG_CONST)
        beta[T, V] = self.delta_eos
        for t in range(T, -1, -1):
            for v in range(V, -1, -1):
                summands = [beta[t, v]]
                if v < V:
                    summands.append(self.delta_ins[target[v]] + beta[t, v + 1])
                if t < T:
                    summands.append(self.delta_del[source[t]] + beta[t + 1, v])
                if v < V and t < T:
                    summands.append(self.delta_sub[(source[t], target[v])] + beta[t + 1, v + 1])
                beta[t, v] = scipy.misc.logsumexp(summands)
        return beta

    def ll(self, sources: Sequence, targets: Sequence, weights: Sequence[float] = None,
           decode_threads: Optional[int] = None):
        """
        Computes weighted log likelihood.
        :param sources: Source strings.
        :param targets: Target strings.
        :param weights: Weights for pairs of source-target strings.
        :param decode_threads: Speed up the for loop.
        :return: Weighted log likelihood.
        """
        if decode_threads and decode_threads > 1:

            data_len = len(sources)
            step_size = math.ceil(data_len / decode_threads)
            print('Will compute weighted likelihood in {} chunks of size {}'.format(
                decode_threads, step_size))
            pool = multiprocessing.Pool()
            grouped_samples = [(sources[i:i + step_size], targets[i:i + step_size], weights[i:i + step_size])
                               for i in range(0, data_len, step_size)]
            results = pool.map(self._weighted_forward, grouped_samples)
            pool.terminate()
            pool.join()
            ll = np.mean([w for ww in results for w in ww])
        else:
            # single thread computation
            ll = np.mean([weight * self.forward_evaluate(source, target)[-1, -1]
                          for source, target, weight in zip(sources, targets, weights)])
        return ll

    def _weighted_forward(self, ss_tt_ww: Tuple[List, List, List]):
        """
        Helper function for parallelized computation of weighted log likelihood.
        :param ss_tt_ww: Tuple of sources, targets, weights.
        :return: Weighted log likelihood of the samples.
        """
        ss, tt, ww = ss_tt_ww
        return [w * self.forward_evaluate(s, t)[-1, -1] for s, t, w in zip(ss, tt, ww)]

    class Gammas:
        def __init__(self, sed: 'StochasticEditDistance_Channel'):
            """
            Container for non-normalized probabilities.
            :param sed: Channel.
            """
            self.sed = sed
            self.eos = 0
            self.sub = {k: 0. for k in self.sed.delta_sub}
            self.del_ = {k: 0. for k in self.sed.delta_del}
            self.ins = {k: 0. for k in self.sed.delta_ins}

        def normalize(self):
            """
            Normalize probabilities and assign them to the channel's deltas.
            :param discount: Unnormalized quantity for edits unseen in training.
            """
            # all mass to distribute among edits
            denom = np.log(self.eos + sum(self.del_.values()) + sum(self.ins.values()) +
                           sum(self.sub.values()) + self.sed.discount * self.sed.N)
            self.sub = {k: np.log(self.sub[k] + self.sed.discount) - denom for k in self.sub}
            self.del_ = {k: np.log(self.del_[k] + self.sed.discount) - denom for k in self.del_}
            self.ins = {k: np.log(self.ins[k] + self.sed.discount) - denom for k in self.ins}
            self.eos = np.log(self.eos + self.sed.discount) - denom

            assert len(self.sub) + len(self.del_) + len(self.ins) + 1 == self.sed.N
            check_sum = scipy.misc.logsumexp(list(self.sub.values()) + list(self.del_.values()) +
                                             list(self.ins.values()) + [self.eos])
            assert np.isclose(0., check_sum), check_sum
            # set the channel's delta to log normalized gammas
            self.sed.delta_eos = self.eos
            self.sed.delta_sub = self.sub
            self.sed.delta_ins = self.ins
            self.sed.delta_del = self.del_

    def em(self, sources: Sequence, targets: Sequence, weights: Optional[Sequence[float]] = None,
           iterations: int = 10, decode_threads: Optional[int] = None, test: bool = True,
           verbose: bool = False) -> None:
        """
        Update the channel parameter's delta* with Expectation-Maximization.
        :param sources: Source strings.
        :param targets: Target strings.
        :param weights: Weights for the pairs of source and target strings.
        :param iterations: Number of iterations of EM.
        :param decode_threads: Number of threads to use for the computation of log-likelihood if test is True.
        :param test: Whether to report log-likelihood.
        :param verbose: Verbosity.
        """
        if weights is None:
            weights = [1.] * len(sources)
        if test:
            print('Initial weighted LL=', self.ll(sources, targets, weights, decode_threads=0))
        for i in range(iterations):
            gammas = self.Gammas(self)  # container for unnormalized probabilities
            for sample_num, (source, target, weight) in enumerate(zip(sources, targets, weights)):
                if weight < TOL:
                    if verbose:
                        print('Weight is below TOL. Skipping: ', source, target, weight)
                    continue
                self.expectation_step(source, target, gammas, weight)
                if test and sample_num > 0 and sample_num % 1000 == 0:
                    print(f'\t...processed {sample_num} samples')
            gammas.normalize()  # maximization step: normalize and assign to self.delta*
            if test:
                print('IT_{}='.format(i), self.ll(sources, targets, weights, decode_threads=0))

    def expectation_step(self, source: Sequence, target: Sequence, gammas: Gammas, weight: float = 1.) -> None:
        """
        Accumumate soft counts.
        :param source: Source string.
        :param target: Target string.
        :param gammas: Unnormalized probabilities that we are learning.
        :param weight: Weight for the pair (`source`, `target`).
        """
        alpha = np.exp(self.forward_evaluate(source, target))
        beta = np.exp(self.backward_evaluate(source, target))
        gammas.eos += weight
        T, V = len(source), len(target)
        for t in range(T + 1):
            for v in range(V + 1):
                # (alpha = probability of prefix) * probability of edit * (beta = probability of suffix)
                rest = beta[t, v] / alpha[T, V]
                if t > 0:
                    gammas.del_[source[t - 1]] += \
                        weight * alpha[t - 1, v] * np.exp(self.delta_del[source[t - 1]]) * rest
                if v > 0:
                    gammas.ins[target[v - 1]] += \
                        weight * alpha[t, v - 1] * np.exp(self.delta_ins[target[v - 1]]) * rest
                if t > 0 and v > 0:
                    gammas.sub[(source[t - 1], target[v - 1])] += \
                        weight * alpha[t - 1, v - 1] * np.exp(self.delta_sub[(source[t - 1], target[v - 1])]) * rest


    def bulkscore(self, cands: Iterable, word: Any, with_viterbi_distance: bool = True) -> \
            Generator[Tuple[Any, float], None, None]:
        """
        Score a set of candidates with either Viterbi or stochastic edit distance.
        :param cands: Candidates.
        :param word: Possibly erroneous word to be explained.
        :param with_viterbi_distance: Whether to use Viterbi.
        :return Pairs of candidates and their scores.
        """
        if with_viterbi_distance:
            distance = self.viterbi_distance
        else:
            distance = self.stochastic_distance

        if self.big_table and word in self.big_table:
            word_results = self.big_table[word]
            for cand in cands:
                cidx = word_results['candidates'].index(cand)
                yield cand, word_results['log_prob'][cidx]
        else:
            for cand in cands:
                yield cand, distance(source=cand, target=word)

    def viterbi_distance(self, source: Sequence, target: Sequence, with_alignment: bool = False) -> \
            Union[float, Tuple[List, float]]:
        """
        Viterbi edit distance \propto max_{edits} p(target, edit | source).
        :param source: Source string.
        :param target: Target string.
        :param with_alignment: Whether to output the corresponding sequence of edits.
        :return: Probability score and, optionally, the sequence of edits that gives this score.
        """
        T, V = len(source), len(target)
        alpha = np.full((T + 1, V + 1), LARGE_NEG_CONST)
        alpha[0, 0] = 0.
        for t in range(T + 1):
            for v in range(V + 1):
                alternatives = [alpha[t, v]]
                if v > 0:
                    alternatives.append(self.delta_ins.get(target[v - 1], self.default) + alpha[t, v - 1])
                if t > 0:
                    alternatives.append(self.delta_del.get(source[t - 1], self.default) + alpha[t - 1, v])
                if v > 0 and t > 0:
                    alternatives.append(self.delta_sub.get((source[t - 1], target[v - 1]), self.default) + alpha[t - 1, v - 1])
                alpha[t, v] = max(alternatives)
        alpha[T, V] += self.delta_eos
        optim_score = alpha[T, V]
        if not with_alignment:
            return optim_score
        else:
            # compute an optimal alignment
            alignment = []
            ind_w, ind_c = len(source), len(target)
            while ind_w >= 0 and ind_c >= 0:
                if ind_w == 0 and ind_c == 0:
                    return alignment[::-1], optim_score
                elif ind_w == 0:
                    # can only go left, i.e. via insertions
                    ind_c -= ind_c
                    alignment.append(Ins(target[ind_c]))  # minus 1 is due to offset
                elif ind_c == 0:
                    # can only go up, i.e. via deletions
                    ind_w -= ind_w
                    alignment.append(Del(source[ind_w]))  # minus 1 is due to offset
                else:
                    # pick the smallest cost actions
                    pind_w = ind_w - 1
                    pind_c = ind_c - 1
                    action_idx = np.argmax([alpha[pind_w, pind_c],
                                            alpha[ind_w, pind_c],
                                            alpha[pind_w, ind_c]])
                    if action_idx == 0:
                        action = Sub(source[pind_w], target[pind_c])
                        ind_w = pind_w
                        ind_c = pind_c
                    elif action_idx == 1:
                        action = Ins(target[pind_c])
                        ind_c = pind_c
                    else:
                        action = Del(source[pind_w])
                        ind_w = pind_w
                    alignment.append(action)

    def stochastic_distance(self, source: Sequence, target: Sequence) -> float:
        """
        Stochastic edit distance \propto sum_{edits} p(target, edit | source) = p(target | source)
        :param source: Source string.
        :param target: Target string.
        :return: Probability score.
        """
        return self.forward_evaluate(source, target)[-1, -1]

    def to_pickle(self, pkl: str, relative2data: bool = True) -> None:
        """
        Write parameters delta* to file.
        :param pkl: The pickle filename.
        :param relative2data: Write to data directory.
        """
        with open(expand_path(pkl) if relative2data else pkl, 'wb') as w:
            pickle.dump((self.delta_sub, self.delta_del, self.delta_ins, self.delta_eos, self.discount), w)

    def save_model(self, path2model: str) -> None:
        return self.to_pickle(path2model, relative2data=False)

    def update_model(self, sources: Sequence, targets: Sequence, weights: Optional[Sequence[float]] = None,
                     em_iterations: int = 10, decode_threads: Optional[int] = None, epoch_number: Optional[int] = None,
                     test: bool = True, verbose: bool = False, **kwargs) -> None:
        """
        Update channel model parameters by maximizing weighted likelihood by Expectation-Maximization.
        :param sources: Source strings.
        :param targets: Target strings.
        :param weights: Weights for the pairs of source and target strings.
        :param em_iterations: Number of iterations of EM.
        :param decode_threads: Number of threads to use for the computation of log-likelihood if test is True.
        :param epoch_number: For saving the weights after this update, the ordinal number of the training epoch.
        :param test: Whether to report log-likelihood.
        :param verbose: Verbosity.
        """
        if test:
            print('Updating channel model parameters by maximizing weighted likelihood using EM ({} iterations).'.
                  format(em_iterations))
        self.em(sources=sources, targets=targets, weights=weights, iterations=em_iterations,
                decode_threads=decode_threads, test=test, verbose=verbose)

        self._reset_big_table(decode_threads=decode_threads)

        if self.em_data_dir:
            path2pkl = os.path.join(self.em_data_dir,
                                    'param_dict_e{}.pkl'.format('latest' if epoch_number is None else epoch_number))
            self.save_model(path2pkl)
            print(f'Wrote latest model weights to "{path2pkl}".')

    def _batch_bulkscore(self, words_candidates):
        """
        Score a batch of (word, candidates) pairs with either Viterbi or stochastic edit distance.
        :param words_candidates: Words and sets of candidates.
        :return Dictionary from words to dictionaries of candidates and their scores.
        """
        words_candidates, with_viterbi_distance = words_candidates
        if with_viterbi_distance:
            distance = self.viterbi_distance
        else:
            distance = self.stochastic_distance
        output = dict()
        for word, cands in words_candidates:
            cands = list(cands)
            log_prob = [distance(source=cand, target=word) for cand in cands]
            output[word] = {'candidates': cands, 'log_prob': log_prob}
        return output

    def _reset_big_table(self, decode_threads: Optional[int] = 4, with_viterbi_distance: bool = True) -> None:
        """
        Score (source, target) pairs from the fixed dataset with the latest parameters.
        """
        if self.generated_candidates_fn is not None and os.path.exists(self.generated_candidates_fn):
            print(f'Will reset the big table by scoring candidates from "{self.generated_candidates_fn}" ...')
            then = time.time()
            data_samples = defaultdict(set)
            with open(self.generated_candidates_fn) as f:
                for line in f:
                    line = line.rstrip()
                    if line:
                        source, target, *_rest = line.split('\t')
                        data_samples[target].add(source)

            data_samples = list(data_samples.items())
            len_data = len(data_samples)
            step_size = math.ceil(len_data / decode_threads if decode_threads else 1)
            grouped_samples = [(data_samples[i:i + step_size], with_viterbi_distance)
                               for i in range(0, len_data, step_size)]
            pool = multiprocessing.Pool()
            results = pool.map(self._batch_bulkscore, grouped_samples)
            pool.terminate()
            pool.join()
            self.big_table = dict()
            for batch_dict in results:
                self.big_table.update(batch_dict)
            print(f'Finished in {(time.time() - then):.3f} sec.')
        else:
            print(f'Path does not exist: {self.generated_candidates_fn}. Cannot reset the big table.')

    def reset_from_path(self, fn: str) -> None:
        """
        Load scores from file and re-compute alphabets.
        :param fn: Path to file with scores.
        """
        self.from_pickle(fn)
        self._reset_big_table()

    def best_param_path(self, epoch_number: Optional[int], **kwargs) -> str:
        """
        Given an EM training epoch number, figure out the name of the file with channel scores for to this epoch.
        :param epoch_number: Epoch number.
        :return: The corresponding filename.
        """
        path2pkl = os.path.join(self.em_data_dir,
                                'param_dict_e{}.pkl'.format('latest' if epoch_number is None else epoch_number))
        if not os.path.exists(path2pkl) and epoch_number == -1:
            print(f'"{path2pkl}" does not exists, loading from "{self.param_dicts}" ...')
            path2pkl = self.param_dicts
        assert os.path.exists(path2pkl), f"best path does not exist: {path2pkl}"
        return path2pkl