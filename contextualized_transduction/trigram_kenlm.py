from typing import Sequence, Tuple, ClassVar, Union, List, Any

from dataclasses import dataclass
import kenlm
import numpy as np
import heapq

from contextualized_transduction.candidate_generator import CandidateGenerator
from contextualized_transduction.lm import KenLM
from contextualized_transduction.sed_channel import Channel
from contextualized_transduction.utils import LARGE_NEG_CONST, logsumexp10
from contextualized_transduction.bigram_kenlm import KEN_EOS

KEN_BOS = "<s>"

@dataclass
class IJContext:
    word_i: str
    state_ij: Union[kenlm.State, List[kenlm.State]]

@dataclass
class IJContainer:
    word_j: str
    contexts: List[IJContext]

class Context:
    def __init__(self, state: Any, context: Sequence, index: Tuple):
        self.state = state
        self.context = context
        self.recent = self.context[1:]
        self.head = self.context[-1]
        self.index = index

    def __repr__(self):
        return f'Context({self.context})'


class TrigramKenLM(KenLM):
    ORDER: ClassVar[int] = 3

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.model.order == self.ORDER, 'Wrong n-gram order: {0} vs {1}'.format(
            self.model.order - 1, self.ORDER - 1)

    def fviterbi_evaluate(self, words: Sequence[str], channel_model: Channel,
                         candidate_generator: CandidateGenerator,
                         num_candidates: int, test: bool = True) -> \
            Tuple[np.ndarray, List[List[str]], List[str]]:
        """
        Compute Viterbi sequence. Reference implementation.
        :param words: Sequence of error words.
        :param channel_model: Channel model that return p(error word | correct word) score.
        :param candidate_generator: Generator of correct word candidates for an input error word.
        :param num_candidates: Number of candidates to generate using `candidate_generator`.
        :param test: Whether this is testing and the method can be verbose.
        :return: Viterbi probabilities, the list of lists of correct word candidates, one for each cell in
            the viterbi probabilities matrix, and the viterbi path.
        """

        T = len(words)
        # num_candidates + 1 is due to "error word" itself possibly being an extra candidate
        # add 2 timesteps for eos observation and the storage of the final value
        shape = (num_candidates + 1), (num_candidates + 1), (T + 2)
        delta = np.full(shape, LARGE_NEG_CONST)  # each state is one context
        psi = np.full(shape, 0, dtype=np.uint32)  # traceback
        contexts: List[List[IJContainer]] = []
        candidate_states: List[List[str]] = []

        # INITIALIZATION
        candidates = candidate_generator.generate(words[0], num_candidates=num_candidates)
        start_state = kenlm.State()
        self.model.BeginSentenceWrite(start_state)
        new_contexts: List[IJContainer] = []
        for k, (candidate, channel_score) in enumerate(channel_model.bulkscore(candidates, words[0])):
            state_0 = kenlm.State()
            delta[k, 0, 0] = channel_score + self.model.BaseScore(start_state, candidate, state_0)
            if test:
                print('Transiting to (%s) and emitting (%s): %.3f' % (candidate, words[0], delta[k, 0, 0]))
            future_ij_contexts: List[IJContext] = [IJContext(KEN_BOS, state_0)]
            new_contexts.append(IJContainer(candidate, future_ij_contexts))
        contexts.append(new_contexts)
        candidate_states.append(list(candidates))

        # RECURSION
        for t, word in enumerate(words[1:], start=1):
            candidates = candidate_generator.generate(word, num_candidates=num_candidates)
            new_contexts: List[IJContainer] = []
            for k, (candidate, channel_score) in enumerate(channel_model.bulkscore(candidates, word)):
                future_ij_contexts: List[IJContext] = []
                for j, ij_container in enumerate(contexts[-1]):
                    state_jk = kenlm.State()  # state_jk is the emission state
                    all_transition_proba = []
                    for i, context in enumerate(ij_container.contexts):
                        # state_ij stores bigram-state context (i, j)
                        all_transition_proba.append(delta[j, i, t - 1] +
                                                    self.model.BaseScore(context.state_ij, candidate, state_jk))
                    future_ij_contexts.append(IJContext(ij_container.word_j, state_jk))
                    max_i = np.argmax(all_transition_proba)
                    delta[k, j, t] = channel_score + all_transition_proba[max_i]
                    psi[k, j, t] = max_i
                new_contexts.append(IJContainer(candidate, future_ij_contexts))
            contexts.append(new_contexts)
            candidate_states.append(list(candidates))

        # TERMINATION
        state_k = kenlm.State()
        for j, ij_container in enumerate(contexts[-1]):
            all_transition_proba = []
            for i, context in enumerate(ij_container.contexts):
                all_transition_proba.append(delta[j, i, T - 1] +
                                            self.model.BaseScore(context.state_ij, KEN_EOS, state_k))
            max_i = np.argmax(all_transition_proba)
            delta[0, j, T] = all_transition_proba[max_i]  # emit only KEN_EOS hence emission factor equals 0.
            psi[0, j, T] = max_i

        # TRACEBACK
        max_j = np.argmax(delta[0, :, T])
        delta[0, 0, T + 1] = delta[0, max_j, T]  # unnecessary

        max_i = psi[0, max_j, T]
        viterbi_path: List[str] = [contexts[T - 1][max_j].word_j]
        for t in range(T - 2, -1, -1):
            viterbi_path.append(contexts[t][max_i].word_j)
            l = psi[max_j, max_i, t + 1]
            max_j = max_i
            max_i = l

        return delta, candidate_states, viterbi_path[::-1]


    def viterbi_evaluate(self, words: Sequence[str], channel_model: Channel,
                         candidate_generator: CandidateGenerator,
                         num_candidates: int, test: bool = True) -> \
            Tuple[np.ndarray, List[List[str]], List[str]]:
        """
        Compute Viterbi sequence.
        :param words: Sequence of error words.
        :param channel_model: Channel model that return p(error word | correct word) score.
        :param candidate_generator: Generator of correct word candidates for an input error word.
        :param num_candidates: Number of candidates to generate using `candidate_generator`.
        :param test: Whether this is testing and the method can be verbose.
        :return: Viterbi probabilities, the list of lists of correct word candidates, one for each cell in
            the viterbi probabilities matrix, and the viterbi path.
        """

        T = len(words)
        # num_candidates + 1 is due to "error word" itself possibly being an extra candidate
        # add 2 timesteps for eos observation and the storage of the final value
        shape = (T + 2), (num_candidates + 1), (num_candidates + 1)
        delta = np.full(shape, LARGE_NEG_CONST)  # each state is one context
        psi = np.full(shape, 0, dtype=np.uint32)  # traceback
        candidate_states: List[List[str]] = [[KEN_BOS]]
        candidate_states_idxes: List[range] = [range(1)]

        # 2 X (num_candidates + 1) X (num_candidates + 1)
        kenlm_states = tuple(
            tuple(
                tuple(kenlm.State() for _ in range(num_candidates + 1))
                for _ in range(num_candidates + 1))
            for _ in range(2))

        # INITIALIZATION
        candidates = list(candidate_generator.generate(words[0], num_candidates=num_candidates))
        start_state = kenlm.State()
        self.model.BeginSentenceWrite(start_state)
        _, channel_scores = zip(*channel_model.bulkscore(candidates, words[0]))
        cand_idxes = range(len(candidates))
        delta[0, cand_idxes, 0] = \
            [self.model.BaseScore(start_state, candidate, kenlm_states[0][k][0])
             for k, candidate in enumerate(candidates)]
        delta[0, cand_idxes, 0] += channel_scores
        candidate_states.append(candidates)
        candidate_states_idxes.append(cand_idxes)

        # RECURSION
        for t, word in enumerate(words[1:], start=1):
            states_from = kenlm_states[int(t % 2 == 0)]
            states_to = kenlm_states[int(t % 2 == 1)]
            candidates = list(candidate_generator.generate(word, num_candidates=num_candidates))
            _, channel_scores = zip(*channel_model.bulkscore(candidates, word))
            cmin2_idxes = candidate_states_idxes[-2]
            cmin1_idxes = candidate_states_idxes[-1]
            cand_idxes = range(len(candidates))
            # 1 X j X i
            delta_tmin1 = delta[np.ix_([t - 1], cmin1_idxes, cmin2_idxes)]
            # k X j X i
            transition_scores = delta_tmin1 + [
                    [
                        [self.model.BaseScore(states_from[j][i], candidate, states_to[k][j])
                            for i in cmin2_idxes]
                    for j in cmin1_idxes]
                for k, candidate in enumerate(candidates)]
            # k X j
            max_i = transition_scores.argmax(axis=2)
            max_i_idxes = np.array(cand_idxes)[:, np.newaxis], cmin1_idxes, max_i
            # 1 X k X j
            t_k_j_idx = np.ix_([t], cand_idxes, cmin1_idxes)
            delta[t_k_j_idx] = np.array(channel_scores)[:, np.newaxis] + transition_scores[max_i_idxes]
            psi[t_k_j_idx] = max_i

            candidate_states.append(candidates)
            candidate_states_idxes.append(cand_idxes)

        # TERMINATION
        state_k = kenlm.State()
        states_from = kenlm_states[int(T % 2 == 0)]
        cmin2_idxes = candidate_states_idxes[-2]
        cmin1_idxes = candidate_states_idxes[-1]
        # 1 X j X i + j X i
        transition_scores = delta[np.ix_([T - 1], cmin1_idxes, cmin2_idxes)].squeeze(axis=0) + \
            [[self.model.BaseScore(states_from[j][i], KEN_EOS, state_k) for i in cmin2_idxes] for j in cmin1_idxes]
        max_i = transition_scores.argmax(axis=1)
        delta[T, 0, cmin1_idxes] = transition_scores[(cmin1_idxes, max_i)]  # emission score 0 (only KEN_EOS emitted)
        psi[T, 0, cmin1_idxes] = max_i

        # TRACEBACK
        max_j = np.argmax(delta[T, 0, :])
        delta[T + 1, 0, 0] = delta[T, 0, max_j]  # unnecessary

        max_i = psi[T, 0, max_j]
        viterbi_path: List[str] = [candidate_states[T][max_j]]
        # `candidate_states` has [KEN_BOS] at index 0 therefore all indices into it are + 1
        for t in range(T - 1, 0, -1):
            viterbi_path.append(candidate_states[t][max_i])
            l = psi[t, max_j, max_i]
            max_j = max_i
            max_i = l

        # ... transpose since we want to retrieve [0, 0, -1]-th element
        return delta.transpose((1, 2, 0)), candidate_states[1:], viterbi_path[::-1]

    def forward_evaluate(self, words: Sequence[str], channel_model: Channel,
                         candidate_generator: CandidateGenerator,
                         num_candidates: int, test: bool = True) -> Tuple[np.ndarray, List[List[str]]]:
        """
        Compute forward probabilities for an observed error word sequence. Correct words as produced by
        candidate generator are hidden states.
        :param words: Sequence of error words.
        :param channel_model: Channel model that return p(error word | correct word) score.
        :param candidate_generator: Generator of correct word candidates for an input error word.
        :param num_candidates: Number of candidates to generate using `candidate_generator`.
        :param test: Whether this is testing and the method can be verbose.
        :return: Forward probabilities and the list of lists of correct word candidates, one for each cell in
            the forward probabilities matrix.
        """

        T = len(words)
        # num_candidates + 1 is due to "error word" itself possibly being an extra candidate
        # add 2 timesteps for eos observation and the storage of the final value
        shape = (T + 2), (num_candidates + 1), (num_candidates + 1)
        alpha = np.full(shape, LARGE_NEG_CONST)  # each state is one context
        candidate_states: List[List[str]] = [[KEN_BOS]]
        candidate_states_idxes: List[range] = [range(1)]

        # 2 X (num_candidates + 1) X (num_candidates + 1)
        kenlm_states = tuple(
            tuple(
                tuple(kenlm.State() for _ in range(num_candidates + 1))
                for _ in range(num_candidates + 1))
            for _ in range(2))

        # INITIALIZATION
        candidates = list(candidate_generator.generate(words[0], num_candidates=num_candidates))
        start_state = kenlm.State()
        self.model.BeginSentenceWrite(start_state)
        _, channel_scores = zip(*channel_model.bulkscore(candidates, words[0]))
        cand_idxes = range(len(candidates))
        alpha[0, cand_idxes, 0] = \
            [self.model.BaseScore(start_state, candidate, kenlm_states[0][k][0])
             for k, candidate in enumerate(candidates)]
        alpha[0, cand_idxes, 0] += channel_scores
        candidate_states.append(candidates)
        candidate_states_idxes.append(cand_idxes)

        # RECURSION
        for t, word in enumerate(words[1:], start=1):
            states_from = kenlm_states[int(t % 2 == 0)]
            states_to = kenlm_states[int(t % 2 == 1)]
            candidates = list(candidate_generator.generate(word, num_candidates=num_candidates))
            _, channel_scores = zip(*channel_model.bulkscore(candidates, word))
            cmin2_idxes = candidate_states_idxes[-2]
            cmin1_idxes = candidate_states_idxes[-1]
            cand_idxes = range(len(candidates))
            # 1 X j X i
            alpha_tmin1 = alpha[np.ix_([t - 1], cmin1_idxes, cmin2_idxes)]
            # k X j X i
            transition_scores = alpha_tmin1 + [
                    [
                        [self.model.BaseScore(states_from[j][i], candidate, states_to[k][j])
                            for i in cmin2_idxes]
                    for j in cmin1_idxes]
                for k, candidate in enumerate(candidates)]
            # 1 X k X j <= k X 1 + k X j
            alpha[np.ix_([t], cand_idxes, cmin1_idxes)] = np.array(channel_scores)[:, np.newaxis] + \
                logsumexp10(transition_scores, axis=2)
            candidate_states.append(candidates)
            candidate_states_idxes.append(cand_idxes)

        # TERMINATION
        state_k = kenlm.State()
        states_from = kenlm_states[int(T % 2 == 0)]
        cmin2_idxes = candidate_states_idxes[-2]
        cmin1_idxes = candidate_states_idxes[-1]
        # j X i
        transition_scores = alpha[np.ix_([T - 1], cmin1_idxes, cmin2_idxes)].squeeze(axis=0) + \
            [[self.model.BaseScore(states_from[j][i], KEN_EOS, state_k) for i in cmin2_idxes] for j in cmin1_idxes]
        alpha[T, 0, cmin1_idxes] = logsumexp10(transition_scores, axis=1)  # emission score 0

        alpha[T + 1, 0, 0] = logsumexp10(alpha[T, 0, :])
        # ... transpose since we want to retrieve [0, 0, -1]-th element
        return alpha.transpose((1, 2, 0)), candidate_states[1:]


    def forward_and_viterbi_evaluate(self, words: Sequence[str], channel_model: Channel,
                                     candidate_generator: CandidateGenerator,
                                     num_candidates: int, test: bool = True) -> \
            Tuple[np.ndarray, List[List[str]], List[range], np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Compute both simultaneously forward probabilities for an observed error word sequence and the viterbi path.
        Correct words as produced by candidate generator are hidden states.
        :param words: Sequence of error words.
        :param channel_model: Channel model that return p(error word | correct word) score.
        :param candidate_generator: Generator of correct word candidates for an input error word.
        :param num_candidates: Number of candidates to generate using `candidate_generator`.
        :param test: Whether this is testing and the method can be verbose.
        :return: Forward probabilities and the list of lists of correct word candidates, one for each cell in
            the forward probabilities matrix.
        """

        T = len(words)
        # num_candidates + 1 is due to "error word" itself possibly being an extra candidate
        # add 2 timesteps for eos observation and the storage of the final value
        shape = (T + 2), (num_candidates + 1), (num_candidates + 1)
        delta = np.full(shape, LARGE_NEG_CONST)  # each state is one context
        alpha = np.full(shape, LARGE_NEG_CONST)
        psi = np.full(shape, 0, dtype=np.uint32)  # traceback
        candidate_states: List[List[str]] = [[KEN_BOS]]
        candidate_states_idxes: List[range] = [range(1)]

        # channel and transition scores to pass on to backward evaluation
        full_channel_scores = np.full((T, num_candidates + 1), LARGE_NEG_CONST)
        # t X k X j X i
        full_transitions = np.full((T + 1, num_candidates + 1, num_candidates + 1, num_candidates + 1), LARGE_NEG_CONST)

        # 2 X (num_candidates + 1) X (num_candidates + 1)
        kenlm_states = tuple(
            tuple(
                tuple(kenlm.State() for _ in range(num_candidates + 1))
                for _ in range(num_candidates + 1))
            for _ in range(2))

        # INITIALIZATION
        candidates = list(candidate_generator.generate(words[0], num_candidates=num_candidates))
        start_state = kenlm.State()
        self.model.BeginSentenceWrite(start_state)
        _, channel_scores = zip(*channel_model.bulkscore(candidates, words[0]))
        cand_idxes = range(len(candidates))
        channel_scores = np.array(channel_scores)
        transitions = [self.model.BaseScore(start_state, candidate, kenlm_states[0][k][0])
                       for k, candidate in enumerate(candidates)]
        scores = channel_scores + transitions
        delta[0, cand_idxes, 0] = scores
        alpha[0, cand_idxes, 0] = scores
        candidate_states.append(candidates)
        candidate_states_idxes.append(cand_idxes)
        full_channel_scores[0, cand_idxes] = channel_scores
        full_transitions[0, cand_idxes, 0, 0] = transitions

        # RECURSION
        for t, word in enumerate(words[1:], start=1):
            states_from = kenlm_states[int(t % 2 == 0)]
            states_to = kenlm_states[int(t % 2 == 1)]
            candidates = list(candidate_generator.generate(word, num_candidates=num_candidates))
            _, channel_scores = zip(*channel_model.bulkscore(candidates, word))
            cmin2_idxes = candidate_states_idxes[-2]
            cmin1_idxes = candidate_states_idxes[-1]
            cand_idxes = range(len(candidates))
            full_channel_scores[t, cand_idxes] = channel_scores
            # k X 1
            channel_scores = np.array(channel_scores)[:, np.newaxis]
            # 1 X j X i
            tmin1_j_i_idx = np.ix_([t - 1], cmin1_idxes, cmin2_idxes)
            delta_tmin1 = delta[tmin1_j_i_idx]
            alpha_tmin1 = alpha[tmin1_j_i_idx]
            # k X j X i
            transitions = [
                [
                    [self.model.BaseScore(states_from[j][i], candidate, states_to[k][j])
                        for i in cmin2_idxes]
                    for j in cmin1_idxes]
                for k, candidate in enumerate(candidates)]
            full_transitions[np.ix_([t], cand_idxes, cmin1_idxes, cmin2_idxes)] = transitions
            delta_transition_scores = delta_tmin1 + transitions
            alpha_transition_scores = alpha_tmin1 + transitions
            # k X j
            max_i = delta_transition_scores.argmax(axis=2)
            max_i_idxes = np.array(cand_idxes)[:, np.newaxis], cmin1_idxes, max_i
            # 1 X k X j
            t_k_j_idx = np.ix_([t], cand_idxes, cmin1_idxes)
            delta[t_k_j_idx] = channel_scores + delta_transition_scores[max_i_idxes]
            alpha[t_k_j_idx] = channel_scores + logsumexp10(alpha_transition_scores, axis=2)
            psi[t_k_j_idx] = max_i

            candidate_states.append(candidates)
            candidate_states_idxes.append(cand_idxes)

        # TERMINATION
        state_k = kenlm.State()
        states_from = kenlm_states[int(T % 2 == 0)]
        cmin2_idxes = candidate_states_idxes[-2]
        cmin1_idxes = candidate_states_idxes[-1]
        # 1 X j X i => j X i
        tmin1_j_i_idx = np.ix_([T - 1], cmin1_idxes, cmin2_idxes)
        delta_tmin1 = delta[tmin1_j_i_idx].squeeze(axis=0)
        alpha_tmin1 = alpha[tmin1_j_i_idx].squeeze(axis=0)
        # j X i
        transitions = [
                [self.model.BaseScore(states_from[j][i], KEN_EOS, state_k) for i in cmin2_idxes]
            for j in cmin1_idxes]
        full_transitions[np.ix_([T], [0], cmin1_idxes, cmin2_idxes)] = transitions
        delta_transition_scores = delta_tmin1 + transitions
        alpha_transition_scores = alpha_tmin1 + transitions
        # j
        max_i = delta_transition_scores.argmax(axis=1)
        delta[T, 0, cmin1_idxes] = delta_transition_scores[(cmin1_idxes, max_i)]  # emission score 0
        psi[T, 0, cmin1_idxes] = max_i
        alpha[T, 0, cmin1_idxes] = logsumexp10(alpha_transition_scores, axis=1)  # emission score 0

        # TRACEBACK
        max_j = np.argmax(delta[T, 0, :])
        delta[T + 1, 0, 0] = delta[T, 0, max_j]  # unnecessary

        alpha[T + 1, 0, 0] = logsumexp10(alpha[T, 0, :])

        max_i = psi[T, 0, max_j]
        viterbi_path: List[str] = [candidate_states[T][max_j]]
        # `candidate_states` has [KEN_BOS] at index 0 therefore all indices into it are + 1
        for t in range(T - 1, 0, -1):
            viterbi_path.append(candidate_states[t][max_i])
            l = psi[t, max_j, max_i]
            max_j = max_i
            max_i = l

        # ... transpose since we want to retrieve [0, 0, -1]-th element
        return alpha.transpose((1, 2, 0)), candidate_states[1:], candidate_states_idxes[1:], \
               full_channel_scores, full_transitions, \
               delta.transpose((1, 2, 0)), viterbi_path[::-1]


    def fbackward_evaluate(self, words: Sequence[str], channel_model: Channel,
                          candidate_states: Sequence[Sequence[str]],
                          num_candidates: int, test: bool = True) -> np.ndarray:
        """
        Compute backward probabilities for an observed error word sequence. Correct words as produced by
        candidate generator are hidden states.

        @TODO N.B.: beta_t corresponds to alpha_{t-1} and word t-1.

        :param words: Sequence of error words.
        :param channel_model: Channel model that return p(error word | correct word) score.
        :param candidate_states: Lists of candidates computed for each error word.
        :param num_candidates: Maximum possible number of candidates generated per error word.
        :param test: Whether this is testing and the method can be verbose.
        :return: Backward probabilities and the list of lists of correct word candidates, one for each cell in
            the backward probabilities matrix.
        """

        T = len(words)
        beta = np.full((num_candidates + 1, num_candidates + 1, T + 1), LARGE_NEG_CONST)  # each state is one previous word

        # INITIALIZATION
        T_minus3_state = kenlm.State()
        self.model.NullContextWrite(T_minus3_state)  # null (=irrelevant) context for T_minus3_state: state previous to j.
        T_minus2_state = kenlm.State()
        T_minus1_state = kenlm.State()
        end_state = kenlm.State()
        for j, candidate_j in enumerate(candidate_states[T - 2] if T > 1 else [KEN_BOS]):
            _ = self.model.BaseScore(T_minus3_state, candidate_j, T_minus2_state)
            for k, candidate_k in enumerate(candidate_states[T - 1] if T > 0 else [KEN_BOS]):
                _ = self.model.BaseScore(T_minus2_state, candidate_k, T_minus1_state)  # now T_minus1_state has right context
                beta[k, j, T] = self.model.BaseScore(T_minus1_state, KEN_EOS, end_state)  # p(<eos> | candidate_k, candidate_j) + emit <eos>

        # RECURSION
        null_context_state = kenlm.State()
        self.model.NullContextWrite(null_context_state)
        state_t = kenlm.State()
        state_t_minus1 = kenlm.State()
        state_t_plus1 = kenlm.State()
        for t in range(T - 2, -2, -1):
            candidates_scores_i_t_plus1 = list(channel_model.bulkscore(candidate_states[t + 1], words[t + 1]))
            for j, candidate_j in enumerate(candidate_states[t - 1] if t > 0 else [KEN_BOS]):
                _ = self.model.BaseScore(null_context_state, candidate_j, state_t_minus1)
                for k, candidate_k in enumerate(candidate_states[t] if t >= 0 else [KEN_BOS]):
                    # set `state_t` to `candidate_k, candidate_j` context
                    _ = self.model.BaseScore(state_t_minus1, candidate_k, state_t)
                    all_transition_proba = []
                    for i, (candidate_i, channel_score_i_t_plus1) in enumerate(candidates_scores_i_t_plus1):
                        all_transition_proba.append(
                            self.model.BaseScore(state_t, candidate_i, state_t_plus1) +  # p(s_{t+1}=cand_{i,k} | s_t=cand_{k,j})
                            channel_score_i_t_plus1 +  # emit p(words[t+1] | s_{t+1}=cand_{i,k})
                            beta[i, k, t + 2])  # all backward (=future) probability mass at state s_{t+1}=cand_{i,k}
                    beta[k, j, t + 1] = logsumexp10(all_transition_proba)
        return beta

    def backward_evaluate(self, candidate_state_indices: List[range],
                          full_channel_scores: np.ndarray, full_transitions: np.ndarray) -> np.ndarray:
        """
        Compute backward probabilities for an observed error word sequence. Correct words as produced by
        candidate generator are hidden states.

        @TODO N.B.: beta_t corresponds to alpha_{t-1} and word t-1.

        :param candidate_state_indices: Lists of range objects for each input word.
        :param full_channel_scores: Channel scores p(word | candidate) for each word and candidate.
        :param full_transitions: Transition scores p(s_{candidate_t} | s_{candidate_t-2, candidate_t-1}) for each t
            (indexed by words) plus final state s_{KEN_EOS}.
        :return: Backward probabilities matrix.
        """

        T, num_candidates_plus1 = full_channel_scores.shape

        # each state is one previous word
        beta = np.full((T + 1, num_candidates_plus1, num_candidates_plus1), LARGE_NEG_CONST)

        # INITIALIZATION
        cmin1_idxes = candidate_state_indices[T - 1] if T > 0 else [0]  # k
        cmin2_idxes = candidate_state_indices[T - 2] if T > 1 else [0]  # j
        # 1 X k X j
        # p(<eos> | candidate_k, candidate_j) + emit <eos>
        beta[np.ix_([T], cmin1_idxes, cmin2_idxes)] = \
            full_transitions[np.ix_([T], [0], cmin1_idxes, cmin2_idxes)].squeeze(axis=1)

        # RECURSION
        for t in range(T - 2, -2, -1):
            cmin1_idxes = candidate_state_indices[t - 1] if t > 0 else [0]  # j
            cand_idxes = candidate_state_indices[t] if t > -1 else [0]  # k
            cplus1_idxes = candidate_state_indices[t + 1]  # i
            # i
            channel_scores = full_channel_scores[t + 1, cplus1_idxes]
            # 1 X i X k X j
            transitions = full_transitions[np.ix_([t + 1], cplus1_idxes, cand_idxes, cmin1_idxes)]
            # 1 X i X k
            beta_tplus2 = beta[np.ix_([t + 2], cplus1_idxes, cand_idxes)]
            # 1 X k X j = 1 X i X 1 X 1 + 1 X i k X j + 1 X i X k X 1
            beta[np.ix_([t + 1], cand_idxes, cmin1_idxes)] = \
                logsumexp10(
                    channel_scores[np.newaxis, :, np.newaxis, np.newaxis] +  # emit p(words[t+1] | s_{t+1}=cand_{i,k})
                    transitions +  # transition p(s_{t+1}=cand_{i,k} | s_t=cand_{k,j})
                    beta_tplus2[..., np.newaxis],  # all backward (=future) probability mass at state s_{t+1}=cand_{i,k}
                    axis=1
                )
        return beta.transpose((1, 2, 0))

    def _compute_weight(self, alpha: np.ndarray, beta: np.ndarray, test: bool = True) -> np.ndarray:
        """
        Compute posterior P(s_{t}=candidate_i | words) as sum_k P(s_{t}=(candidate_i, candidate_k) | words) = sum_k

            alpha[t, i, k] * beta[t + 1, i, k] / sum_{j, k}  alpha[t, j, k] * beta[t + 1, j, k]

        @TODO Recall that alpha[-, t] refers the same timestep as beta[-, t + 1]

        This quantity also goes into computation of the (joint probability) weight for this word type and
        candidate word type since

            P(candidate_i, word) \propto sum_{t=1}^T P(s_{t}=candidate_i | words) * 1_{words_t == word}

        We ignore normalization by P(word).
        :param alpha: Matrix of forward probabilities.
        :param beta: Matrix of backward probabilities.
        :param test: Whether this is testing and the method can be verbose.
        :return: Weights for each error word and each of its candidates.
        """
        I, K, T = beta.shape

        gamma = np.full((I, K, T - 1), LARGE_NEG_CONST)  # ignore end-of-sequence word ...

        for t in range(T - 1):
            gamma[..., t] = alpha[..., t] + beta[..., t + 1]
            gamma[..., t] -= logsumexp10(gamma[..., t])

        out = np.power(10., gamma).sum(axis=1)
        assert np.allclose(out.sum(axis=0), np.ones(T - 1)), out.sum(axis=0)
        return out

    def compute_weight(self, words: Sequence[str], channel: Channel,
                       candidate_generator: CandidateGenerator, num_candidates: int,
                       predict=True, test=False) -> Union[Tuple[np.ndarray, List[List[str]]],
                                                          Tuple[np.ndarray, List[List[str]], Tuple[List[str], float]]]:
        """
        Compute posterior weights for each error word and each of its candidates.
        :param words: Sequence of error words.
        :param channel: Channel.
        :param candidate_generator: Candidate generator.
        :param num_candidates: Number of candidates to generate.
        :param predict: Whether to additionally return the viterbi path.
        :param test: Whether this is testing and the method can be verbose.
        :return: Weights for each error word and each of its candidates as well as the full ca.
            Optionally, viterbi path and its score.
        """
        if predict:

            (alpha, candidate_states, candidate_state_idxes, full_channel_scores, full_transitions,
                viter, viterbi_path) = \
                self.forward_and_viterbi_evaluate(words, channel, candidate_generator, num_candidates, test=test)

            beta = self.backward_evaluate(candidate_state_idxes, full_channel_scores, full_transitions)

            if test:
                # verify the scores
                test_viter, _, test_viterbi_path = self.viterbi_evaluate(words, channel,
                    candidate_generator, num_candidates, test=test)
                assert np.allclose(test_viter, viter)
                assert viterbi_path == test_viterbi_path

            viterbi = viterbi_path, viter[0, 0, -1]

        else:

            raise NotImplementedError('Requires fast backward evaluation.')

        assert np.isclose(alpha[0, 0, -1], beta[0, 0, 0]), (alpha[0, 0, -1], '!=', beta[0, 0, 0])
        weights = self._compute_weight(alpha, beta, test=False)

        if predict:
            output = weights, candidate_states, viterbi
        else:
            output = weights, candidate_states
        return output

    def k_best_viterbi_evaluate(self, words: Sequence[str], channel_model: Channel,
                                candidate_generator: CandidateGenerator,
                                num_candidates: int, topK: int, test: bool = True) -> \
            Tuple[np.ndarray, List[List[str]], List[List[str]], List[float]]:
        """
        Compute k-best Viterbi sequences.
        :param words: Sequence of error words.
        :param channel_model: Channel model that return p(error word | correct word) score.
        :param candidate_generator: Generator of correct word candidates for an input error word.
        :param num_candidates: Number of candidates to generate using `candidate_generator`.
        :param test: Whether this is testing and the method can be verbose.
        :return: Viterbi probabilities, the list of lists of correct word candidates, one for each cell in
            the viterbi probabilities matrix, and the viterbi path.
        """

        if topK == 1:
            delta, candidate_states, viterbi_path = self.viterbi_evaluate(words, channel_model, candidate_generator,
                                                                          num_candidates, test)
            return delta, candidate_states, [viterbi_path], [delta[0, 0, -1]]

        T = len(words)
        # num_candidates + 1 is due to "error word" itself possibly being an extra candidate
        # add 2 timesteps for eos observation and the storage of the final value
        shape = (num_candidates + 1), (num_candidates + 1), topK, (T + 2)
        delta = np.full(shape, LARGE_NEG_CONST)  # each state is one context
        contexts: List[List[IJContainer]] = []
        candidate_states: List[List[str]] = []

        # predecessor state for k best path that passes through state i and word t: <word number t, state i, rank k>
        psi = np.full(shape, 0, dtype=np.uint32)  # default is initial state

        # rank (0...topK) of predecessor state for k best path that passes through state i and word t
        rank = np.full(shape, -1, dtype=np.int16)  # default is no rank

        # INITIALIZATION
        candidates = candidate_generator.generate(words[0], num_candidates=num_candidates)
        start_state = kenlm.State()
        self.model.BeginSentenceWrite(start_state)
        new_contexts: List[IJContainer] = []
        for k, (candidate, channel_score) in enumerate(channel_model.bulkscore(candidates, words[0])):
            state_0 = kenlm.State()
            delta[k, 0, 0, 0] = channel_score + self.model.BaseScore(start_state, candidate, state_0)
            future_ij_contexts: List[IJContext] = [IJContext(KEN_BOS, [state_0])]
            new_contexts.append(IJContainer(candidate, future_ij_contexts))
        contexts.append(new_contexts)
        candidate_states.append(list(candidates))

        for t, word in enumerate(words[1:], start=1):
            candidates = candidate_generator.generate(word, num_candidates=num_candidates)
            new_contexts: List[IJContainer] = []
            for k, (candidate, channel_score) in enumerate(channel_model.bulkscore(candidates, word)):
                future_ij_contexts: List[IJContext] = []
                for j, ij_container in enumerate(contexts[-1]):
                    state_jk = kenlm.State()  # state_jk is the emission state
                    min_heap = []
                    for i, context in enumerate(ij_container.contexts):
                        for r, state_ij in enumerate(context.state_ij):
                            # state_ij stores bigram-state context (i, j), one for each rank r
                            score = delta[j, i, r, t - 1] + self.model.BaseScore(state_ij, candidate, state_jk)
                            heapq.heappush(min_heap, (-score, i, state_jk))  # N.B. **minus** score
                    # keep a ranking if a path crosses a state more than once
                    rank_dict = dict()
                    jk_kenlm_states: List[kenlm.State] = []
                    # get topK best previous states out
                    for r in range(min(topK, len(min_heap))):
                        neg_score, i, state_jk = heapq.heappop(min_heap)
                        delta[k, j, r, t] = channel_score - neg_score  # N.B. undo **minus** score
                        psi[k, j, r, t] = i
                        jk_kenlm_states.append(state_jk)
                        if i in rank_dict:
                            rank_dict[i] += 1
                        else:
                            rank_dict[i] = 0
                        rank[k, j, r, t] = rank_dict[i]  # passing through i on jk's r-th ranked path
                    future_ij_contexts.append(IJContext(ij_container.word_j, jk_kenlm_states))
                new_contexts.append(IJContainer(candidate, future_ij_contexts))
            contexts.append(new_contexts)
            candidate_states.append(list(candidates))

        # TERMINATION
        Tplus1_heap = []
        state_jk = kenlm.State()
        for j, ij_container in enumerate(contexts[-1]):
            min_heap = []
            for i, context in enumerate(ij_container.contexts):
                for r, state_ij in enumerate(context.state_ij):
                    score = delta[j, i, r, T - 1] + self.model.BaseScore(state_ij, KEN_EOS, state_jk)
                    heapq.heappush(min_heap, (-score, i))  # N.B. **minus** score
            rank_dict = dict()
            for r in range(min(topK, len(min_heap))):
                neg_score, i = heapq.heappop(min_heap)
                delta[0, j, r, T] = - neg_score  # N.B. undo **minus** score ; emission factor equals 0
                psi[0, j, r, T] = i
                if i in rank_dict:
                    rank_dict[i] += 1
                else:
                    rank_dict[i] = 0
                rank[0, j, r, T] = rank_dict[i]  # passing through i on jk's r-th ranked path
                heapq.heappush(Tplus1_heap, (neg_score, j, r))

        # TRACEBACK topK paths
        paths: List[List[str]] = []
        scores: List[float] = []
        for r in range(min(topK, len(Tplus1_heap))):
            # The maximum probability and the state it came from
            neg_score, j, rankK = heapq.heappop(Tplus1_heap)
            delta[0, 0, r, T + 1] = delta[0, j, rankK, T]  # unnecessary

            scores.append(delta[0, j, rankK, T])
            rankK_path: List[str] = [contexts[T - 1][j].word_j]

            i = psi[0, j, rankK, T]
            rankK = rank[0, j, rankK, T]
            for t in range(T - 2, -1, -1):
                rankK_path.append(contexts[t][i].word_j)
                l = psi[j, i, rankK, t + 1]
                rankK = rank[j, i, rankK, t + 1]
                j = i
                i = l
            paths.append(rankK_path[::-1])
        return delta.transpose((1, 2, 3, 0)), candidate_states, paths, scores
