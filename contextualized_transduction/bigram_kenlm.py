from typing import Sequence, Tuple, ClassVar, Union, List

import kenlm
import numpy as np
import heapq

from contextualized_transduction.candidate_generator import CandidateGenerator
from contextualized_transduction.lm import KenLM
from contextualized_transduction.sed_channel import Channel
from contextualized_transduction.utils import LARGE_NEG_CONST, logsumexp10

KEN_EOS = "</s>"


# Example:
# ========
#
# #Stateful query
# state = kenlm.State()
# state2 = kenlm.State()
# #Use <s> as context.  If you don't want <s>, use model.NullContextWrite(state).
# model.BeginSentenceWrite(state)
# accum = 0.0
# accum += model.BaseScore(state, "a", state2)
# accum += model.BaseScore(state2, "sentence", state)
# #score defaults to bos = True and eos = True.  Here we'll check without the end
# #of sentence marker.
# assert (abs(accum - model.score("a sentence", eos = False)) < 1e-3)
# accum += model.BaseScore(state, "</s>", state2)
# assert (abs(accum - model.score("a sentence")) < 1e-3)


class BigramKenLM(KenLM):
    ORDER: ClassVar[int] = 2

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.model.order == self.ORDER, 'Wrong n-gram order: {0} vs {1}'.format(
            self.model.order - 1, self.ORDER - 1)

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
        alpha = np.full((num_candidates + 1, T + 1), LARGE_NEG_CONST)  # each state is one previous word
        candidate_states: List[List[str]] = []
        previous_kenlm_states: List[kenlm.State] = []

        # INITIALIZATION
        candidates = candidate_generator.generate(words[0], num_candidates=num_candidates)
        start_state = kenlm.State()
        self.model.BeginSentenceWrite(start_state)  # NB!
        for k, (candidate, channel_score) in enumerate(channel_model.bulkscore(candidates, words[0])):
            # each candidate is a STATE k
            state_0 = kenlm.State()
            alpha[k, 0] = channel_score + self.model.BaseScore(start_state, candidate, state_0)  # p(candidate | <bos>)
            if test:
                print('Transiting to (%s) and emitting (%s): %.3f' % (candidate, words[0], alpha[k, 0]))
            previous_kenlm_states.append(state_0)
        candidate_states.append(list(candidates))

        # RECURSION
        for t, word in enumerate(words[1:], start=1):
            new_previous_kenlm_states: List[kenlm.State] = []
            candidates = candidate_generator.generate(word, num_candidates=num_candidates)
            for k, (candidate, channel_score) in enumerate(channel_model.bulkscore(candidates, word)):
                # current STATE k
                all_transition_proba = []
                state_k = kenlm.State()
                for i, previous_candidate in enumerate(candidate_states[-1]):
                    state_i = previous_kenlm_states[i]
                    all_transition_proba.append(alpha[i, t - 1] + self.model.BaseScore(state_i, candidate, state_k))
                    if test:
                        print('Transition: (%s) => (%s): %.3f' % (
                            previous_candidate, candidate, all_transition_proba[-1]))
                # does not matter which state_i it was, important is that state_k emitted "candidate"
                new_previous_kenlm_states.append(state_k)
                alpha[k, t] = channel_score + logsumexp10(all_transition_proba)
                if test:
                    print('Transiting to (%s) and emitting (%s): %.3f' % (candidate, word, alpha[k, t]))
            previous_kenlm_states = new_previous_kenlm_states
            candidate_states.append(list(candidates))

        # TERMINATION
        all_transition_proba = []
        state_k = kenlm.State()
        for i, previous_candidate in enumerate(candidate_states[-1]):
            state_i = previous_kenlm_states[i]
            all_transition_proba.append(alpha[i, T - 1] + self.model.BaseScore(state_i, KEN_EOS, state_k))
        alpha[0, T] = logsumexp10(all_transition_proba)  # emit only KEN_EOS hence emission factor equals 0.
        return alpha, candidate_states

    def forward_and_viterbi_evaluate(self, words: Sequence[str], channel_model: Channel,
                                     candidate_generator: CandidateGenerator,
                                     num_candidates: int, test: bool = True) -> \
            Tuple[np.ndarray, List[List[str]], np.ndarray, List[str]]:
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
        alpha = np.full((num_candidates + 1, T + 1), LARGE_NEG_CONST)  # each state is one previous word
        delta = np.full((num_candidates + 1, T + 1), LARGE_NEG_CONST)
        candidate_states: List[List[str]] = []
        previous_kenlm_states: List[kenlm.State] = []

        # INITIALIZATION
        candidates = candidate_generator.generate(words[0], num_candidates=num_candidates)
        start_state = kenlm.State()
        self.model.BeginSentenceWrite(start_state)  # NB!
        for k, (candidate, channel_score) in enumerate(channel_model.bulkscore(candidates, words[0])):
            # each candidate is a STATE k
            state_0 = kenlm.State()
            score = channel_score + self.model.BaseScore(start_state, candidate, state_0)  # p(candidate | <bos>)
            alpha[k, 0] = score
            delta[k, 0] = score
            if test:
                print('Transiting to (%s) and emitting (%s): %.3f' % (candidate, words[0], alpha[k, 0]))
            previous_kenlm_states.append(state_0)
        candidate_states.append(list(candidates))

        try:
            # RECURSION
            for t, word in enumerate(words[1:], start=1):
                new_previous_kenlm_states: List[kenlm.State] = []
                candidates = candidate_generator.generate(word, num_candidates=num_candidates)
                for k, (candidate, channel_score) in enumerate(channel_model.bulkscore(candidates, word)):
                    # current STATE k
                    all_transition_proba = []
                    all_viterbi_transition_proba = []
                    state_k = kenlm.State()
                    for i, previous_candidate in enumerate(candidate_states[-1]):
                        state_i = previous_kenlm_states[i]
                        transition_score = self.model.BaseScore(state_i, candidate, state_k)
                        all_transition_proba.append(alpha[i, t - 1] + transition_score)
                        all_viterbi_transition_proba.append(delta[i, t - 1] + transition_score)
                        if test:
                            print('Transition: (%s) => (%s): %.3f' % (
                                previous_candidate, candidate, all_transition_proba[-1]))
                    # does not matter which state_i it was, important is that state_k emitted "candidate"
                    new_previous_kenlm_states.append(state_k)
                    alpha[k, t] = channel_score + logsumexp10(all_transition_proba)
                    delta[k, t] = channel_score + max(all_viterbi_transition_proba)
                    if test:
                        print('Transiting to (%s) and emitting (%s): %.3f' % (candidate, word, alpha[k, t]))
                previous_kenlm_states = new_previous_kenlm_states
                candidate_states.append(list(candidates))
        except Exception as e:
            print('words: ', words)
            raise e

        # TERMINATION
        all_transition_proba = []
        all_viterbi_transition_proba = []
        state_k = kenlm.State()
        for i, previous_candidate in enumerate(candidate_states[-1]):
            state_i = previous_kenlm_states[i]
            transition_score = self.model.BaseScore(state_i, KEN_EOS, state_k)
            all_transition_proba.append(alpha[i, T - 1] + transition_score)
            all_viterbi_transition_proba.append(delta[i, T - 1] + transition_score)
        alpha[0, T] = logsumexp10(all_transition_proba)  # emit only KEN_EOS hence emission factor equals 0.
        delta[0, T] = max(all_viterbi_transition_proba)  # emit only KEN_EOS hence emission factor equals 0.

        # TRACEBACK
        s_t_star = np.argmax(all_viterbi_transition_proba)  # (T - 1)'s best state
        viterbi_candidate = candidate_states[T - 1][s_t_star]
        viterbi_path: List[str] = [viterbi_candidate]  # ... reversed
        null_context_state = kenlm.State()
        self.model.NullContextWrite(null_context_state)  # no problem for t = 0 as only p(t=1 | t=0) transition matters.
        for t in range(T - 2, -1, -1):
            state_t_plus1 = kenlm.State()  # its word is last on `viterbi_path`
            alternatives = []
            for i, candidate in enumerate(candidate_states[t]):
                state_t = kenlm.State()  # state_t will hold context `candidate`
                _ = self.model.BaseScore(null_context_state, candidate, state_t)
                alternatives.append(delta[i, t] + self.model.BaseScore(state_t, viterbi_candidate, state_t_plus1))
            s_t_star = np.argmax(alternatives)
            viterbi_candidate = candidate_states[t][s_t_star]
            viterbi_path.append(viterbi_candidate)

        return alpha, candidate_states, delta, viterbi_path[::-1]

    def backward_evaluate(self, words: Sequence[str], channel_model: Channel,
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
        beta = np.full((num_candidates + 1, T + 1), LARGE_NEG_CONST)  # each state is one previous word

        # INITIALIZATION
        T_minus2_state = kenlm.State()
        self.model.NullContextWrite(T_minus2_state)  # null context for T_minus1_state
        T_minus1_state = kenlm.State()
        end_state = kenlm.State()
        for k, candidate in enumerate(candidate_states[T - 1]):
            _ = self.model.BaseScore(T_minus2_state, candidate, T_minus1_state)  # now T_minus1_state has right context
            beta[k, T] = self.model.BaseScore(T_minus1_state, KEN_EOS, end_state)  # p(<eos> | candidate) + emit <eos>
            if test:
                print('Transiting from (%s) to <EOS>: %.3f' % (candidate, beta[k, T]))

        # RECURSION
        null_context_state = kenlm.State()
        self.model.NullContextWrite(null_context_state)
        state_t = kenlm.State()
        state_t_plus1 = kenlm.State()
        for t in range(T - 2, -1, -1):
            candidates_scores_i_t_plus1 = list(channel_model.bulkscore(candidate_states[t + 1], words[t + 1]))
            for k, candidate_k in enumerate(candidate_states[t]):
                # set `state_t` to `candidate_k` context
                _ = self.model.BaseScore(null_context_state, candidate_k, state_t)
                all_transition_proba = []
                for i, (candidate_i, channel_score_i_t_plus1) in enumerate(candidates_scores_i_t_plus1):
                    all_transition_proba.append(
                        self.model.BaseScore(state_t, candidate_i, state_t_plus1) +  # p(s_{t+1}=cand_i | s_t={cand_k})
                        channel_score_i_t_plus1 +  # emit p(words[t+1] | s_{t+1}=cand_i)
                        beta[i, t + 2])  # all backward (=future) probability mass at state s_{t+1}=cand_i
                    if test:
                        print('Transiting from (%s) to (%s): %.3f' %
                              (candidate_k, candidate_i, all_transition_proba[-1]))
                beta[k, t + 1] = logsumexp10(all_transition_proba)

        # TERMINATION
        all_transition_proba = []
        start_state = kenlm.State()
        self.model.BeginSentenceWrite(start_state)
        state_t_plus1 = kenlm.State()
        for i, (candidate_i, channel_score_i_t_plus1) in enumerate(
                channel_model.bulkscore(candidate_states[0], words[0])):
            all_transition_proba.append(
                self.model.BaseScore(start_state, candidate_i, state_t_plus1) +  # p(s_{t+1}=candidate_i | <bos>)
                channel_score_i_t_plus1 +  # emit p(words[t+1] | s_{t+1}=candidate_i)
                beta[i, 1])  # all backward (=future) probability mass at state s_{t+1}=candidate_i
        beta[0, 0] = logsumexp10(all_transition_proba)

        return beta

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
        delta = np.full((num_candidates + 1, T + 1), LARGE_NEG_CONST)  # each state is one previous word
        candidate_states: List[List[str]] = []
        previous_kenlm_states: List[kenlm.State] = []

        # INITIALIZATION
        candidates = candidate_generator.generate(words[0], num_candidates=num_candidates)
        start_state = kenlm.State()
        self.model.BeginSentenceWrite(start_state)  # NB!
        for k, (candidate, channel_score) in enumerate(channel_model.bulkscore(candidates, words[0])):
            # each candidate is a STATE k
            state_0 = kenlm.State()
            delta[k, 0] = channel_score + self.model.BaseScore(start_state, candidate, state_0)  # p(candidate | <bos>)
            if test:
                print('Transiting to (%s) and emitting (%s): %.3f' % (candidate, words[0], delta[k, 0]))
            previous_kenlm_states.append(state_0)
        candidate_states.append(list(candidates))

        # RECURSION
        for t, word in enumerate(words[1:], start=1):
            new_previous_kenlm_states: List[kenlm.State] = []
            candidates = candidate_generator.generate(word, num_candidates=num_candidates)
            for k, (candidate, channel_score) in enumerate(channel_model.bulkscore(candidates, word)):
                # current STATE k
                all_transition_proba = []
                state_k = kenlm.State()
                for i, previous_candidate in enumerate(candidate_states[-1]):
                    state_i = previous_kenlm_states[i]
                    all_transition_proba.append(delta[i, t - 1] + self.model.BaseScore(state_i, candidate, state_k))
                    if test:
                        print('Transition: (%s) => (%s): %.3f' % (
                        previous_candidate, candidate, all_transition_proba[-1]))
                # does not matter which state_i it was, important is that state_k emitted "candidate"
                new_previous_kenlm_states.append(state_k)
                delta[k, t] = channel_score + max(all_transition_proba)
                if test:
                    print('Transiting to (%s) and emitting (%s): %.3f' % (candidate, word, delta[k, t]))
            previous_kenlm_states = new_previous_kenlm_states
            candidate_states.append(list(candidates))

        # TERMINATION
        all_transition_proba = []
        state_k = kenlm.State()
        for i, previous_candidate in enumerate(candidate_states[-1]):
            state_i = previous_kenlm_states[i]
            all_transition_proba.append(delta[i, T - 1] + self.model.BaseScore(state_i, KEN_EOS, state_k))
        delta[0, T] = max(all_transition_proba)  # emit only KEN_EOS hence emission factor equals 0.

        # TRACEBACK
        s_t_star = np.argmax(all_transition_proba)  # (T - 1)'s best state
        viterbi_candidate = candidate_states[T - 1][s_t_star]
        viterbi_path: List[str] = [viterbi_candidate]  # ... reversed
        null_context_state = kenlm.State()
        self.model.NullContextWrite(null_context_state)  # no problem for t = 0 as only p(t=1 | t=0) transition matters.
        for t in range(T - 2, -1, -1):
            state_t_plus1 = kenlm.State()  # its word is last on `viterbi_path`
            alternatives = []
            for i, candidate in enumerate(candidate_states[t]):
                state_t = kenlm.State()  # state_t will hold context `candidate`
                _ = self.model.BaseScore(null_context_state, candidate, state_t)
                alternatives.append(delta[i, t] + self.model.BaseScore(state_t, viterbi_candidate, state_t_plus1))
            s_t_star = np.argmax(alternatives)
            viterbi_candidate = candidate_states[t][s_t_star]
            viterbi_path.append(viterbi_candidate)
        return delta, candidate_states, viterbi_path[::-1]

    def _compute_weight(self, alpha: np.ndarray, beta: np.ndarray, test: bool = True) -> np.ndarray:
        """
        Compute posterior P(s_{t}=candidate_i | words) as:

            alpha[i, t] * beta[i, t + 1] / sum_{j}  alpha[j, t] * beta[j, t + 1]

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
        K, T = alpha.shape
        gamma = np.full((K, T - 1), LARGE_NEG_CONST)  # ignore end-of-sequence word ...

        for t in range(T - 1):
            gamma[:, t] = alpha[:, t] + beta[:, t + 1]
            gamma[:, t] -= logsumexp10(gamma[:, t])

        return np.power(10., gamma)

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

            alpha, candidate_states, viter, viterbi_path = self.forward_and_viterbi_evaluate(words,
                channel, candidate_generator, num_candidates, test=test)

            if test:
                # verify the scores
                test_viter, _, test_viterbi_path = self.viterbi_evaluate(words, channel,
                    candidate_generator, num_candidates, test=test)
                assert np.allclose(test_viter, viter)
                assert viterbi_path == test_viterbi_path

            viterbi = viterbi_path, viter[0, -1]

        else:
            alpha, candidate_states = self.forward_evaluate(words, channel, candidate_generator, num_candidates,
                                                            test=test)
            viterbi = None

        beta = self.backward_evaluate(words, channel, candidate_states, num_candidates, test=test)
        assert np.isclose(alpha[0, -1], beta[0, 0]), (alpha[0, -1], '!=', beta[0, 0])
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

        T = len(words)
        shape = num_candidates + 1, T + 1, topK
        delta = np.full(shape, LARGE_NEG_CONST)  # each state is one previous word
        candidate_states: List[List[str]] = []
        previous_kenlm_states: List[List[kenlm.State]] = []

        # predecessor state for k best path that passes through state i and word t: <word number t, state i, rank k>
        psi = np.full(shape, 0, dtype=np.uint32)  # default is initial state

        # rank (0...topK) of predecessor state for k best path that passes through state i and word t
        rank = np.full(shape, -1, dtype=np.int16)  # default is no rank

        # INITIALIZATION
        candidates = candidate_generator.generate(words[0], num_candidates=num_candidates)
        start_state = kenlm.State()
        self.model.BeginSentenceWrite(start_state)  # NB!
        for k, (candidate, channel_score) in enumerate(channel_model.bulkscore(candidates, words[0])):
            # each candidate is a STATE k
            state_0 = kenlm.State()
            delta[k, 0, 0] = channel_score + self.model.BaseScore(start_state, candidate, state_0)  # p(cand. | <bos>)
            rank[k, 0, 0] = 0
            if test:
                print('Transiting to (%s) and emitting (%s): %.3f' % (candidate, words[0], delta[k, 0, 0]))
            previous_kenlm_states.append([state_0])
        candidate_states.append(list(candidates))

        # RECURSION
        for t, word in enumerate(words[1:], start=1):
            new_previous_kenlm_states: List[List[kenlm.State]] = []
            candidates = candidate_generator.generate(word, num_candidates=num_candidates)
            for k, (candidate, channel_score) in enumerate(channel_model.bulkscore(candidates, word)):
                min_heap = []
                for i, previous_candidate in enumerate(candidate_states[-1]):
                    state_k = kenlm.State()  # current STATE k
                    for r, state_i in enumerate(previous_kenlm_states[i]):  # ranked previous states
                        score = delta[i, t - 1, r] + self.model.BaseScore(state_i, candidate, state_k)
                        heapq.heappush(min_heap, (-score, i, state_k))  # N.B. **minus** score
                        if test:
                            print('Transition: (%s, %d) => (%s): %.3f' % (previous_candidate, r, candidate, score))

                # keep a ranking if a path crosses a state more than once
                rank_dict = dict()
                k_kenlm_states: List[kenlm.State] = []
                # get topK best previous states out
                for r in range(min(topK, len(min_heap))):
                    neg_score, i, state_k = heapq.heappop(min_heap)
                    delta[k, t, r] = channel_score - neg_score  # N.B. undo **minus** score
                    psi[k, t, r] = i
                    k_kenlm_states.append(state_k)
                    if i in rank_dict:
                        rank_dict[i] += 1
                    else:
                        rank_dict[i] = 0
                    rank[k, t, r] = rank_dict[i]  # passing through i on k's r-th ranked path
                    if test:
                        print('Transiting to (%s) and emitting (%s): %.3f' % (candidate, word, delta[k, t, r]))
                new_previous_kenlm_states.append(k_kenlm_states)
            candidate_states.append(list(candidates))
            previous_kenlm_states = new_previous_kenlm_states

        # TERMINATION
        min_heap = []
        state_k = kenlm.State()
        for i, previous_candidate in enumerate(candidate_states[-1]):
            for r, state_i in enumerate(previous_kenlm_states[i]):  # ranked previous states
                score = delta[i, T - 1, r] + self.model.BaseScore(state_i, KEN_EOS, state_k)
                heapq.heappush(min_heap, (-score, i, r))  # N.B. **minus** score, rank r !

        paths: List[List[str]] = []
        scores: List[float] = []
        # TRACEBACK topK paths
        for r in range(min(topK, len(min_heap))):
            # The maximum probability and the state it came from
            neg_score, i, rankK = heapq.heappop(min_heap)
            delta[0, T, r] = -neg_score  # unnecessary
            psi[0, T, r] = i  # unnecessary
            rankK_path: List[str] = [candidate_states[-1][i]]
            scores.append(delta[0, T, r])
            for t in range(T - 2, -1, -1):
                p = psi[i, t + 1, rankK]  # get previous state i at t + 1
                rankK_path.append(candidate_states[t][p])
                rankK = rank[i, t + 1, rankK]  # get the new rank
                i = p
            paths.append(rankK_path[::-1])
        return delta, candidate_states, paths, scores
