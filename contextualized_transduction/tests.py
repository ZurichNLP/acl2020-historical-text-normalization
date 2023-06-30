import unittest
import tempfile
import os

import numpy as np

from contextualized_transduction.bigram_kenlm import BigramKenLM
from contextualized_transduction.candidate_generator import CrudeMEDSGenerator
from contextualized_transduction.charspacer import CharSpacer
from contextualized_transduction.sed_channel import DummyChannel
from contextualized_transduction.trigram_kenlm import TrigramKenLM


class Tests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.bigram_kenlm = BigramKenLM("train2.apra")
        cls.trigram_kenlm = TrigramKenLM("train3.apra")
        cls.channel = DummyChannel()
        cls.candidate_generator = CrudeMEDSGenerator(**generator_params)
        cls.words = ["a", "b", "c"]
        cls.number_candidates = 3

    def test_bigram(self):
        candidate_states = self.candidate_generator.generate_candidate_state_space(
            self.words, self.number_candidates)
        alpha, candidate_states_ = self.bigram_kenlm.forward_evaluate(
            self.words, self.channel, self.candidate_generator,
            self.number_candidates, test=False)
        beta = self.bigram_kenlm.backward_evaluate(
            self.words, self.channel, candidate_states,
            self.number_candidates, test=False)
        delta, _, path = self.bigram_kenlm.viterbi_evaluate(
            self.words, self.channel, self.candidate_generator,
            self.number_candidates, test=False)
        lm_score = self.bigram_kenlm.model.score(' '.join(path), bos=True, eos=True)
        alpha_, candidate_states_, delta_, path_ = self.bigram_kenlm.forward_and_viterbi_evaluate(
            self.words, self.channel, self.candidate_generator, self.number_candidates, test=False)
        _, _, paths, scores = self.bigram_kenlm.k_best_viterbi_evaluate(
            self.words, self.channel, self.candidate_generator, self.number_candidates, topK=10, test=False)
        weights = self.bigram_kenlm._compute_weight(alpha, beta, test=False)
        weights_, _, (path__, _) = self.bigram_kenlm.compute_weight(
            self.words, self.channel, self.candidate_generator,
            self.number_candidates, predict=True, test=False)
        self.assertEqual(candidate_states, candidate_states_)
        self.assertTrue(np.isclose(alpha[0, -1], beta[0, 0]))
        self.assertTrue(np.isclose(lm_score, delta[0, -1]))
        self.assertTrue(np.allclose(alpha_, alpha))
        self.assertTrue(np.allclose(delta_, delta))
        self.assertEqual(paths[0], path)
        self.assertTrue(np.isclose(scores[0], delta[0, -1]))
        self.assertTrue(np.allclose(weights, weights_))

    def test_trigram(self):
        delta_ref, candidate_states_ref, path_ref = self.trigram_kenlm.fviterbi_evaluate(
            self.words, self.channel, self.candidate_generator, self.number_candidates, test=False)
        delta, candidate_states, path = self.trigram_kenlm.viterbi_evaluate(
            self.words, self.channel, self.candidate_generator, self.number_candidates, test=False)
        lm_score = self.trigram_kenlm.model.score(" ".join(path), bos=True, eos=True)
        alpha_ref, _ = self.trigram_kenlm.forward_evaluate(
            self.words, self.channel, self.candidate_generator, self.number_candidates, test=False)
        alpha, _, si, fc, ft, delta_, path_ = self.trigram_kenlm.forward_and_viterbi_evaluate(
            self.words, self.channel, self.candidate_generator, self.number_candidates, test=False)
        beta_ref = self.trigram_kenlm.fbackward_evaluate(
            self.words, self.channel, candidate_states_ref, self.number_candidates, test=False)
        beta = self.trigram_kenlm.backward_evaluate(si, fc, ft)
        _, _, paths, scores = self.trigram_kenlm.k_best_viterbi_evaluate(
            self.words, self.channel, self.candidate_generator, self.number_candidates, topK=10)
        weights = self.trigram_kenlm._compute_weight(alpha, beta)
        weights_, _, (path__, _) = self.trigram_kenlm.compute_weight(
            self.words, self.channel, self.candidate_generator, self.number_candidates,
            predict=True, test=False)
        self.assertTrue(np.isclose(lm_score, delta[0, 0, -1]))
        self.assertTrue(np.allclose(delta_ref, delta))
        self.assertTrue(np.allclose(delta_ref, delta_))
        self.assertEqual(path_ref, path)
        self.assertEqual(path_ref, path_)
        self.assertTrue(np.allclose(alpha_ref, alpha))
        self.assertTrue(np.allclose(beta_ref, beta))
        self.assertTrue(np.isclose(alpha[0, 0, -1], beta[0, 0, 0]))
        self.assertEqual(paths[0], path)
        self.assertTrue(np.isclose(scores[0], delta[0, 0, -1]))
        self.assertEqual(path, path__)
        self.assertTrue(np.allclose(weights, weights_))

    def test_charspacer(self):
        with open(os.path.join(os.path.dirname(__file__), "data/words.txt")) as f:
            words = [word.strip() for word in f]
        spacer = CharSpacer(words)
        temp_pkl = tempfile.mktemp()
        spacer.to_pickle(temp_pkl)
        spacer = CharSpacer.from_pickle(temp_pkl, words)
        word = 'vel'
        top_k = 50
        candidates = spacer.candidates(word=word, top_k=top_k, s=1.)
        self.assertSetEqual(
            {'vestibulum', 'venenatis', 'velit', 'vel', 'vehicula'},
            set(candidates)
        )


generator_params = dict(meds_fns=[os.path.join(os.path.dirname(__file__), "data/med.txt")],
                        language="german",
                        candidates_alphabet={"e", "t", "S", "a", "b", "d"},
                        indomain_candidates=None,
                        crudefilter_regex_candidate=r"[^\d\.,!?]",
                        crudefilter_maxlen=50, crudefilter_maxedit=50,
                        lowercase=False,
                        verbose=False)


if __name__ == '__main__':
    unittest.main()
