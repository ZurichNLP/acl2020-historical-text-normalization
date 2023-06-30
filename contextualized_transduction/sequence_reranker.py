import json
import abc
import editdistance
import numpy as np
import multiprocessing
import math
import time

from collections import Counter
from pathlib import Path
from typing import Union, Generator, List, Tuple, Optional

from contextualized_transduction.reranker import ScoreFunction, SimpleLanguageModel, Sample, direct_fn_reader, PROReranker,\
    PROClassifierReranker, SGDClassifier, NormalizedEditDistanceMargin, stepfunc

from nn_lm.custom_lm import CharLanguageModel, WordLanguageModel

class SequenceSample:
    def __init__(self, sent_num: int, origin_words: List[str], words: List[str], reference: List[str],
                 scores: List[float], paths: List[List[str]]):
        """
        Holds k-best predictions and their scores for a sentence.
        :param sent_num:
        :param origin_words:
        :param words:
        :param reference:
        :param scores:
        :param paths:
        """
        self.sent_num = sent_num
        self.origin_words = origin_words
        self.words = words
        self.reference = reference

        # SORTING BY CHAR LENGTH
        self.sorting_indices = np.argsort([len(' '.join(p)) for p in paths])[::-1]
        self.scores = [scores[idx] for idx in self.sorting_indices]
        self.paths = [paths[idx] for idx in self.sorting_indices]


class SequenceCollection:
    """
    A collection of SequenceSamples.
    """

    def __init__(self, sequences: List[SequenceSample]):
        self.sequences = sequences
        self.prediction_per_input, _ = Counter(len(s.paths) for s in self.sequences).most_common(1)[0]

    def __iter__(self):
        yield from self.sequences

    def __len__(self):
        return len(self.sequences)

    def shape(self):
        return self.__len__(), self.prediction_per_input

    def __getitem__(self, sliced):
        return self.sequences[sliced]


def data_reader(path2data: Union[str, Path], encoding: str = 'utf8') -> Generator[SequenceSample, None, None]:
    """
    Read k-best predictions.
    :param path2data: Path to dataset file.
    :param encoding: Encoding.
    """
    with Path(path2data).open(encoding=encoding) as f:
        for data_dict in json.load(f):
            yield SequenceSample(**data_dict)


class HammingLoss(ScoreFunction):
    def score(self, target: List[str], hypothesis: List[str]) -> float:
        """
        Each correct word gets a score of 1, each incorrect a score of 0.
        :param target: Target.
        :param hypothesis: Hypothesis.
        :return: Hamming distance.
        """
        assert (len(target) == len(hypothesis))
        return sum(h == t for h, t in zip(hypothesis, target)) / len(target)


class SequenceNormalizedEditDistanceMargin(NormalizedEditDistanceMargin):
    def score(self, target: List[str], hypothesis: List[str]) -> float:
        """
        `NormalizedEditDistanceMargin` averaged over single prediction words.
        :param target: Target.
        :param hypothesis: Hypothesis.
        :return: Hamming distance.
        """
        assert (len(target) == len(hypothesis))
        out = []
        for h, t in zip(hypothesis, target):
            if t == h:
                out.append(self.margin)
            else:
                out.append(- self.margin * editdistance.eval(t, h) / max(len(t), len(h)))
        return float(np.mean(out))


class Featurizer(abc.ABC):

    @abc.abstractmethod
    def _featurize(self, predictions: SequenceSample) -> List[np.ndarray]:
        pass

    def _batch_featurize(self, input):

        X_: List[np.ndarray] = []
        Y_: List[np.ndarray] = []
        Truth_: List[np.ndarray] = []

        start, number_inputs, predictions_per_input, score_function, sequences = input
        for sn, sequence_sample in enumerate(sequences, start=start):
            target = sequence_sample.reference

            tmp_xs: List[np.ndarray] = self._featurize(sequence_sample)
            tmp_ys: List[float] = []
            tmp_truth: List[float] = []

            for h, hypothesis in enumerate(sequence_sample.paths):
                score = score_function.score(hypothesis, target)
                tmp_ys.append(score)
                tmp_truth.append(float(np.mean([h == t for h, t in zip(hypothesis, target)])))

            this_predictions_per_input = len(tmp_xs)

            if this_predictions_per_input != predictions_per_input:
                print(f'Input {sequence_sample.sent_num} has an unexpected number of predictions: '
                      f'{this_predictions_per_input} vs {predictions_per_input}')
                diff = predictions_per_input - this_predictions_per_input
                if diff > 0:
                    count = 0
                    while count < diff:
                        tmp_xs.append(tmp_xs[count])
                        tmp_ys.append(tmp_ys[count])
                        tmp_truth.append(tmp_truth[count])
                        count += 1
                else:
                    tmp_xs = tmp_xs[:predictions_per_input]
                    tmp_ys = tmp_ys[:predictions_per_input]
                    tmp_truth = tmp_truth[:predictions_per_input]

            X_.append(np.array(tmp_xs))
            Y_.append(np.array(tmp_ys))
            Truth_.append(np.array(tmp_truth))

            print(f'\tFeaturized {sn}/{number_inputs} prediction...')

        return X_, Y_, Truth_

    def featurize(self, sequence_collection: SequenceCollection, score_function: ScoreFunction,
                  num_threads: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        X: List[np.ndarray] = []
        Y: List[np.ndarray] = []
        Truth: List[np.ndarray] = []

        number_inputs, predictions_per_input = sequence_collection.shape()

        if num_threads > 1:
            print(f'Featuring with {num_threads} threads...')
            # chunk the data into the number of threads
            step_size = math.ceil(number_inputs / num_threads)
            grouped_sequences = [(i, number_inputs, predictions_per_input, score_function,
                                  sequence_collection[i:i + step_size]) for i in range(0, number_inputs, step_size)]
            pool = multiprocessing.Pool(num_threads)
            results = pool.map(self._batch_featurize, grouped_sequences)
            pool.terminate()
            pool.join()

            for Xs, Ys, Truths in results:
                X.extend(Xs)
                Y.extend(Ys)
                Truth.extend(Truths)

        else:
            print(f'Featuring all data sequentially...')
            for sn, sequence_sample in enumerate(sequence_collection):
                target = sequence_sample.reference

                tmp_xs: List[np.ndarray] = self._featurize(sequence_sample)
                tmp_ys: List[float] = []
                tmp_truth: List[float] = []

                for h, hypothesis in enumerate(sequence_sample.paths):
                    score = score_function.score(hypothesis, target)
                    tmp_ys.append(score)
                    tmp_truth.append(float(np.mean([h == t for h, t in zip(hypothesis, target)])))

                this_predictions_per_input = len(tmp_xs)

                if this_predictions_per_input != predictions_per_input:
                    print(f'Input {sequence_sample.sent_num} has an unexpected number of predictions: '
                          f'{this_predictions_per_input} vs {predictions_per_input}')
                    diff = predictions_per_input - this_predictions_per_input
                    if diff > 0:
                        count = 0
                        while count < diff:
                            tmp_xs.append(tmp_xs[count])
                            tmp_ys.append(tmp_ys[count])
                            tmp_truth.append(tmp_truth[count])
                            count += 1
                    else:
                        tmp_xs = tmp_xs[:predictions_per_input]
                        tmp_ys = tmp_ys[:predictions_per_input]
                        tmp_truth = tmp_truth[:predictions_per_input]

                X.append(np.array(tmp_xs))
                Y.append(np.array(tmp_ys))
                Truth.append(np.array(tmp_truth))

                #if sn > sn % 10 == 0:
                print(f'\tFeaturized {sn}/{len(sequence_collection)} predictions...')

        X_ = np.array(X)
        Y_ = np.array(Y)
        Truth_ = np.array(Truth)

        # 1st dim: Inputs, 2nd dim: predictions, 3rd dim: features / score
        assert X_.ndim == 3 and Y_.ndim == 2, (X_.shape, Y_.shape)
        assert X_.shape[0] == Y_.shape[0], (X_.shape, Y_.shape)
        assert X_.shape[1] == Y_.shape[1] == predictions_per_input, (X_.shape, Y_.shape)
        assert Y_.shape == Truth_.shape, (Y_.shape, Truth_.shape)

        return X_, Y_, Truth_


class SequenceFeaturizer(Featurizer):

    def __init__(self, language_model: SimpleLanguageModel, train_data: List[Sample],
                 char_nn_language_model: Optional[CharLanguageModel] = None,
                 word_nn_language_model: Optional[WordLanguageModel] = None,
                 pseudocount: int = 1., prefix_size: int = 2, suffix_size: int = 2,
                 word_unk_score: Optional[float] = None):
        """
        Tuns a prediction into a feature vector using the following features:
          * language model score (hypothesis)
          * noisy channel score (hypothesis | source)
          * average levenshtein distance (hypothesis, source)
          * (source, hypothesis) seen in training?
          * relative frequency of (source, hypothesis) in training

        :param language_model: Language model.
        :param train_data: Training data to derive some statistics from (e.g. source, hypothesis relative frequencies).
        :param char_nn_language_model:
        :param word_nn_language_model:
        :param pseudocount: Pseudocount for computing UNK probability for decoding.
        :param word_unk_score: If not None, will use this ln score for <unk>. This could be beneficial as, if <unk> had
            a high relative frequency in the training data, it will get unreasonably high scores by the model.
        """
        self.language_model = language_model
        self.char_nn_language_model = char_nn_language_model
        self.word_nn_language_model = word_nn_language_model

        self.train_data = [(s.source, s.target) for s in train_data]

        self.train_set_uniq = set(self.train_data)
        if self.train_data:
            num_types = len(self.train_set_uniq)
            len_train_data = len(self.train_data)
            self.discount = pseudocount / len_train_data
            self.train_counter = {sample: (count + pseudocount) / (len_train_data + pseudocount * num_types)
                                  for sample, count in Counter(self.train_data).items()}
        else:
            self.discount = 0.
            self.train_counter = dict()
        self.prefix_size = prefix_size
        self.suffix_size = suffix_size
        self.word_unk_score = word_unk_score

    def char_nn_lm_score(self, x):
        # workaround for pickle for multiprocessing
        if self.char_nn_language_model:
            scores = self.char_nn_language_model.score_batch([' '.join(p) for p in x], length_normalized=True)
        else:
            scores = [0.]*len(x)
        return scores

    def word_nn_lm_score(self, x):
        # workaround for pickle for multiprocessing
        if self.word_nn_language_model:
            scores = self.word_nn_language_model.score_batch(x, length_normalized=True, unk_score=self.word_unk_score)
        else:
            scores = [0.]*len(x)
        return scores

    def _featurize(self, predictions: SequenceSample) -> List[np.ndarray]:
        """
        Turn k-best predictions into a list of feature vectors.
        :param predictions: k-best predictions.
        :return: Feature vectors.
        """
        feature_vectors: List[np.ndarray] = []
        source = predictions.origin_words

        char_nn_scores = self.char_nn_lm_score(predictions.paths)
        word_nn_scores = self.word_nn_lm_score(predictions.paths)

        for i, (score, hypothesis) in enumerate(zip(predictions.scores, predictions.paths)):
            obss = list(zip(hypothesis, source))
            length = len(source)
            feature_vector = np.array([
                1.,
                length,
                self.language_model.score(hypothesis) / length,
                char_nn_scores[i],
                word_nn_scores[i],
                score / length,
                sum(w in self.language_model for w in hypothesis) / length,
                sum(h[:self.prefix_size] == s[:self.prefix_size] for h, s in obss) / length,
                sum(h[-self.suffix_size:] == s[-self.prefix_size:] for h, s in obss) / length,
                self.language_model.score(hypothesis) * score / length,
                np.mean([editdistance.eval(h, s) for h, s in obss]),
                np.mean([float(obs in self.train_set_uniq) for obs in obss]),
                np.mean([self.train_counter.get(obs, self.discount) for obs in obss]),
            ])
            feature_vectors.append(feature_vector)
        return feature_vectors


def step0func(d):
    # for pickle for multiprocessing
    return stepfunc(d, step=0)


class SequenceReranker:

    def __init__(self, config):

        # paths
        model_dir = config.get('model_dir')
        reranker_dir = config.get('reranker_dir')
        dev_dec_json = config['dev_dec_json']
        test_dec_json = config.get('test_dec_json')
        test2_dec_json = config.get('test2_dec_json')
        train = config.get('train')
        dev = config.get('dev')
        test = config.get('test')
        test2 = config.get('test2')

        load_reranker_weights = config.get('load_reranker_weights')
        save_reranker_weights = config.get('save_reranker_weights')

        # features
        lm = config['lm']
        char_nn_lm = config.get('char_nn_lm')
        word_nn_lm = config.get('word_nn_lm')

        # reranker classifier hyperparams
        epochs = config.get('epochs', 5)
        tau = config.get('tau', 500)
        cutoff = config.get('cutoff', 100)
        eta = config.get('eta', 0.05)
        restarts = config.get('restarts', 1)
        compute_exact = config.get('compute_exact')
        sklearn = config.get('sklearn')
        random_seed = config.get('random_seed')
        word_lm_unk_score = config.get('word_lm_unk_score')

        self.num_threads = config.get('num_threads', 1)

        def path(path: str, project_path: Path) -> Path:
            path = Path(path)
            if not path.exists():
                print(f'File {path} not found. Looking into project path now...')
                path = project_path / path
                assert path.exists(), ('Path doesn\'t exist: ', path)
                print(f'... file {path} found. OK!')
            return path

        if reranker_dir is not None:
            self.project_path = Path(reranker_dir)
        elif model_dir is not None:
            self.project_path = Path(model_dir)
        else:
            self.project_path = Path('/')

        self.path2dev_json = path(dev_dec_json, self.project_path)
        if test_dec_json:
            self.path2test_json = path(test_dec_json, self.project_path)
        else:
            self.path2test_json = None
        if test2_dec_json:
            self.path2test2_json = path(test2_dec_json, self.project_path)
        else:
            self.path2test2_json = None
        if train is not None:
            path2train_tsv = path(train, self.project_path)
        else:
            path2train_tsv = None
#            path2train_tsv = path('train.tsv', self.project_path)
        if dev:
            self.path2dev_tsv = path(dev, self.project_path)
        else:
            self.path2dev_tsv = path('dev.tsv', self.project_path)
        if test:
            self.path2test_tsv = path(test, self.project_path)
        elif self.path2test_json:
            self.path2test_tsv = path('test.tsv', self.project_path)
        else:
            self.path2test_tsv = None
        if test2:
            self.path2test2_tsv = path(test2, self.project_path)
        elif self.path2test2_json:
            self.path2test2_tsv = path('test2.tsv', self.project_path)
        else:
            self.path2test2_tsv = None

        print('Paths:')
        for name, path_obj in (
                ('project_path', self.project_path), ('path2dev_json', self.path2dev_json),
                ('path2test_json', self.path2test_json), ('path2test2_json', self.path2test2_json),
                ('path2train_tsv', path2train_tsv), ('path2dev_tsv', self.path2dev_tsv),
                ('path2test_tsv', self.path2test_tsv), ('path2test2_tsv', self.path2test2_tsv),
                ('lm', lm), ('char_nn_lm', char_nn_lm), ('word_nn_lm', word_nn_lm)):
            print(f'{name: <20} : {path_obj}')
        print()

        if load_reranker_weights:
            path2weights = Path(load_reranker_weights)
            if path2weights.suffix != '.npy':  # fix possibly empty extension
                path2weights = path2weights.parent / (path2weights.stem + '.npy')
            self.path2weights = path(path2weights, self.project_path)
        else:
            self.path2weights = None
        if save_reranker_weights:
            path2save_weights = self.project_path / save_reranker_weights
            if path2save_weights.suffix != '.npy':  # fix possibly empty extension
                self.path2save_weights = path2save_weights.parent / (path2save_weights.stem + '.npy')
            else:
                self.path2save_weights = path2save_weights
            assert not self.path2save_weights.exists(), (
                'Path for saving reranker params exists!', self.path2save_weights)
        else:
            self.path2save_weights = None

        reranker_params = dict(epochs=epochs, tau=tau, cutoff=cutoff, eta=eta, restarts=restarts,
                               upper_bound_exact=800 if compute_exact else 0, alpha=step0func,
                               num_threads=self.num_threads, random_seed=random_seed)
        print('Reranker training params: ')
        for k, v in reranker_params.items():
            print(f'{k: <40} : {v}')
        if path2train_tsv is not None:
            train_data = list(direct_fn_reader(path2data=path2train_tsv))
        else:
            train_data = []

        language_model = SimpleLanguageModel(path2model=lm)

        if char_nn_lm:
            char_nn_language_model = CharLanguageModel.load_language_model(char_nn_lm)
        else:
            char_nn_language_model = None

        if word_nn_lm:
            word_nn_language_model = WordLanguageModel.load_language_model(word_nn_lm)
        else:
            word_nn_language_model = None

        self.score_function = SequenceNormalizedEditDistanceMargin()

        self.featurizer = SequenceFeaturizer(language_model, train_data, char_nn_language_model, word_nn_language_model,
                                             word_unk_score=word_lm_unk_score)

        if sklearn:
            self.reranker = PROClassifierReranker(SGDClassifier(fit_intercept=False, early_stopping=False,
                                                                max_iter=50, tol=1e-3, n_jobs=-1), **reranker_params)
            print(f'Using the following base classifier: {self.reranker.classifier}')
        else:
            self.reranker = PROReranker(**reranker_params)
            print(f'Using the Cassius Clay base classifier.')


    def rerank(self):

        if self.path2weights:
            self.reranker.load_from_file(self.path2weights)
            print(f'Loaded reranker parameters from {self.path2weights}...')
        else:
            print('** TRAINING **')
            then = time.time()
            print('Featurize data ...')
            predictions = SequenceCollection(list(data_reader(path2data=self.path2dev_json)))
            X, y, truth = self.featurizer.featurize(predictions, self.score_function, num_threads=self.num_threads)
            print(f'Finished featurizing in {(time.time() - then):.3f} sec.')

            then = time.time()
            print('Training ...')
            self.reranker.train(X, y, truth)
            print(f'Finished training in {(time.time() - then):.3f} sec.')

            then = time.time()
            print('Predicting ...')
            reranker_predictions, _ = self.reranker.predict(X, y, truth)
            print(f'Finished predicting in {(time.time() - then):.3f} sec.')

            if self.path2save_weights:
                self.reranker.save_to_file(self.path2save_weights)
                print(f'Saved reranker parameters to {self.path2save_weights}...')

        print('** PREDICTION **')

        for withheld_name, path2tsv, path2json in (('dev', self.path2dev_tsv, self.path2dev_json),
                                                   ('test', self.path2test_tsv, self.path2test_json),
                                                   ('test2', self.path2test2_tsv, self.path2test2_json)):

            print(withheld_name.upper(), '...\n')

            if path2tsv:

                if withheld_name == 'dev' and len(predictions) > 20:

                    print('...for speed, reusing featurized dev data.')

                else:
                    predictions = SequenceCollection(list(data_reader(path2data=path2json)))

                    then = time.time()
                    print('Featurize data ...')
                    X, y, truth = self.featurizer.featurize(predictions, self.score_function,
                                                            num_threads=self.num_threads)
                    print(f'Finished featurizing in {(time.time() - then):.3f} sec.')

                then = time.time()
                print('Predicting ...')
                reranker_predictions, _ = self.reranker.predict(X, y, truth)
                print(f'Finished predicting in {(time.time() - then):.3f} sec.')

                predictions_fn = self.project_path / f'f.rerank.{withheld_name}.predictions'
                eval_fn = self.project_path / f'f.rerank.{withheld_name}.eval'
                # @TODO for tag-stratified eval, launch evalm.py

                reranker_predict = list()
                correct = 0
                total = 0
                lev = []
                nlev = []
                for e, (p, r) in enumerate(zip(predictions.sequences, reranker_predictions)):
                    best_path = p.paths[r]
                    correct += sum(r == p for r, p in zip(p.reference, best_path))
                    total += len(p.reference)
                    lev.extend(editdistance.eval(r, p) for r, p in zip(p.reference, best_path))
                    nlev.extend(editdistance.eval(r, p) / len(r) for r, p in zip(p.reference, best_path))
                    reranker_predict.append(zip(p.origin_words, best_path))

                reranker_predict.append(iter([]))  # dummy object to terminate iteration
                reranker_predict = iter(reranker_predict)

                with predictions_fn.open(mode='w') as w:
                    newline = ''
                    with Path(path2tsv).open() as f:
                        ps = next(reranker_predict)
                        for l, line in enumerate(f):
                            line = line.rstrip()
                            if line:
                                source, target = line.rsplit('\t')
                                try:
                                    check_source, pred = next(ps)
                                except StopIteration as e:
                                    print(f'Featurized input exhausted too early probably because of '
                                          f'constraint on sentence length: line {l, line}')
                                    ps = next(reranker_predict)
                                    check_source, pred = next(ps)
                                assert source == check_source, (source, check_source, list(ps))
                                newline = check_source, pred
                            else:
                                try:
                                    next(ps)
                                except StopIteration as e:
                                    ps = next(reranker_predict)
                                    newline = ''
                            w.write('\t'.join(newline) + '\n')

                # SANITY CHECK
                with Path(path2tsv).open() as f:
                    trgts = [l.rstrip() for l in f if l.rstrip()]
                with predictions_fn.open() as f:
                    preds = [l.rstrip() for l in f if l.rstrip()]
                assert np.isclose(sum(t == p for t, p in zip(trgts, preds)) / len(trgts), correct / total)

                print(f'Accuracy: {correct * 100 / total:.1f}')
                with eval_fn.open(mode='w') as w:
                    w.write('MODEL\tTAG\tACCURACY\tLEVDIST\tNORMLEVDIST\n')
                    w.write(f'{predictions_fn}\t---\t{correct / total}\t{np.mean(lev)}\t{np.mean(nlev)}\n')

            else:
                print(f'... skipping as {withheld_name} data tsv / predictions json are available.')

        print('Done.')


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description='Train a reranker for word sequences and decode with it.')
    parser.add_argument('--model-dir', help=('Path to the directory of the trained model. '
                                             'Necessary if the paths to train / dev / test '
                                             '/ decoding / ... files are not specified.'
                                             'Defaults to / if not specified.'),
                        type=str, default=None)
    parser.add_argument('--reranker-dir', help=('Path to the directory of the reranker output.'
                                                'Defaults to model-dir option if not specified.'),
                        type=str, default=None)
    parser.add_argument('--dev-dec-json', help=('Filename of in the `model_dir` or path to the json with hypotheses '
                                                'for the dev set.'), type=str, required=True)
    parser.add_argument('--test-dec-json', help=('Filename of in the `model_dir` or path to the json with hypotheses '
                                                 'for the test set.'), type=str, default=None)
    parser.add_argument('--test2-dec-json', help=('Filename of in the `model_dir` or path to the json with hypotheses '
                                                 'for the other test set.'), type=str, default=None)
    parser.add_argument('--load-reranker-weights', help='Path to reranker weight file.', type=str, default=None)
    parser.add_argument('--save-reranker-weights', help='Path to file where to save reranker weights.',
                        type=str, default=None)
    parser.add_argument('--lm', help='Path to the kenlm language model.', type=str, required=True)
    parser.add_argument('--char-nn-lm', help='Path to the character-level RNN language model.',
                        type=str, required=False)
    parser.add_argument('--word-nn-lm', help='Path to the word-level RNN language model.', type=str, required=False)
    parser.add_argument('--train', help='Path to the training data.', type=str, default=None)
    parser.add_argument('--dev', help='Path to the dev set data.', type=str, default=None)
    parser.add_argument('--test', help='Path to the test set data.', type=str, default=None)
    parser.add_argument('--test2', help='Path to the other test set data.', type=str, default=None)
    parser.add_argument('--epochs', help='Reranker training: Number of training epochs.', type=int, default=5)
    parser.add_argument('--tau', help='Reranker training: Number of samples drawn per input.', type=int, default=500)
    parser.add_argument('--cutoff', help='Reranker training: Number of samples per input to retain.',
                        type=int, default=100)
    parser.add_argument('--eta', help='Reranker training: Perceptron learning rate.', type=float, default=0.05)
    parser.add_argument('--restarts', help='Reranker training: Number of restarts.', type=int, default=1)
    parser.add_argument('--compute-exact', help='Reranker training: Do not sample, compute all 2-combinations.',
                        action='store_true')
    parser.add_argument('--sklearn', help='Reranker training: Use sklearn\'s SVM as base classifier.',
                        action='store_true')
    parser.add_argument('--num-threads', help='Number of threads to use for featurization and training.', type=int,
                        default=1)
    parser.add_argument('--random-seed', help='Random seed.', type=int, default=None)
    parser.add_argument('--word-lm-unk-score', help='UNK ln score for word-level RNN language model.', type=float,
                        default=None)

    args = parser.parse_args()

    sequence_reranker = SequenceReranker(vars(args))
    sequence_reranker.rerank()
