import csv
import os
import time
import uuid
import math
import shutil
import json
import operator
import multiprocessing
from collections import Counter
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Union

import numpy as np

from contextualized_transduction.candidate_generator import CandidateGeneratorWriter, CrudeCharNGramGenerator, \
    CrudeMEDSGenerator
from contextualized_transduction.lm import KenLM
from contextualized_transduction.bigram_kenlm import BigramKenLM
from contextualized_transduction.trigram_kenlm import TrigramKenLM
from contextualized_transduction.neural_channel import OffLine_NT_Channel
from contextualized_transduction.sed_channel import StochasticEditDistance_Channel, DummyChannel
from contextualized_transduction.exact_model import ExactModel
from contextualized_transduction.inexact_model import ViterbiEStepModel, BeamEStepModel

from contextualized_transduction.utils import ENCODING, NT_DEFAULT_FEATURE, LARGE_POS_CONST, UNK, read_dataset, \
    ComponentConfig, scheduler_dequeue, scheduler_queue
from contextualized_transduction.vocabulary import Vocab, VocabSet, Dataset, SentSample, write_vocabsets

from nn_lm.custom_lm import CharLanguageModel, WordLanguageModel

Batch = List[SentSample]


class BatchDecoderResult:

    def __init__(self, correct: int, total: int, times: List[float], batch_result_fn: Union[Path, str],
                 kbest_result_fn: Union[Path, str, None] = None, upper_correct: Optional[int] = None,
                 weights: Optional[np.ndarray] = None):
        """
        Container for decoding a batch of sentences.
        :param correct: The number of correctly predicted words in the batch.
        :param total: The total number of words in the batch.
        :param times: The list of decoding times---one measurement per sentence.
        :param batch_result_fn: The name of the file that holds the MAP predictions for the batch's sentences.
        :param kbest_result_fn: The name of the json file that holds the k-best predictions for the batch's sentences.
        :param upper_correct: The upper bound on the number of correctly predicted words in the batch, computed from
            the most accurate k-best prediction for each sentence.
        :param weights: Optionally---for training---the posterior weights aggrerated for each candidate and each word
            of every sentence in the batch.
        """
        self.correct = correct
        self.total = total
        self.times = times
        self.batch_result_fn = batch_result_fn
        self.kbest_result_fn = kbest_result_fn
        self.upper_correct = upper_correct
        self.weights = weights


def print_report(words: List[str], prediction: List[str], ref: List[str], sent_num: int, len_data: int) -> None:
    """
    Print source word sequence, prediction sequence, and reference sequence together with any true, missed, and
    over-corrections.
    :param words: A sequence of source (corrupted) words.
    :param prediction: A sequence of predicted correct words.
    :param ref: The reference.
    :param sent_num: The ordinal number of this sequence of words.
    :param len_data: The total number of word sequences in the dataset.
    """
    check = 'V'
    delta_input = []
    delta_target = []
    for z, (ww, cw, tw) in enumerate(zip(words, prediction, ref)):
        if cw == tw:
            if ww != cw:
                # This a good correction of input!
                delta_input.append((z, ww, cw))
        else:
            check = 'X'
            # This is incorrect. Is this due to correction (+)? Else =.
            delta_target.append((z, cw, tw, '+' if ww != cw else '='))
    print('=\t' if words == ref else '!\t', check,
          '\t{0}/{1}\t'.format(sent_num, len_data),
          'words, corrected, target:\n', ' '.join(words), '\n',
          ' '.join(prediction), '\n', ' '.join(ref), '\n', '*' * 80)
    if delta_target or delta_input:
        print('Delta from target: ', delta_target)
        print('Delta from input: ', delta_input)


class Trainer:

    def __init__(self, config: Dict):

        self.config = dict(config)

        # * GIVEN RESULTS PATH, DEFINE ALL OTHER PATHS
        self._define_paths()

        # * GENERAL TRAIN SETTINGS
        train = self.config['train']

        self.write_hypotheses_to_file = train.get('write_hypotheses_to_file', True)
        self.verbose = train.get('verbose', True)

        self.num_candidates = train['num_candidates']
        self.add_word = train.get('add_word', False)
        self.kbest = train.get('kbest', 50)
        self.final_kbest = train.get('final_kbest', 500)

        self.lowercase = train.get('lowercase', True)
        self.epochs = train.get('epochs', 35)
        self.num_threads = train.get('train_threads', 1)
        self.alpha_unsupervised = train.get('alpha_supervised', 0.8)
        self.tol = train.get('posterior_tol', 0.001)
        self.dummy_channel_init = train.get('dummy_channel_init', False)

        self.inexact_estep_kbest = train.get('inexact_estep_kbest', None)

        # * READ DATA
        self._read_data()

        # * DEFINE MODEL
        components = self.config['components']
        candidate_generator_class = eval(components.get('candidate_generator')) or CrudeCharNGramGenerator
        channel_class = eval(components.get('channel_model')) or OffLine_NT_Channel
        language_model_class = eval(components.get('language_model')) or BigramKenLM

        candidate_generator_params = self.config['candidates']
        candidate_generator_params['candidates_alphabet'] = self.target_charset.set
        candidate_generator_params['indomain_candidates'] = self.indomain_candidates
        candidate_generator_params['lowercase'] = candidate_generator_params.get('lowercase', self.lowercase)
        assert self.lowercase == candidate_generator_params['lowercase'], candidate_generator_params['lowercase']
        candidate_generator_params['verbose'] = candidate_generator_params.get('verbose', False)

        # * CANDIDATES
        candidate_generator_config = ComponentConfig(candidate_generator_class, candidate_generator_params)

        # * CHANNEL
        channel_params = self.config['channel']
        channel_params['generated_candidates_fn'] = self.generated_candidates_fn
        # NB! The channel's source and target are direct model's target and source, hence reversed below:
        channel_params['source_alphabet'] = self.target_charset.set
        channel_params['target_alphabet'] = self.source_charset.set
        channel_params['backoff_alphabet'] = self.mixed_charset.set
        channel_params['em_data_dir'] = self.result_path

        path2neural_code = channel_params.get('path2neural_code')
        if path2neural_code:
            assert os.path.exists(path2neural_code)
            os.environ["PATH2NEURAL_CODE"] = path2neural_code

        # these next three paths are important for supervised pretraining of the channel model
        self.path2candidate_scores = channel_params.get('path2candidate_scores', None)
        self.path2param_dicts = channel_params.get('param_dicts', None)
        self.channel_reload_path = channel_params.get('reload_path', None)

        channel_config = ComponentConfig(channel_class, channel_params)

        self.channel_update_params = self.config['channel_update_params']
        self.channel_pretrain_params = self.config.get('channel_pretrain_params', self.channel_update_params)

        self.channel_update_params['dev_path'] = self.channel_update_params.get('dev_path', self.path2channel_dev)
        self.channel_pretrain_params['dev_path'] = self.channel_pretrain_params.get('dev_path', self.path2channel_dev)

        # * LANGUAGE MODEL
        language_model_params = self.config['language_model']
        assert language_model_params['lowercase'] == self.lowercase, language_model_params['lowercase']

        if issubclass(language_model_class, KenLM):
            if self.num_threads > 1:
                try:
                    import pickle
                    with open('/tmp/lm.pickle', 'wb') as pf:
                        pickle.dump(language_model_class(**language_model_params), pf, pickle.HIGHEST_PROTOCOL)
                except Exception as e:
                    print('Cannot pickle the language model as a result, multiprocessing cannot be used. Probably '
                          'wrong kenlm (it needs to be installed from github commit).')
                    raise e
        else:
            assert self.inexact_estep_kbest, 'Exact training can only be performed with n-order LM.'

        language_model_config = ComponentConfig(language_model_class, language_model_params)

        # * IDENTIFYING CORRECT E-STEP MODEL
        if self.inexact_estep_kbest is None:
            # exact E-step with n-gram LM
            self.EStepModel = ExactModel
            print('Computing E-step exactly...')
        else:
            if issubclass(language_model_class, KenLM):
                # inexact E-step with n-gram LM
                self.EStepModel = ViterbiEStepModel
            else:
                # inexact E-step with RNN LM
                self.EStepModel = BeamEStepModel
            print(f'Computing inexact E-step using {self.EStepModel} model '
                  f'from k={self.inexact_estep_kbest} predictions...')

        # * PUTTING ALL TOGETHER
        self.model = self.EStepModel(candidate_generator_config=candidate_generator_config,
                                     language_model_config=language_model_config,
                                     channel_config=channel_config, num_candidates=self.num_candidates,
                                     kbest=self.inexact_estep_kbest)

        self.candidate_generator = self.model.candidate_generator
        self.language_model = self.model.language_model
        self.channel = self.model.channel

        self.writer_cand_generator = CandidateGeneratorWriter(generated_cadidates_fn=self.generated_candidates_fn,
                                                              candidate_generator=self.candidate_generator)

        # * HOW TO INITIALIZE CHANNEL ON FIRST ITERATION IF NO SUPERVISED TRAINING AVAILABLE
        if self.dummy_channel_init or not (self.sup_direct_dataset or self.path2candidate_scores or
                                           self.path2param_dicts or self.channel_reload_path):
            self.dummy_model = self.EStepModel(candidate_generator_config=candidate_generator_config,
                                               language_model_config=language_model_config,
                                               channel_config=ComponentConfig(DummyChannel, dict()),
                                               num_candidates=self.num_candidates,
                                               kbest=self.inexact_estep_kbest)
        else:
            self.dummy_channel_init = False
            print('dummy channel initialization is set to False.')
            self.dummy_model = None

        # * REPORT ON ALL TRAINING SETTINGS
        self._print_report()

        # * DEFINE DATA STRUCTURES FOR STORING WEIGHTS, ENCODE SOURCE WORDS AND CANDIDATES
        self._integerize()

    def _define_paths(self):

        paths = self.config['paths']

        self.result_path = Path(paths['result_path'])
        self.result_path.mkdir(parents=True, exist_ok=False)
        self.generated_candidates_fn = paths.get('generated_candidates_fn') or self.result_path / 'candidates.tsv'
        self.all_hypotheses_fn_format = os.path.join(self.result_path, 'channel_e{}.tsv')
        self.path2vocabs = self.result_path / 'vocab.json'
        # best path hypothesis per sentence
        self.train_results_fn_format = os.path.join(self.result_path, 'train_corrections_e{}.tsv')
        self.decoding_results_fn_format = os.path.join(self.result_path, '{}_corrections_e{}.tsv')
        self.kbest_decoding_results_fn_format: str = os.path.join(self.result_path, '{}_kbest_e{}.json')
        self.report_fn = self.result_path / 'corrections.results'
        self.final_params_fn = self.result_path / 'best_model.path'
        self.best_param_path: Union[Path, str, None] = None

    def _read_data(self):

        paths2data = self.config['data']
        self.dataset_name = paths2data.get('name')
        self.path2unsupervised_data = Path(paths2data['unsupervised'])
        self.path2supervised_dev = Path(paths2data['supervised_dev'])
        self.data_cutoff = paths2data.get('unsupervised_data_cutoff', LARGE_POS_CONST)
        self.max_seg_len = paths2data.get('max_seg_len', LARGE_POS_CONST)
        self.path2supervised_train = paths2data.get('supervised_train')
        self.path2test = paths2data.get('test')
        self.path2another_test = paths2data.get('another_test')
        self.path2indomain_candidates = paths2data.get('indomain_candidates_fn')

        # Whereas the source character set can be taken from the unsupervised and supervised data,
        # the target character set should ideally be learned on an external dataset.
        self.path2target_charset = paths2data.get('target_charset')
        if self.path2target_charset is not None:
            # assume an array of characters
            with Path(self.path2target_charset).open(encoding=ENCODING) as f:
                self.target_charset: VocabSet = VocabSet.from_json(f)
            print(f'Loaded target charset from "{self.target_charset}".')
        else:
            self.target_charset: Optional[VocabSet] = None

        # unlabeled training set
        self.dataset, self.source_charset, target_charset, train_copy_baseline = \
            read_dataset(self.path2unsupervised_data, dataset_name="train", dir_="",
                         data_cutoff=self.data_cutoff, max_seg_len=self.max_seg_len)
        print(f'Loaded unsupervised dataset from "{self.path2unsupervised_data}".')
        if self.target_charset is None:
            self.target_charset = target_charset

        self.len_data = len(self.dataset)
        print(f'The number of sentences in the data: {self.len_data}.')

        # labeled training set
        self.sup_direct_dataset: List[Tuple[str, str]] = []  # historical => modern
        if self.path2supervised_train is not None:
            print('Semi-supervised training...')
            # assume direct order (source => target)
            self.trainset, sup_source_charset, sup_target_charset, _ = \
                read_dataset(self.path2supervised_train, dataset_name="sup_train",
                             dir_="", max_seg_len=self.max_seg_len)
            self.source_charset.update(sup_source_charset)
            self.target_charset.update(sup_target_charset)

            for _, words, ref in self.trainset:
                # direct order
                self.sup_direct_dataset.extend(zip(words, ref))

            print(f'Loaded supervised dataset from "{self.path2supervised_train}". '
                  f'The number of supervised samples: {len(self.sup_direct_dataset)}.')
            print('Examples (CHECK DIRECT order !!!): ', self.sup_direct_dataset[:20])
            print(f'The number of sentences in the supervised train set: {len(self.trainset)}.')
        else:
            print('Unsupervised training...')
            self.trainset = []

        # add UNK to charsets
        assert UNK not in self.source_charset
        assert UNK not in self.target_charset
        self.source_charset.update(UNK)
        self.target_charset.update(UNK)

        self.mixed_charset = VocabSet()
        self.mixed_charset.update(self.source_charset)
        self.mixed_charset.update(self.target_charset)

        # labeled dev set
        self.devset, *_, dev_copy_baseline = read_dataset(self.path2supervised_dev, dir_="",
                                                          dataset_name="dev", max_seg_len=self.max_seg_len)
        print(f'Loaded labeled dev set from "{self.path2supervised_dev}".')
        print(f'The number of sentences in the dev set: {len(self.devset)}.')
        # path to dev set used for model selection of e.g. neural channel model
        self.path2channel_dev = self.result_path / \
                                (self.path2supervised_dev.stem + '.channel' + self.path2supervised_dev.suffix)
        with self.path2channel_dev.open(mode='w', encoding=ENCODING) as w:
            for _, words, ref in self.devset:
                for word, ref_ in zip(words, ref):
                    # channel order:
                    w.write('\t'.join((ref_, word, NT_DEFAULT_FEATURE)) + '\n')
                # w.write('\n')  => \n breaks transducer reader
        print(f'Wrote dev set in the channel order to "{self.path2channel_dev}".')

        # labeled test set
        if self.path2test is not None:
            print('Will score on test set after training...')
            self.testset, *_, test_copy_baseline = read_dataset(self.path2test, dir_="",
                                                                dataset_name="test", max_seg_len=self.max_seg_len)
            print(f'Loaded test set from "{self.path2test}".')
            print(f'The number of sentences in the test set: {len(self.testset)}.')
        else:
            print('No test set.')
            self.testset = None

        # another labeled test set
        if self.path2another_test is not None:
            print('Will score on test set 2 after training...')
            self.testset2, *_, test2_copy_baseline = read_dataset(self.path2another_test, dir_="",
                                                                  dataset_name="test2",
                                                                  max_seg_len=self.max_seg_len)
            print(f'Loaded test set 2 from "{self.path2another_test}".')
            print(f'The number of sentences in test set 2: {len(self.testset2)}.')
        else:
            print('No test set 2.')
            self.testset2 = None

        # candidates for candidate generation (used in upper-bound tests)
        self.indomain_candidates: List[Tuple[str, str]] = list(self.sup_direct_dataset)
        if self.path2indomain_candidates is not None:

            ind_cand_set, ind_source_charset, ind_target_charset, _ = \
                read_dataset(self.path2indomain_candidates, dataset_name="indomain_cands", dir_="")
            self.source_charset.update(ind_source_charset)
            self.target_charset.update(ind_target_charset)

            for _, words, ref in ind_cand_set:
                # direct order
                self.indomain_candidates.extend(zip(words, ref))

            print(f'Loaded in-domain candidates from "{self.path2indomain_candidates}". '
                  f'Together with labeled samples, {len(self.indomain_candidates)} have targets.')
            print('Examples (CHECK source => target order !!!): ', self.indomain_candidates[-20:])

        # write copy baseline accuracy to file
        with self.report_fn.open(mode='a') as a:
            # @TODO add all settings
            if dev_copy_baseline:
                report = (f'Copy dev accuracy:\t{dev_copy_baseline:.3f}.\t\t\t'
                          f'Copy train accuracy:\t{train_copy_baseline:.3f}.\n')
            else:
                report = f'Copy train accuracy:\t{train_copy_baseline:.3f}\n'
            a.write(report)

    def _print_report(self):

        print('Using the following configuration for training: ')
        for k, v in self.__dict__.items():
            if isinstance(v, (str, int, float, bool, Path)):
                print(f'{k: <40} : {v}')

        print('\nCharacter sets:')
        for k, v in ('Source charset', self.source_charset), ('Target charset', self.target_charset):
            print(f'{k: <40} : {v}')

        print('\nModel:')
        for k, cl, v in (
                ('Candidate generator', self.model.candidate_generator_class, self.model.candidate_generator_params),
                ('Channel model', self.model.channel_class, self.model.channel_params),
                ('Language model', self.model.language_model_class, self.model.language_model_params)):
            print(f'{k: <40} : {cl}')
            for p, z in v.items():
                if isinstance(z, (str, int, float, bool, Path)):
                    print(f'{p: <40} : {z}')
            print()

    def _integerize(self):

        print('Integerizing the data...')
        start_prep = time.time()

        # COMPUTE SOURCE WORD TYPES ; COMPUTE ALL CANDIDATE TYPES
        self.WORD_VOCAB = Vocab()
        self.CAND_VOCABS: List[Vocab] = []

        # handle UNK: words without candidates (i.e. returning [UNK] as candidates) get replaced with UNK.
        _word_unk_index = self.WORD_VOCAB[UNK]
        cand_vocab = Vocab()
        _cand_unk_index = cand_vocab[UNK]
        self.CAND_VOCABS.append(cand_vocab)

        # encode supervised train set and unsupervised train set candidates
        wordset = VocabSet()
        wordset.add(UNK)
        for dataset in (self.dataset, self.trainset):
            for m, (_, words, _) in enumerate(dataset):
                for h, word in enumerate(words):
                    if word not in self.WORD_VOCAB:
                        candidates = list(self.writer_cand_generator.generate(
                            word, self.num_candidates, add_word=self.add_word))
                        if candidates == [UNK]:
                            print(f'** Warning: "{word}" generated no candidates! UNK-ing it.')
                            dataset[m].reset_word(h, UNK)  # replacement in the dataset with UNK
                            continue
                        else:
                            wordset.add(word)
                        _word_index = self.WORD_VOCAB[word]
                        cand_vocab = Vocab()
                        for candidate in candidates:
                            # 1) add candidates to a vocab associated with their source word
                            # 2) `WRITER_CAND_GENERATOR` writes candidates to file
                            _cand_index = cand_vocab[candidate]
                        self.CAND_VOCABS.append(cand_vocab)

        # COMPUTE DIMENSIONS FOR WEIGHT MATRIX
        self.WEIGHTS_DIMENSIONS = self.num_candidates + int(self.add_word), len(self.WORD_VOCAB)
        print('Size of the source vocabulary appearing in training: ', len(self.WORD_VOCAB))

        print(f'Actual average number of candidates per word: {np.mean([len(v) for v in self.CAND_VOCABS]):.1f}')

        # write vocabularies to a file: CHANNEL ORDER!
        write_vocabsets(self.path2vocabs, act=self.mixed_charset, feat=VocabSet(Counter(NT_DEFAULT_FEATURE)),
                        word=wordset)

        # encode dev and test set candidates
        seen = set(self.WORD_VOCAB.idx2word)
        for dataset in (self.devset, self.testset, self.testset2):
            if dataset:
                for ds, (_, words, ref) in enumerate(dataset):
                    for h, word in enumerate(words):
                        if word not in seen:
                            candidates = list(self.writer_cand_generator.generate(
                                word, self.num_candidates, add_word=self.add_word))
                            if candidates == [UNK]:
                                print(f'** Warning: "{word}" generated no candidates! UNK-ing it.')
                                dataset[ds].reset_word(h, UNK)  # replacement in the dataset with UNK
                            else:
                                seen.add(word)

        print(f'Finished preprocessing in {time.time() - start_prep:.3f} sec.')

    def _pretrain(self) -> None:
        # if pretrained model exists (as parameters or pre-computed scores), don't pretrain
        if isinstance(self.channel, OffLine_NT_Channel) and (self.path2candidate_scores or self.channel_reload_path):
            # assert not self.channel_update_params.get('reload'), 'No reload path given!'
            return
        elif not isinstance(self.channel, StochasticEditDistance_Channel) and self.path2param_dicts:
            return

        if self.sup_direct_dataset:
            print('Pretraining the channel...')
            start_pretrain = time.time()
            all_words, all_hypotheses = zip(*self.sup_direct_dataset)
            self.channel.update_model(sources=all_hypotheses, targets=all_words, weights=None,
                                      epoch_number=-1, **self.channel_pretrain_params)
            print('Finished pretraining in %.3f sec.' % (time.time() - start_pretrain))
        else:
            print('Creating placeholder channel score with dummy channel...')
            if isinstance(self.channel, OffLine_NT_Channel):
                dummy_pretrain_dir = self.result_path / self.channel.channel_data_template.format(-1)
                dummy_pretrain_dir.mkdir()
                with self.generated_candidates_fn.open(encoding=ENCODING) as f:
                    dummy_dict = dict()
                    for cand, word, feat in csv.reader(f, delimiter='\t', lineterminator='\n',
                                                       quotechar='', quoting=csv.QUOTE_NONE):
                        if word not in dummy_dict:
                            dummy_dict[word] = dict(candidates=[], feats=[], log_prob=[])
                        dummy_dict[word]["candidates"].append(cand)
                        dummy_dict[word]["feats"].append(feat)
                        dummy_dict[word]["log_prob"].append(-LARGE_POS_CONST)

                dummy_best_param_path = dummy_pretrain_dir / "dev_channel.json"
                with dummy_best_param_path.open(mode="w", encoding=ENCODING) as w:
                    json.dump(dummy_dict, w, ensure_ascii=False, indent=4)
            else:
                raise NotImplementedError
            self.dummy_model.channel.reset_from_path(str(dummy_best_param_path))

    def _decode(self, dataset: Dataset, epoch: Union[int, str], kbest: bool = False) -> Tuple[float, float]:
        """
        Decode a dev or test dataset.
        :param dataset: Dataset to decode.
        :param epoch: The epoch number.
        :param kbest: Additionally, generate k-best predictions.
        :return: Per-word accuracy over all sentences in the set as well as the upper bound on this accuracy from k-best
            decoding, if available.
        """
        dataset_name = dataset.dataset_name

        print(f'Decoding with {dataset_name} dataset...')

        correct: int = 0
        upper_correct: Optional[int] = 0 if kbest else None
        total: int = 0
        times: List[float] = []
        batch_result_fns: List[Union[Path, str]] = []
        kbest_result_fns: List[Union[Path, str]] = []

        dec_result_path = Path(self.decoding_results_fn_format.format(dataset_name, epoch))
        kbest_result_path = Path(self.kbest_decoding_results_fn_format.format(dataset_name, epoch))

        len_data = len(dataset)

        start_time = time.time()

        if self.num_threads > 1:
            try:
                scheduler_queue(lockfile_name=os.environ.get("CNNC_MULTIPROCESSING_RUNNING",
                                                             "CNNC_MULTIPROCESSING_RUNNING"))

                print('Computing posterior in %d threads...' % self.num_threads)

                # chunk the data into the number of threads
                step_size = math.ceil(len_data / self.num_threads)
                grouped_segs = [(i, len_data, dataset[i:i + step_size]) for i in range(0, len_data, step_size)]
                # limit the number of processes to the requested number of threads
                # (not to https://docs.python.org/3.7/library/os.html#os.cpu_count)
                job_not_done = True
                cur_num_threads = self.num_threads
                while job_not_done:
                    try:
                        pool = multiprocessing.Pool(cur_num_threads)
                        results = pool.map(self._kbest_decode_no_weights if kbest else self._decode_no_weights, grouped_segs)
                        job_not_done = False
                        pool.terminate()
                        pool.join()
                    except (OSError, MemoryError) as e:
                        print(e)
                        old_num_threads = cur_num_threads
                        cur_num_threads = max(1, cur_num_threads - 1)
                        print(f"Attempt to reduce the number of parallel threads from {old_num_threads} to {cur_num_threads}")

            finally:
                scheduler_dequeue(lockfile_name=os.environ.get("CNNC_MULTIPROCESSING_RUNNING",
                                                               "CNNC_MULTIPROCESSING_RUNNING"))

            for batch_result in results:
                correct += batch_result.correct
                if kbest:
                    upper_correct += batch_result.upper_correct
                total += batch_result.total
                times.extend(batch_result.times)
                batch_result_fns.append(batch_result.batch_result_fn)
                if kbest:
                    kbest_result_fns.append(batch_result.kbest_result_fn)

            # MOVE BATCH RESULTS INTO ONE RESULTS FILE
            # @TODO The correct file name template
            with dec_result_path.open(mode='w', encoding=ENCODING) as results_w:
                for batch_result_fn in batch_result_fns:
                    print(f'Copying batch from {batch_result_fn} to {results_w.name}')
                    with Path(batch_result_fn).open(encoding=ENCODING) as batch_result_f:
                        shutil.copyfileobj(batch_result_f, results_w)
                    print(f'Cleanup {batch_result_fn}')
                    os.unlink(batch_result_fn)

            if kbest:
                # WRITE K-BEST RESULTS TO FILE
                kbest_all = []
                for kbest_result_fn in kbest_result_fns:
                    with Path(kbest_result_fn).open(encoding=ENCODING) as kbest_result_f:
                        print(f'Copying k-best batch from {kbest_result_fn} to {kbest_result_path.name}')
                        kbest_results = json.load(kbest_result_f)
                        kbest_all.extend(kbest_results)
                    print(f'Cleanup {kbest_result_fn}')
                    os.unlink(kbest_result_fn)

                kbest_all.sort(key=operator.itemgetter('sent_num'))

                with kbest_result_path.open(mode='w', encoding=ENCODING) as kbest_results_w:
                    json.dump(kbest_all, kbest_results_w, indent=4, ensure_ascii=False)
        else:
            # SINGLE THREAD
            decode_params = dict(kbest=self.kbest) if kbest else dict()
            kbest_results = [] if kbest else None

            with dec_result_path.open(mode='w', encoding=ENCODING) as results_w:

                for sent_num, (orig_words, words, ref) in enumerate(dataset):
                    if self.verbose:
                        print(sent_num, words, '...')
                    then = time.time()
                    # @TODO return weights in LOG
                    results = self.current_model.decode(source_words=words, origin_words=orig_words, map_predict=True,
                                                        **decode_params)
                    best_path = results.best_path
                    times.append(time.time() - then)

                    correct += sum(p.lower() == t.lower() if self.lowercase else p == t
                                   for p, t in zip(best_path, ref))
                    total += len(words)

                    # SOME REPORTING
                    if self.verbose:
                        print_report(words, best_path, ref, sent_num, len_data)

                    # WRITE RESULTS TO FILE
                    csv.writer(results_w, delimiter='\t').writerows(list(zip(orig_words, best_path)) + [('', '')])

                    if kbest_results is not None:
                        kbest_results.append(
                            dict(sent_num=sent_num, origin_words=orig_words, words=words, reference=ref,
                                 scores=results.k_best_path_scores, paths=results.k_best_paths))

                        this_upper_correct = 0
                        for path in results.k_best_paths:
                            path_correct = sum(p.lower() == t.lower() if self.lowercase else p == t
                                               for p, t in zip(path, ref))
                            if path_correct > this_upper_correct:
                                this_upper_correct = path_correct
                        upper_correct += this_upper_correct


            if kbest_results is not None:
                # WRITE K-BEST RESULTS TO FILE
                with kbest_result_path.open(mode='w', encoding=ENCODING) as kbest_results_w:
                    json.dump(kbest_results, kbest_results_w, indent=4, ensure_ascii=False)


        epoch_compute_time = time.time() - start_time
        epoch_acc = 100 * np.mean(correct / total) if total else 1.
        if upper_correct is not None:
            epoch_up_acc = 100 * np.mean(upper_correct / total) if total else 1.
        print('Epoch %s %s accuracy: %.3f.%s Compute time: %.3f sec.' % (
            epoch, dataset_name, epoch_acc,
            '' if upper_correct is None else ' Upper bound: %.3f.' % epoch_up_acc,
            epoch_compute_time))
        with self.report_fn.open('a') as a:
            a.write('Epoch %s %s accuracy:\t%.3f.%s Compute time:\t%.3f sec.\t' % (
                epoch, dataset_name, epoch_acc,
                '' if upper_correct is None else '\tUB: %.3f.\t' % epoch_up_acc,
                epoch_compute_time))
        return epoch_acc, epoch_up_acc

    def _decode_batch(self, sents: Batch, start_num: int, len_data: int, with_weights: bool = False,
                      kbest: bool = False) -> BatchDecoderResult:
        """
        Decode unlabeled sentences using viterbi decoding and optionally, collect weights.
        :param sents: A list of sentences.
        :param start_num: For reporting, the index of the first sentence of the batch in the original dataset.
        :param len_data: For reporting, the length of the dataset.
        :param kbest: Decoding with k-best viterbi decoding. Otherwise, only best viterbi path.
        :param with_weights: Collect weights.
        :return: The number of correctly predicted words, the total number of words, the times per sentence that it has
            taken to compute the weights, the name of the file with best-path predictions for the batch,
            optional name of the file with k-best path predictions, and optional weights.
        """
        correct: int = 0
        total: int = 0
        times: List[float] = []
        batch_result_fn: Path = Path('/tmp/') / uuid.uuid4().hex
        # the weights data structure is only used in training
        acc_weights: Optional[np.ndarray] = np.zeros(self.WEIGHTS_DIMENSIONS) if with_weights else None
        if kbest:
            predict_params = dict(kbest=self.kbest)
            kbest_results = []
            upper_correct: Optional[int] = 0
        else:
            predict_params = dict()
            kbest_results = None
            upper_correct = None

        with batch_result_fn.open(mode='w', encoding=ENCODING) as tmp_results_w:

            for sent_num, sent in enumerate(sents, start=start_num):

                orig_words = sent.original_words
                words = sent.words
                ref = sent.refs

                then = time.time()
                # @TODO return weights in LOG
                assert words, (sent_num, len_data, words, ref)
                assert ref, (sent_num, len_data, words, ref)
                results = self.current_model.decode(source_words=words, origin_words=orig_words,
                                                    map_predict=True, **predict_params)
                best_path = results.best_path

                times.append(time.time() - then)

                if with_weights:
                    # aggregate weights
                    for word, word_t_candidates in enumerate(results.candidates):
                        word_index = self.WORD_VOCAB[words[word]]
                        for k, candidate in enumerate(word_t_candidates):
                            candidate_index = self.CAND_VOCABS[word_index][candidate]
                            acc_weights[candidate_index, word_index] += results.weights[k, word]
                if self.verbose:
                    print('\t{0}/{1}\t'.format(sent_num, len_data),
                          'words, corrected, target:\n', ' '.join(words), '\n',
                          ' '.join(best_path), '\n', ' '.join(ref), '\n', '*' * 80)

                correct += sum(p.lower() == t.lower() if self.lowercase else p == t for p, t in zip(best_path, ref))
                total += len(words)

                csv.writer(tmp_results_w, delimiter='\t').writerows(list(zip(orig_words, best_path)) + [('', '')])

                if kbest_results is not None:
                    kbest_results.append(
                        dict(sent_num=sent_num, origin_words=orig_words, words=words, reference=ref,
                             scores=results.k_best_path_scores, paths=results.k_best_paths))

                    this_upper_correct = 0
                    for path in results.k_best_paths:
                        path_correct = sum(p.lower() == t.lower() if self.lowercase else p == t
                                           for p, t in zip(path, ref))
                        if path_correct > this_upper_correct:
                            this_upper_correct = path_correct
                    upper_correct += this_upper_correct


            if kbest_results is not None:

                kbest_result_fn: Optional[Path] = Path('/tmp/') / uuid.uuid4().hex

                with kbest_result_fn.open(mode='w', encoding=ENCODING) as kbest_results_w:
                    json.dump(kbest_results, kbest_results_w, indent=4, ensure_ascii=False)
            else:
                kbest_result_fn = None

        return BatchDecoderResult(correct=correct, total=total, times=times, batch_result_fn=batch_result_fn,
                                  weights=acc_weights, kbest_result_fn=kbest_result_fn, upper_correct=upper_correct)

    def _collect_weights(self, input_sents: Tuple[int, int, Batch]) -> BatchDecoderResult:
        """
        Collect weights from a subset of unlabeled training sentences.
        :param input_sents: The index of the first sentence in the dataset, length of the dataset,
            and a list of sentences.
        :return: Weights, the number of correctly predicted words, the total number of words, the times per sentence
            that it has taken to compute the weights, and the name of the file with best-path predictions for the batch.
        """
        start_num, len_data, sents = input_sents
        return self._decode_batch(sents=sents, start_num=start_num, len_data=len_data, with_weights=True)

    def _decode_no_weights(self, input_sents: Tuple[int, int, Batch]) -> BatchDecoderResult:
        """
        Collect MAP predictions for a subset of sentences.
        :param input_sents: The index of the first sentence in the dataset, length of the dataset,
            and a list of sentences.
        :return: The number of correctly predicted words, the total number of words, the times per sentence
            that it has taken to compute the weights, and the name of the file with best-path predictions for the batch.
        """
        start_num, len_data, sents = input_sents
        return self._decode_batch(sents=sents, start_num=start_num, len_data=len_data, with_weights=False)

    def _kbest_decode_no_weights(self, input_sents: Tuple[int, int, Batch]) -> BatchDecoderResult:
        """
        Collect k-best predictions for a subset of sentences.
        :param input_sents: The index of the first sentence in the dataset, length of the dataset,
            and a list of sentences.
        :return: The number of correctly predicted words, the total number of words, the times per sentence
            that it has taken to compute the weights, and the name of the file with best-path predictions for the batch.
        """
        start_num, len_data, sents = input_sents
        return self._decode_batch(sents=sents, start_num=start_num, len_data=len_data, kbest=True, with_weights=False)

    def _write_hypotheses_to_file(self, epoch: int,
                                  all_hypotheses: List[str], all_words: List[str], all_weights: List[float]) -> None:
        """
        Write to file this epoch's predictions for all hypotheses with their weights (> self.TOL).
        :param epoch: The ordinal number of the epoch.
        :param all_hypotheses: All hypotheses.
        :param all_words: All source words.
        :param all_weights: All weights for each (source, hypothesis) pair.
        """
        hyp_fn = Path(self.all_hypotheses_fn_format.format(epoch))
        with hyp_fn.open(mode='w', encoding=ENCODING) as w:
            csv.writer(w, delimiter='\t').writerows((s, t, NT_DEFAULT_FEATURE, weight) for s, t, weight
                                                    in zip(all_hypotheses, all_words, all_weights))

        print(f'Wrote weighted hypotheses to file {hyp_fn} in '
              f'"source word \t target word \t {NT_DEFAULT_FEATURE} \t weight" format.')

    def train(self) -> None:
        """
        Training.
        """
        self._pretrain()

        best_devset_acc = 0.
        best_devset_epoch = -1
        best_trainset_acc = 0.
        best_trainset_epoch = -1
        prev_weights: Optional[np.ndarray] = None

        self.current_model = self.model

        for epoch in range(self.epochs):
            accum_weights = np.zeros(self.WEIGHTS_DIMENSIONS)
            correct = 0
            total = 0
            times: List[float] = []
            batch_result_fns: List[str] = []
            print('Starting epoch %d ...' % epoch)

            if epoch == 0 and self.dummy_model:
                print('Using dummy model in epoch 0...')
                self.current_model = self.dummy_model
            else:
                self.current_model = self.model

            print('... Using channel: ', self.current_model.channel)

            dev_acc, dev_acc_at_k = self._decode(dataset=self.devset, epoch=epoch, kbest=True)
            if dev_acc >= best_devset_acc:
                best_devset_acc = dev_acc
                best_devset_epoch = epoch
                print(f'This iteration ({epoch}) has the highest devset accuracy so far.')
                self.best_param_path = self.current_model.channel.best_param_path(epoch - 1)
                self.final_params_fn.write_text(str(self.best_param_path) + '\n')

            start_time = time.time()
            # We need to compute the following:
            #
            #     for each word w and each of its candidates c: sum_{c} p(c | w) log p(w | c)
            #
            # Although we could directly optimize the following:
            #
            #     sum_{s \in Sentences} sum{w \in s} sum{c \in Cand(w)} weight(c, w) log p(w | c)
            #
            # it's easier to make the summations where possible and drop some (word, candidate) pairs whose p(c | w)
            # is below TOL tolerance level.
            #
            # Therefore, we compute p(c, w) = sum_t p(c, t, w), where p(c, t, w) is a single contextualized weight.
            # Then p(c | w) = p(c, w) / sum_k p(k, w).
            #
            # Thus, we will summarize all (c, w) pair weights as we iterate over sentences, and then divide that by
            # the sum over all candidates for w.
            if self.num_threads is not None and self.num_threads > 1 and self.len_data:
                try:
                    scheduler_queue(lockfile_name=os.environ.get("CNNC_MULTIPROCESSING_RUNNING",
                                                                 "CNNC_MULTIPROCESSING_RUNNING"))
                    print('Computing posterior in %d threads...' % self.num_threads)

                    # chunk the data into the number of threads
                    step_size = min(100, math.ceil(self.len_data / self.num_threads))
                    print('Chunk size is %d...' % step_size)
                    grouped_segs = [
                        (i, self.len_data, self.dataset[i:i + step_size]) for i in range(0, self.len_data, step_size)]
                    job_not_done = True
                    cur_num_threads = self.num_threads
                    while job_not_done:
                        try:
                            pool = multiprocessing.Pool(cur_num_threads)
                            results = pool.map(self._collect_weights, grouped_segs)
                            job_not_done = False
                            pool.terminate()
                            pool.join()
                        except (OSError, MemoryError) as e:
                            print(e)
                            old_num_threads = cur_num_threads
                            cur_num_threads = max(1, cur_num_threads - 1)
                            print(f"Attempt to reduce the number of parallel threads from {old_num_threads} to {cur_num_threads}")

                    for batch_result in results:
                        accum_weights += batch_result.weights
                        correct += batch_result.correct
                        total += batch_result.total
                        times.extend(batch_result.times)
                        batch_result_fns.append(batch_result.batch_result_fn)

                    # MOVE BATCH RESULTS INTO ONE RESULTS FILE
                    with open(self.train_results_fn_format.format(epoch), mode='w', encoding=ENCODING) as results_w:
                        for batch_result_fn in batch_result_fns:
                            with open(batch_result_fn, encoding=ENCODING) as batch_result_f:
                                print(f'Copying batch from {batch_result_fn} to {results_w.name}')
                                shutil.copyfileobj(batch_result_f, results_w)
                            print(f'Cleanup {batch_result_fn}')
                            os.unlink(batch_result_fn)
                finally:
                    scheduler_dequeue(lockfile_name=os.environ.get("CNNC_MULTIPROCESSING_RUNNING",
                                                                   "CNNC_MULTIPROCESSING_RUNNING"))
            else:
                # SINGLE THREAD
                with open(self.train_results_fn_format.format(epoch), mode='w', encoding=ENCODING) as results_w:

                    for sent_num, (orig_words, words, ref) in enumerate(self.dataset):
                        if self.verbose:
                            print(sent_num, words, '...')
                        then = time.time()
                        # @TODO return weights in LOG
                        results = self.current_model.decode(source_words=words, origin_words=orig_words,
                                                            map_predict=True)
                        best_path = results.best_path
                        times.append(time.time() - then)

                        for word, word_t_candidates in enumerate(results.candidates):
                            if self.verbose:
                                scored_cands = [(s, results.weights[i, word]) for i, s in enumerate(word_t_candidates)]
                                print('\t%d. "%s" =>' % (word, words[word]),
                                      ', '.join('"%s"/%.3f' % (s, w) for s, w in sorted(scored_cands,
                                                                                        key=lambda x: x[1],
                                                                                        reverse=True)[:15]))
                            word_index = self.WORD_VOCAB[words[word]]
                            for k, candidate in enumerate(word_t_candidates):
                                candidate_index = self.CAND_VOCABS[word_index][candidate]
                                accum_weights[candidate_index, word_index] += results.weights[k, word]

                        correct += sum(p.lower() == t.lower() if self.lowercase else p == t
                                       for p, t in zip(best_path, ref))
                        total += len(words)

                        # SOME REPORTING
                        if self.verbose:
                            print_report(words, best_path, ref, sent_num, self.len_data)

                        # WRITE RESULTS TO FILE
                        csv.writer(results_w, delimiter='\t').writerows(list(zip(orig_words, best_path)) + [('', '')])

            epoch_compute_time = time.time() - start_time
            epoch_acc = 100 * np.mean(correct / total) if total else 1.
            print('Epoch %d accuracy: %.3f. Finished in %.3f sec.' % (epoch, epoch_acc, epoch_compute_time))
            with open(self.report_fn, 'a') as a:
                # @TODO add all settings
                a.write('Epoch %d train accuracy:\t%.3f. Compute time:\t%.3f sec.\n' %
                        (epoch, epoch_acc, epoch_compute_time))
            if epoch_acc >= best_trainset_acc:
                best_trainset_acc = epoch_acc
                best_trainset_epoch = epoch
                print(f'This iteration ({epoch}) has the highest unsupervised trainset accuracy so far.')

            # ALWAYS ADD UNK SCORE
            word_index = self.WORD_VOCAB[UNK]
            candidate_index = self.CAND_VOCABS[word_index][UNK]
            accum_weights[candidate_index, word_index] += 1.

            # NORMALIZE WEIGHTS
            accum_weights /= accum_weights.sum(axis=0)

            # SANITY CHECK ON WEIGHTS
            if self.sup_direct_dataset:
                check_sum = accum_weights.sum(axis=0)
                sup_words = {word for word, target in self.sup_direct_dataset}
                for word in self.WORD_VOCAB:
                    word_index = self.WORD_VOCAB[word]
                    check_weight = check_sum[word_index]
                    assert np.isclose(check_weight, 1.) or (np.isnan(check_weight) and word in sup_words), \
                        ('Unexpected weight: ', check_weight, word, check_sum)
            else:
                assert np.allclose(accum_weights.sum(axis=0), np.ones(len(self.WORD_VOCAB)))

            all_words = []
            all_hypotheses = []
            all_weights = []

            # IF SEMI-SUPERVISED, SCALE UNSUPERVISED DATA AND ADD SUPERVISED DATA INTO TRAIN
            if self.sup_direct_dataset:
                accum_weights *= self.alpha_unsupervised
                for word, target in self.sup_direct_dataset:
                    # @TODO think about alternative ways of weight aggregation
                    all_words.append(word)
                    all_hypotheses.append(target)
                    all_weights.append(1.)

            # FLATTEN OUT AND FILTER UNSUPERVISED DATA
            for word in self.WORD_VOCAB:
                word_index = self.WORD_VOCAB[word]
                cand_vocab = self.CAND_VOCABS[word_index]
                for candidate in cand_vocab:
                    weight = accum_weights[cand_vocab[candidate], word_index]
                    if weight < self.tol or np.isnan(weight):
                        # ignore too low-weight samples or weights for words only from the supervised data
                        continue
                    all_words.append(word)
                    all_hypotheses.append(candidate)
                    all_weights.append(weight)

            print(f'Processed {self.len_data} sentences at an average speed of {np.mean(times):.3f} sec per sentence.')
            print(f'EM... with {len(all_hypotheses)} weighted samples.')
            start_em_time = time.time()

            if self.write_hypotheses_to_file:
                self._write_hypotheses_to_file(epoch, all_hypotheses, all_words, all_weights)

            train_params = dict(self.channel_update_params)

            if prev_weights is not None and np.allclose(accum_weights[~np.isnan(accum_weights)], prev_weights):
                print('** NB: Channel hasn\'t been updated from previous epoch. Will restart channel training.')
                train_params['reload'] = False
            else:
                prev_weights = np.copy(accum_weights[~np.isnan(accum_weights)])
                print('Will train with channel update params.')
                train_params['reload'] = self.channel_update_params['reload']

            self.channel.update_model(sources=all_hypotheses, targets=all_words, weights=all_weights,
                                      epoch_number=epoch, **train_params)

            print('Finished EM in %.3f sec.' % (time.time() - start_em_time))

        if self.epochs > 0:
            # final pass over dev set with the updated parameters
            self.current_model = self.model
            print('... Using channel: ', self.current_model.channel)
            epoch = self.epochs
            dev_acc, dev_acc_at_k = self._decode(dataset=self.devset, epoch=epoch, kbest=True)
            if dev_acc >= best_devset_acc:
                best_devset_acc = dev_acc
                best_devset_epoch = epoch
                print(f'This iteration ({epoch}) has the highest devset accuracy so far.')
                self.best_param_path = self.current_model.channel.best_param_path(epoch - 1)
                self.final_params_fn.write_text(str(self.best_param_path) + '\n')
            print(f'Finished training. Best iteration: {best_devset_epoch} with devset acc {best_devset_acc:.3f}.')
            print(f'(Highest unsupervised trainset acc: {best_trainset_acc:.3f}, iteration={best_trainset_epoch})')
            best_epoch = best_devset_epoch - 1  # params are from epoch - 1
            print(f'Best parameters are trained in epoch {best_epoch}')
        else:
            print('... will evaluate the pretrained model or model loaded from elsewhere.')
            # params are from either pretraining (-1) or candidate scores / sed params from a file
            self.best_param_path = self.current_model.channel.best_param_path(-1, **self.channel_update_params)
            self.final_params_fn.write_text(str(self.best_param_path) + '\n')
        epoch_reporting = 'best'
        self.current_model.channel.reset_from_path(self.best_param_path)
        self.kbest = self.final_kbest
        print(f'... will return k-best predictions with k = {self.kbest}')
        with self.report_fn.open('a') as a:
            a.write('\n' + '#'*80 + '\n')
        self._decode(dataset=self.devset, epoch=epoch_reporting, kbest=True)
        if self.testset:
            print(f'Will decode the test set ({self.path2test}) using the best parameters found on the dev set.')
            self._decode(dataset=self.testset, epoch=epoch_reporting, kbest=True)
        if self.testset2:
            print(f'Will decode test set 2 ({self.path2another_test}) using the best parameters found on the dev set.')
            self._decode(dataset=self.testset2, epoch=epoch_reporting, kbest=True)
        print('Done.')
