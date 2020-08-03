import os
import re
import shutil
import time

import editdistance
import math
import json
import subprocess
import multiprocessing
import uuid

import numpy as np

from contextualized_transduction.reranker import SimpleLanguageModel, WordReranker
from contextualized_transduction.lm import KenLM
from contextualized_transduction.bigram_kenlm import BigramKenLM
from contextualized_transduction.trigram_kenlm import TrigramKenLM
from nn_lm.custom_lm import WordLanguageModel, CharLanguageModel
from contextualized_transduction.utils import UNK, NT_DEFAULT_FEATURE

from pathlib import Path
from typing import Union, ClassVar, Pattern, Optional, Dict, Tuple, List

from contextualized_transduction.utils import ENCODING, scheduler_queue, scheduler_dequeue

DUMMY_TARGET = '࿋࿋࿋'
UNK_NNWORD_SCORE = -18.0

class DirectCandidates:
    ERE: ClassVar[Pattern] = re.compile(r'^neural_channel_e(?P<e>[\d-]+)$')
    PATH2TRAIN_DIRECT: ClassVar[str] = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                                    "scripts/train_direct.sh")
    PATH2DECODER: ClassVar[str] = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                               "scripts/decode_direct.sh")
    PATH2DECODER_MULTI: ClassVar[str] = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                                     "scripts/decode_direct_multithread.sh")

    def __init__(self, direct_result_dir: Union[str, Path], channel_dir: Union[str, Path],
                 train_params: Dict, decode_params: Dict, path2neural_code: Union[str, Path, None] = None,
                 reload_path: Union[str, Path, None] = None, test: Union[str, Path, None] = None,
                 test2: Union[str, Path, None] = None, language_model: Optional[str] = None,
                 language_model_params: Optional[Dict] = None, reranker_params: Optional[dict] = None,
                 decode_threads: Optional[int] = None, *args, **kargs):
        """
        Read in a directory with a model trained with `il_replementation.new_trainer`; use the best epoch's weighted
        candidates file (=q(modern | historical)) as training data to fit a direct model (possibly reload); generate
        beam width k of candidates for all train / dev / test files and store them into med format text file; and / or
        score med candidates add select m top candidates; @TODO Use a channel and a language model to produce
        scores for beam candidates to select a fitting subset of them; @TODO use scores as train signal for REINFORCE
        updates and further train the direct model.
        :param direct_result_dir: Directory where to store the direct model and any its intermiediate files.
        :param channel_dir: Directory of the noisy channel model.
        :param train_params: Hyperparameters for training.
        :param decode_params: Hyperparameters for decoding.
        :param path2neural_code: Optionally, the path to the neural transducer project.
        :param reload_path: Optionally, the path to the direct model to start training from.
        :param test: Optionally, path to a test file to decode with beam search.
        :param test2: Optionally, path to a another test file to decode with beam search.
        :param language_model: Optionally, language model class, in case LM is used to score candidates.
        :param language_model: Optionally, language model parameters (e.g. path to parameters).
        :param reranker_params: Optionally, hyperparameters for reranking beam candidates.
        :param decode_threads: How many threads to use for decoding.
        """

        self.direct_result_dir = Path(direct_result_dir)
        self.channel_dir = Path(channel_dir)
        self.reload_path = Path(reload_path) if reload_path else None
        self.test = Path(test) if test else None
        self.test2 = Path(test2) if test2 else None

        self.path2neural_code = str(path2neural_code) if path2neural_code else None
        if self.path2neural_code:
            assert os.path.exists(self.path2neural_code)
            os.environ["PATH2NEURAL_CODE"] = self.path2neural_code

        self.language_model_params = language_model_params
        if language_model is None:
            path2model = (self.language_model_params.get('apra_fn') or
                          self.language_model_params.get('path2model') or
                          self.language_model_params.get('model'))
            try:
                self.language_model = SimpleLanguageModel(path2model=path2model)
            except OSError:
                self.language_model_class = None
                self.language_model = None
        else:
            self.language_model_class = eval(language_model)
            try:
                self.language_model = self.language_model_class.load_language_model(**self.language_model_params)
            except TypeError:
                path2model = (self.language_model_params.get('apra_fn') or
                              self.language_model_params.get('path2model') or
                              self.language_model_params.get('model'))
                self.language_model = SimpleLanguageModel(path2model=path2model)

        self.train_params = train_params
        self.decode_params = decode_params
        self.reranker_params = reranker_params if reranker_params else dict()

        self.decode_threads = decode_threads if decode_threads else 1

        self._prepare_data()

    def _prepare_data(self):
        """
        Prepare direct model's train, dev, and decoding data.
        """

        def reverse_tsv(src_tsv: Path, dst_tsv: Path):
            """
            Write tsv in the correct order for direct model training: SOURCE\tTARGET\tFEAT\TWEIGHT
            :param src_tsv: TSV file in channel order.
            :param dst_tsv: TSV file in direct order.
            """
            if not src_tsv.exists():
                dst_tsv.touch()
                return
            with src_tsv.open(encoding=ENCODING) as f, \
                    dst_tsv.open(mode='w', encoding=ENCODING) as w:
                for line in f:
                    # saft    safft   #       1.0
                    target, source, *feat_weight = line.rstrip().split('\t')
                    w.write('\t'.join((source, target, *feat_weight)) + '\n')

        self.direct_result_dir.mkdir()

        # new paths
        self.train_tsv = self.direct_result_dir / 'full_train.tsv'
        self.sup_train_tsv = self.direct_result_dir / 'train.tsv'
        self.dev_tsv = self.direct_result_dir / 'dev.tsv'
        self.candidates_tsv = self.direct_result_dir / 'candidates.tsv'
        self.beam_cand_json = self.direct_result_dir / 'beam_candidates.json'  # nothing here yet
        self.beam_candidates_tsv = self.direct_result_dir / 'beam_candidates.tsv'
        self.beam_k_candidates_template = str(self.direct_result_dir / 'beam{}_candidates.tsv')
        self.rest_cand_json = self.direct_result_dir / 'scored_candidates.json'  # nothing here yet

        with (self.channel_dir / 'best_model.path').open(encoding=ENCODING) as f:
            best_model_dec_scores_path = Path(f.read().strip())

        self.best_model_dir = self.channel_dir / best_model_dec_scores_path.parent.stem
        assert self.best_model_dir.exists(), (self.channel_dir, best_model_dec_scores_path, self.best_model_dir)

        # identify neural channel training epoch that produces best model
        epoch = int(self.ERE.match(self.best_model_dir.stem).groupdict()['e'])
        assert epoch + 1 > -1, (epoch, self.best_model_dir)

        # TRAIN: best posterior weights come from epoch after best
        try:
            posterior_weights_tsv = self.channel_dir / f'channel_e{epoch + 1}.tsv'
            assert posterior_weights_tsv.exists()
        except AssertionError as e:
            print('The best dev scores are from last - 1 epoch. Unfortunately, we have not produced unsupervised '
                  f'train weights with this channel (\'channel_e{epoch + 1}.tsv\'). You need to manually change '
                  '\'best_model.path\' to refer to a different good epoch whose \'channel_e....tsv\' exists.')
            raise e
        reverse_tsv(posterior_weights_tsv, self.train_tsv)

        # DEV: reverse *channel.txt
        channel_dev = list(self.channel_dir.glob('*.channel.*'))
        assert len(channel_dev) == 1, channel_dev
        reverse_tsv(channel_dev[0], self.dev_tsv)

        # SUP_TRAIN: only supervised training samples
        channel_sup_train = self.channel_dir / 'neural_channel_e-001/train.tsv'
        reverse_tsv(channel_sup_train, self.sup_train_tsv)

        # SCORES: in CHANNEL order !!! --- score all candidates generated with some other method (e.g. MED)
        chanel_candidates = self.channel_dir / 'candidates.tsv'
        reverse_tsv(chanel_candidates, self.candidates_tsv)

        # BEAM CANDS: types of all historical words (train, dev, test, ...) --- generate candidates for them.
        self.types2score_tsv = self.direct_result_dir / 'types2score.tsv'
        with self.types2score_tsv.open(mode='w', encoding=ENCODING) as w:
            source_types = set()
            for tsv, copy_tsv in ((self.sup_train_tsv, True), (self.dev_tsv, True), (self.candidates_tsv, False)):
                with tsv.open(encoding=ENCODING) as f:
                    if copy_tsv:
                        # copy everything from sup train and dev. This preserves target ambiguity.
                        for line in f:
                            source, target, *feat_weight = line.rstrip().split('\t')
                            source_types.add(source)
                            assert feat_weight[0] == NT_DEFAULT_FEATURE, (feat_weight, NT_DEFAULT_FEATURE)
                            line = f'{source}\t{target}\t{NT_DEFAULT_FEATURE}\n'
                            w.write(line)
                    else:
                        # take only source types from unsup train and test sets.
                        for line in f:
                            source, *_ = line.rstrip().split('\t')
                            if source not in source_types and source != UNK:
                                line = f'{source}\t{DUMMY_TARGET}\t{NT_DEFAULT_FEATURE}\n'
                                w.write(line)
                                source_types.add(source)

        for path_obj, name in ((self.direct_result_dir, 'direct_result_dir'), (self.train_tsv, 'train_tsv'),
                               (self.sup_train_tsv, 'sup_train_tsv'), (self.dev_tsv, 'dev_tsv'),
                               (self.test, 'test'), (self.test2, 'test2'), (self.path2neural_code, 'path2neural_code'),
                               (self.candidates_tsv, 'candidates_tsv'), (self.beam_cand_json, 'beam_cand_json'),
                               (self.beam_candidates_tsv, 'beam_candidates_tsv'),
                               (self.rest_cand_json, 'rest_cand_json'), (self.best_model_dir, 'best_model_dir'),
                               (self.types2score_tsv, 'types2score_tsv')):
            print(f'{name: <20} : {path_obj}')

    @classmethod
    def from_config(cls, config: Dict):

        # paths
        direct_result_dir = config['direct_result_dir']
        channel_dir = config['channel_dir']
        reload_path = config.get('reload_path')
        path2neural_code = config.get('path2neural_code')
        test = config.get('test')
        test2 = config.get('test2') or config.get('test2')

        # class
        language_model = config.get('language_model')

        # dicts of params
        language_model_params = config.get('language_model_params')
        train_params = config.get('train_params')
        decode_params = config.get('decode_params')
        reranker_params = config.get('reranker_params')

        decode_threads = config.get('num_threads')

        return cls(direct_result_dir, channel_dir, train_params, decode_params, path2neural_code, reload_path,
                   test, test2, language_model, language_model_params, reranker_params, decode_threads)

    def _train_decode(self):

        # set some environment variables for training
        os.environ["PATIENCE"] = str(self.train_params.get('patience', 100))
        os.environ["EPOCHS"] = str(self.train_params.get('epochs', 5))
        os.environ["SEED"] = str(self.train_params.get('dynet_seed', 1))
        os.environ["ILK"] = str(self.train_params.get('il_k', 12))
        os.environ["TRAIN_BATCH_SIZE"] = str(self.train_params.get('batch_size', 20))
        os.environ["PICK_LOSS"] = '--pick-loss' if self.train_params.get('pick_loss') else ''
        os.environ["TRAIN"] = str(self.train_tsv)
        os.environ["DEV"] = str(self.dev_tsv)

        mle_dir = self.direct_result_dir / 'base'
        os.environ["RESULTS"] = str(mle_dir)
        mle_dir.mkdir()

        os.environ["TRAIN_RELOAD"] = str(self.reload_path) if self.reload_path else ''
        os.environ["LOGFILE"] = str(mle_dir / 'train.log')

        timeout = self.train_params.get('timeout', 12)

        # TRAINING
        print(f'Training the direct model for {os.environ["EPOCHS"]} epochs with patience '
              f'{os.environ["PATIENCE"]}...')
        start_train = time.time()
        return_code = subprocess.call(["timeout", "{}h".format(timeout), "bash", self.PATH2TRAIN_DIRECT])
        finish_train = (time.time() - start_train) / 60
        if return_code == 0:
            print('Finished training in {:.1f} min.'.format(finish_train))
            print('Decoding...')
        else:
            print('\n\n*** Training terminated with error or by user. Trained for {:.1f} min. ***.'
                  '\n\nDecoding...'.format(finish_train))

        # DECODING
        def repackage_json(table: Dict, big_table: Dict):
            """
            Repackage targets into a list of targets.
            :param table: Table.
            :param big_table: Resulting table.
            """
            for source_type, feats_v in table.items():
                for feat, v in feats_v.items():
                    if source_type in big_table:
                        # this source possibly occurs with a different target
                        # because of later json dump, avoid set...
                        if v['target'] not in big_table[source_type]['target']:
                            big_table[source_type]['target'].append(v['target'])
                            big_table[source_type]['feats'].append(feat)
                    else:
                        big_table[source_type] = dict(v)
                        # {'candidates': candidates, 'log_prob': log_prob, 'acts': pred_acts, 'target': sample.word_str}
                        big_table[source_type]['target'] = [v['target']]
                        big_table[source_type]['feats'] = [feat]

        self.beam_width = self.decode_params.get('beam_width', 4)
        os.environ["RESULT_SUFFIX"] = f'{self.beam_width}_all'
        os.environ["LOGFILE"] = str(mle_dir / 'decoding.log')
        os.environ["BEAM_WIDTH"] = str(self.beam_width)
        os.environ["TYPES2DECODE_TSV"] = str(self.types2score_tsv)
        os.environ["MODE"] = 'beam'

        decoding_result_json = mle_dir / f'dev_beam{os.environ["RESULT_SUFFIX"]}.json'

        if self.decode_threads <= 1:
            return_code = subprocess.call(["bash", "-x", self.PATH2DECODER])
            assert return_code == 0

            with decoding_result_json.open(encoding=ENCODING) as f:
                table = json.load(f)

            big_table = dict()
            repackage_json(table, big_table)
        else:
            try:
                scheduler_queue(lockfile_name=os.environ.get("CNNC_MULTIPROCESSING_RUNNING",
                                                             "CNNC_MULTIPROCESSING_RUNNING"))
                print('Decoding using {} threads'.format(self.decode_threads))
                tmp_dir = os.path.join('/tmp', uuid.uuid4().hex)
                os.makedirs(tmp_dir)
                print('Temporary decoding directory: {}'.format(tmp_dir))
                os.environ["RESULTS"] = tmp_dir
                os.environ["RELOAD"] = str(mle_dir)
                line_count = 0
                with self.types2score_tsv.open(encoding=ENCODING) as f:
                    for _ in f:
                        line_count += 1
                    f.seek(0)
                    step_size = math.ceil(line_count / self.decode_threads)
                    print('Will split "{}" ({} lines) into {} chunks of size {}'.format(
                        self.types2score_tsv, line_count, self.decode_threads, step_size))
                    tmp_candidate_fn = os.path.join(tmp_dir, 'candidates0000')
                    tmp_candidate_fns = [(tmp_candidate_fn, '0000')]
                    w = open(tmp_candidate_fn, mode='w', encoding=ENCODING)
                    file_count = 1
                    file_line_count = 0
                    for line in f:
                        if file_line_count == step_size:
                            w.close()
                            tmp_candidate_fn = os.path.join(tmp_dir, 'candidates{:04d}'.format(file_count))
                            tmp_candidate_fns.append((tmp_candidate_fn, '{:04d}'.format(file_count)))
                            w = open(tmp_candidate_fn, mode='w', encoding=ENCODING)
                            file_count += 1
                            file_line_count = 0
                        file_line_count += 1
                        w.write(line)
                    w.close()
                assert len(tmp_candidate_fns) == self.decode_threads, (tmp_candidate_fns, '!=', self.decode_threads)
                os.environ["DECODE_THREAD"] = str(self.decode_threads)
                results = self._launch_decoder(tmp_candidate_fns)
                # repackage predictions into one dictionary + repackage targets into list
                big_table = dict()
                for return_code, (tmp_candidate_fn, suffix) in zip(results, tmp_candidate_fns):
                    assert return_code == 0, 'Decoding failed: {}'.format(tmp_candidate_fn)
                    json_output = os.path.join(tmp_dir, 'dev_beam{}.json'.format(suffix))
                    with open(json_output, encoding=ENCODING) as w:
                        table = json.load(w)
                    repackage_json(table, big_table)
                # cleanup temporary files
                print(f'Cleaning up temporary directory {tmp_dir}...')
                shutil.rmtree(tmp_dir)
            finally:
                scheduler_dequeue(lockfile_name=os.environ.get("CNNC_MULTIPROCESSING_RUNNING",
                                                               "CNNC_MULTIPROCESSING_RUNNING"))

        with self.beam_cand_json.open(mode='w', encoding=ENCODING) as w:
            json.dump(big_table, w, indent=1, ensure_ascii=False)

        print('Finished rescoring.')
        self.reload_path = mle_dir

    def _launch_decoder(self, tmp_candidate_fns: List[Tuple[str, str]]) -> List[int]:
        processes = set()
        for k, (tmp_candidate_fn, suffix) in enumerate(tmp_candidate_fns):
            env = os.environ.copy()
            env['CANDIDATES_TSV'] = tmp_candidate_fn
            env['RESULT_SUFFIX'] = suffix
            processes.add(
                (k, subprocess.Popen(["bash", "-x", self.PATH2DECODER_MULTI], env=env))
            )
        results = []
        for k, p in processes:
            p.wait()
            results.append(p.returncode)
        return results

    def _launch_lm(self, i_total_cands: Tuple[int, int, List[Tuple[str, List[str]]]]) -> Dict[str, List[float]]:
        i, total, cands = i_total_cands
        lm_scores = dict()
        if isinstance(self.language_model, SimpleLanguageModel):
            for source_type, candidates in cands:
                lm_scores[source_type] = self.language_model.score_batch(candidates)
        else:
            for source_type, candidates in cands:
                states_lm_scores = self.language_model.score_word_batch(candidates,
                                                                        state=None,
                                                                        unk_score=UNK_NNWORD_SCORE)
                _, scores = tuple(zip(*states_lm_scores))
                # because json cannot serialize numpy array and dtype np.float32 (does this come from pytorch?)
                lm_scores[source_type] = [float(s) for s in scores]
        print(f'\t...finished scoring {i}-{i + len(cands)}/{total} candidate sets')
        return lm_scores

    def _score_beam(self):

        with self.beam_cand_json.open(encoding=ENCODING) as f:
            big_table = json.load(f)
        len_big_table = len(big_table)
        cands = [(source_type, v['candidates']) for source_type, v in big_table.items()]

        # 1. LM SCORES
        try:
            scheduler_queue(lockfile_name=os.environ.get("CNNC_MULTIPROCESSING_RUNNING",
                                                         "CNNC_MULTIPROCESSING_RUNNING"))
            print(f'Scoring beam candidates with LM using {self.decode_threads} threads...')

            step_size = math.ceil(len_big_table / self.decode_threads)
            grouped_source_types = [
                (i, len_big_table, cands[i:i + step_size]) for i in range(0, len_big_table, step_size)]
            pool = multiprocessing.Pool(self.decode_threads)
            results = pool.map(self._launch_lm, grouped_source_types)
            pool.terminate()
            pool.join()
        finally:
            scheduler_dequeue(lockfile_name=os.environ.get("CNNC_MULTIPROCESSING_RUNNING",
                                                           "CNNC_MULTIPROCESSING_RUNNING"))
        for lm_scores in results:
            for source_type, scores in lm_scores.items():
                big_table[source_type]['lm_scores'] = scores

        # 2. CHANNEL SCORES
        try:
            scheduler_queue(lockfile_name=os.environ.get("CNNC_MULTIPROCESSING_RUNNING",
                                                         "CNNC_MULTIPROCESSING_RUNNING"))
            print(f'Scoring beam candidates with CHANNEL using {self.decode_threads} threads...')
            tmp_dir = Path('/tmp') / uuid.uuid4().hex
            tmp_dir.mkdir()
            print(f'Temporary decoding directory: {tmp_dir}')
            os.environ["RESULTS"] = str(tmp_dir)
            os.environ["RELOAD"] = str(self.best_model_dir)
            os.environ["MODE"] = 'channel'
            line_count = len([_ for source_type, candidates in cands for _ in candidates])
            step_size = math.ceil(line_count / self.decode_threads)
            print(f'Will split beam candidates ({line_count} lines) into '
                  f'{self.decode_threads} chunks of size {step_size}')

            tmp_candidate_fn = tmp_dir / 'candidates0000'
            tmp_candidate_fns = [(tmp_candidate_fn, '0000')]
            w = tmp_candidate_fn.open(mode='w', encoding=ENCODING)
            file_count = 1
            file_line_count = 0
            for source_type, candidates in cands:
                for candidate in candidates:
                    if file_line_count == step_size:
                        w.close()
                        tmp_candidate_fn = tmp_dir / f'candidates{file_count:04d}'
                        tmp_candidate_fns.append((tmp_candidate_fn, f'{file_count:04d}'))
                        w = tmp_candidate_fn.open(mode='w', encoding=ENCODING)
                        file_count += 1
                        file_line_count = 0
                    file_line_count += 1
                    line = f'{candidate}\t{source_type}\t{NT_DEFAULT_FEATURE}\n'
                    w.write(line)
            w.close()
            assert len(tmp_candidate_fns) == self.decode_threads, (tmp_candidate_fns, '!=', self.decode_threads)
            os.environ["DECODE_THREAD"] = str(self.decode_threads)
            results = self._launch_decoder(tmp_candidate_fns)
            # put predictions into big table json
            broken_beam = dict()
            for return_code, (tmp_candidate_fn, suffix) in zip(results, tmp_candidate_fns):
                assert return_code == 0, f'Decoding failed: {tmp_candidate_fn}'
                json_output = tmp_dir / f'dev_channel{suffix}.json'
                with json_output.open(encoding=ENCODING) as w:
                    table = json.load(w)
                for source_type, v in table.items():
                    v_candidates = v['candidates']
                    v_log_prob = v['log_prob']
                    len_v_candidates = len(v_candidates)
                    len_big_table_candidates = len(big_table[source_type]['candidates'])
                    if len_big_table_candidates > len_v_candidates:
                        # something is missing in table's candidates:
                        # if a source type has got broken between two files, some candidate from big_table
                        # will not be found. Collect such broken beams and add them to big_table later.
                        print('Possibly broken beam: ', source_type, v)
                        if source_type in broken_beam:
                            broken_beam[source_type]['candidates'].extend(v_candidates)
                            broken_beam[source_type]['log_prob'].extend(v_log_prob)
                        else:
                            broken_beam[source_type] = dict(candidates=v_candidates, log_prob=v_log_prob)
                    elif len_big_table_candidates == len_v_candidates:
                        channel_scores = []
                        for candidate in big_table[source_type]['candidates']:
                            cand_index = v_candidates.index(candidate)
                            channel_scores.append(v_log_prob[cand_index])
                        big_table[source_type]['channel_scores'] = channel_scores
                    else:
                        raise Exception(
                            f'More candidates than expected: {len_big_table_candidates} vs {len_v_candidates}. '
                            f'{source_type, v} vs {big_table[source_type]}')
            # assert len(broken_beam) < (2 * self.decode_threads - 1), broken_beam
            # <= this assertion won't hold in general anymore because of candidate lists with few unique candidates
            # that match the first condition.
            for source_type, v in broken_beam.items():
                assert 'channel_scores' not in big_table[source_type], (source_type, broken_beam)
                v_candidates = v['candidates']
                v_log_prob = v['log_prob']
                channel_scores = []
                for candidate in big_table[source_type]['candidates']:
                    cand_index = v_candidates.index(candidate)
                    channel_scores.append(v_log_prob[cand_index])
                big_table[source_type]['channel_scores'] = channel_scores
            # cleanup temporary files
            print(f'Cleaning up temporary directory {tmp_dir}...')
            shutil.rmtree(tmp_dir)
        finally:
            scheduler_dequeue(lockfile_name=os.environ.get("CNNC_MULTIPROCESSING_RUNNING",
                                                           "CNNC_MULTIPROCESSING_RUNNING"))
        # 3. UPDATE BEAM JSON
        with self.beam_cand_json.open(mode='w', encoding=ENCODING) as w:
            json.dump(big_table, w, indent=1, ensure_ascii=False)

        # check data integrity
        for k, v in big_table.items():
            assert 'channel_scores' in v, (k, v)
            assert 'lm_scores' in v, (k, v)
            assert len(v['channel_scores']) == len(v['candidates']), (k, v)
            assert len(v['lm_scores']) == len(v['candidates']), (k, v)

    def _rerank_beam(self):
        # @TODO add `rank`, `reranker_scores`

        # 1. SPLIT BEAM CAND JSON INTO DEV JSON (WHERE WE FIT A RERANKER) AND REST JSON
        with self.beam_cand_json.open(encoding=ENCODING) as f:
            big_table = json.load(f)

        with self.dev_tsv.open(encoding=ENCODING) as f:
            dev_keys = set()
            for line in f:
                source, *_ = line.split('\t')
                dev_keys.add(source)

        dev_dec_json = self.direct_result_dir / 'dev_beam_candidates.json'
        with dev_dec_json.open(mode='w', encoding=ENCODING) as w:
            json.dump({k: big_table[k] for k in dev_keys}, w, indent=4, ensure_ascii=False)

        # 2. PREPARE RERANKER CONFIG
        reranker_config = dict(model_dir=self.direct_result_dir,
                               dev_dec_json=dev_dec_json,
                               test_dec_json=self.beam_cand_json,
                               num_threads=self.decode_threads,
                               output_ranks=True)

        # e.g. add other scores (char-rnn-lm) or PRO reranker params
        reranker_config.update(self.reranker_params)

        # 3. RERANK AND GET RANKS AND SCORES
        reranker = WordReranker(reranker_config)
        reranker.rerank()

        # 4. REASSIGN TO REORDERED BEAM CANDIDATES FILE
        self.beam_cand_json = self.direct_result_dir / 'f.rerank.test.ranks.json'

    def _build_med_style_from_beam(self):

        def build_lines(v, k: Optional[int] = None):
            v_slice = slice(None, k)
            for candidate, direct_score, lm_score, channel_score, reranker_score \
                    in zip(v['candidates'][v_slice], v['direct_scores'][v_slice], v['lm_scores'][v_slice],
                           v['channel_scores'][v_slice], v['reranker_scores'][v_slice]):
                # candidate\tsource\t....
                yield f'{source}\t{candidate}\t{direct_score}\t{lm_score}\t{channel_score}\t{reranker_score}\n'

        print('Outputting MED-style candidate file for beam candidates, including for all k...')
        cands_dict = dict()
        with self.beam_candidates_tsv.open(mode='w', encoding=ENCODING) as w, \
                self.beam_cand_json.open(encoding=ENCODING) as f:
            for source, v in json.load(f).items():
                if len(v['target']) > 1:
                    print('\t...Multiple targets:', source, v['target'])
                for line in build_lines(v):
                    w.write(line)
                cands_dict[source] = v['candidates']

        for k in range(1, self.beam_width):
            with open(self.beam_k_candidates_template.format(k), mode='w', encoding=ENCODING) as w, \
                    self.beam_cand_json.open(encoding=ENCODING) as f:
                for source, v in json.load(f).items():
                    for line in build_lines(v, k):
                        w.write(line)

        # compute coverage on dev
        with self.dev_tsv.open(encoding=ENCODING) as f:
            dev = []
            for line in f:
                source, target, *_ = line.split('\t')
                dev.append((source, target))

        acc_at_k = np.zeros(self.beam_width + 1)
        for source, target in dev:
            if source not in cands_dict:
                print(f'\t...Not covered by dev set: {source, target}')
                continue
            for k in range(1, self.beam_width + 1):
                acc_at_k[k] += int(target in cands_dict[source][:k])

        acc_at_k /= len(dev)
        acc_at_k *= 100
        for k in range(1, self.beam_width):
            print(f'Dev accuracy at {k} candidates: {acc_at_k[k]:.1f}')

        # additionally decode all tsvs with beam search
        for tsv_name, tsv in [('dev', self.dev_tsv), ('test', self.test), ('test2', self.test2)]:
            if tsv is not None and os.path.exists(tsv):
                correct = 0
                total = 0
                lev = []
                nlev = []
                predictions_fn = self.direct_result_dir / f'{tsv_name}.direct.predictions'
                with tsv.open(encoding=ENCODING) as f, \
                        predictions_fn.open(mode='w', encoding=ENCODING) as w:
                    for line in f:
                        if line.strip():
                            try:
                                source, target, feats, *_ = line.split('\t')
                                line_template = '{s}\t{t}\t{f}\n'
                            except ValueError:
                                source, target, *_ = line.replace('\n', '').split('\t')
                                line_template = '{s}\t{t}\n'
                                feats = None
                            if source in cands_dict and cands_dict[source]:
                                predict = cands_dict[source][0]
                            else:
                                print(f'\t...No prediction exists for: {source, target}')
                                predict = UNK
                            if target == predict:
                                correct += 1
                            else:
                                ed = editdistance.eval(target, predict)
                                lev.append(ed)
                                nlev.append(ed / len(target))
                            total += 1
                            line = line_template.format(s=source, t=predict, f=feats)
                        w.write(line)

                    print(f'{tsv_name} accuracy: {correct * 100 / total:.1f}')
                    with (self.direct_result_dir / f'{tsv_name}.direct.eval').open(mode='w', encoding=ENCODING) as w:
                        w.write('MODEL\tTAG\tACCURACY\tLEVDIST\tNORMLEVDIST\n')
                        w.write(f'{predictions_fn}\t---\t{correct / total}\t{np.mean(lev)}\t{np.mean(nlev)}\n')
            else:
                print(f'Skipping decoding for "{tsv_name}"...')

    def _score_external_candidates(self):
        print('Scoring externally generated candidates with the trained direct model not implemented.')
        pass

    def train_decode(self):

        # train on posterior weights from channel dir and beam decode
        self._train_decode()

        # score beam-decoded candidates with LM and channel
        self._score_beam()

        # score beam-decoded candidates with a reranker
        self._rerank_beam()

        # build a MED like file from `self.beam_cand_json`
        self._build_med_style_from_beam()

        # score elsewhere generated candidates with this direct model
        self._score_external_candidates()
