import json
import csv
import os
import sys
import uuid
import shutil
import time
import math
import multiprocessing
import subprocess
from pathlib import Path

from typing import Iterable, Any, List, Tuple, Generator, ClassVar, Optional, Set, Union

from contextualized_transduction.utils import ENCODING, NT_DEFAULT_FEATURE, scheduler_dequeue, scheduler_queue
from contextualized_transduction.sed_channel import Channel, StochasticEditDistance_Channel

import logging
log = logging.getLogger(__name__)


class OffLine_NT_Channel(Channel):

    PATH2NEURAL_MSTEP: ClassVar[str] = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                                    "scripts/run_neural_mstep.sh")
    PATH2DECODER: ClassVar[str] = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                                    "scripts/decode.sh")
    PATH2DECODER_MULTI: ClassVar[str] = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                                    "scripts/decode_multithread.sh")
    BACKOFF_CHANNEL: ClassVar[Channel] = StochasticEditDistance_Channel

    def __init__(self, path2candidate_scores: Optional[str],
                 generated_candidates_fn: Optional[str],
                 backoff_alphabet: Optional[Set],
                 em_data_dir: Union[str, Path] = '/tmp',
                 reload_path: Union[str, Path, None] = None,
                 channel_data_template: str = 'neural_channel_e{:04d}', *args, **kwargs):
        """
        This model is an off-line version of Conditional Neural Transducer Channel. This is only useful
        for working with a fixed dataset and experiments.
        The model backs off to some simple channel model on the initial iteration. Then a neural channel
        is trained off-line and all candidates are scored off-line. Scoring candidates on-line then amounts
        to dictionary lookup.
        :param path2candidate_scores: Path to json that maps corrupted word into candidates with scores.
        :param generated_candidates_fn: Path to tsv in sigm2017 format that lists all candidates.
        :param backoff_alphabet: In case `path2candidate_scores` doesn't point to computed scores, which
            alphabet to use with the backoff channel.
        :param em_data_dir: Directory to which to write intermediate EM training results.
        :param reload_path: Path from where to load the channel model and vocabulary. Useful for resuming an
            interrupted model training.
        :param channel_data_template: Template for directory that bundles all EM training results for one EM epoch.
        """
        self.path2candidate_scores = path2candidate_scores
        self.generated_candidates_fn = generated_candidates_fn
        os.environ["CANDIDATES_TSV"] = str(self.generated_candidates_fn)
        self.big_table = None

        if self.path2candidate_scores is not None and os.path.exists(self.path2candidate_scores):
            self.reset_from_path(self.path2candidate_scores)

            if self.generated_candidates_fn is None or not os.path.exists(self.generated_candidates_fn):
                # raise NotImplementedError('TODO: Write the required tsv from `path2candidate_scores`')
                pass
        else:
            print('File with candidates scores does not exist: ', self.path2candidate_scores,
                  'Warm-starting with ', self.BACKOFF_CHANNEL)

            self.source_albhabet = backoff_alphabet
            self.target_alphabet = backoff_alphabet
            self.backoff = self.BACKOFF_CHANNEL(source_alphabet=backoff_alphabet,
                                                target_alphabet=backoff_alphabet,
                                                smart_init=True)

        self.em_data_dir = em_data_dir
        self.channel_data_template = channel_data_template
        self.checksum: Optional[bytes] = None

        if reload_path:
            reload_path = Path(reload_path)
            if (reload_path / 'f.model').exists() and (reload_path / 'vocab.json').exists():
                # could try to reload from this directory, at least necessary filenames exist
                self.reload_path = str(reload_path)
            elif (reload_path / 'best_model.path').exists():
                # maybe it is training project root so this file stores the name of the best directory so far
                self.reload_path = str(reload_path /
                                       Path((reload_path / 'best_model.path').read_text().strip()).parent.stem)
            else:
                raise ValueError(f'Invalid channel model reload path: {reload_path}')
            print(f'Will use this path to initialize channel training: {self.reload_path}')

            if self.path2candidate_scores is None and self.big_table is None:
                # for best path reporting, otherwise assert error
                self.path2candidate_scores = os.path.join(self.reload_path, 'dev_channel.json')
        else:
            self.reload_path = str(em_data_dir)

    def reset_from_path(self, fn: str) -> None:
        """
        Load scores from file and re-compute alphabets.
        :param fn: Path to file with scores.
        """
        print('Loading candidate scores from file: ', fn)
        with open(fn) as f:
            self.big_table = json.load(f)

        self.source_alphabet = set(c for candidate in self.big_table.values() for c in candidate)
        self.target_alphabet = set(c for word in self.big_table.keys() for c in word)

    def best_param_path(self, epoch_number: int, **kwargs) -> str:
        """
        Given an EM training epoch number, figure out the name of the file with channel scores for to this epoch.
        :param epoch_number: Epoch number.
        :param decode_threads: In case the file with channel scores does not exist but a channel model does (from an
            earlier experiment), rescore the candidates file with the channel with `decode_threads` number of threads.
        :return: The corresponding filename.
        """
        fn = os.path.join(self.em_data_dir, self.channel_data_template.format(epoch_number), 'dev_channel.json')
        if not os.path.exists(fn) and epoch_number == -1:
            if self.big_table is None:
                # scores from this `self.path2candidate_scores` has never been used.
                # @TODO: Check that its candidates match with self.generated_candidates_fn.
                decode_threads = kwargs.get("decode_threads", 4)
                print(f'... rescoring candidates with model from {self.reload_path}')
                channel_data_dir = os.path.join(self.em_data_dir, self.channel_data_template.format(epoch_number))
                os.makedirs(channel_data_dir)
                channel_train_data_fn = os.path.join(self.reload_path, "train.tsv")  # @TODO necessary at all?
                self._rescoring(channel_data_dir, self.reload_path, channel_train_data_fn, decode_threads)
            else:
                print(f'"{fn}" does not exists, falling back to "{fn}" ...')
            fn = self.path2candidate_scores
        assert os.path.exists(fn), f"best path does not exist: {fn}"
        return fn

    def bulkscore(self, cands: Iterable, word: Any) -> Generator[Tuple[Any, float], None, None]:

        if self.big_table is None:
            yield from self.backoff.bulkscore(cands, word)
        else:
            word_results = self.big_table.get(word)

            if word_results and word_results['candidates']:
                try:
                    candidates_scores = word_results['candidates']
                    if set(cands) != set(candidates_scores):
                        # print(cands, candidates_scores)
                        print('** Warning: Not getting exact same candidates from neural transducer: ', word)
                    # @TODO NASTY BUG: cands and candidates_scores need to be returned in exact same order:
                    for cand in cands:
                        cidx = word_results['candidates'].index(cand)
                        yield cand, word_results['log_prob'][cidx]
                    # yield from zip(word_results['candidates'], word_results['log_prob'])
                except Exception as e:
                    print('cands: ', cands)
                    print('candidates_scores: ', candidates_scores)
                    print('word: ', word)
                    print('big table: ', self.path2candidate_scores)
                    raise e
            else:
                print('** Warning: Word not found!', word, word_results)
                yield (word, 0.)

    def update_model(self, sources: Iterable, targets: Iterable, weights: Optional[Iterable[float]] = None,
                     epoch_number: int = 0, epochs: int = 5, patience: int = 100,
                     dev_path: Optional[str] = None, decode_threads: int = 1, reload: bool = False,
                     timeout: int = 12, dynet_seed: int =1, **kwargs) -> None:
        """
        Learns a new neural transducer model by maximizing weighted log-likelihood.
        :param sources: Source strings.
        :param targets: Target strings.
        :param weights: Weights for the pairs of source and target strings.
        :param epoch_number: For reporting, the number of the epoch.
        :param epochs: Number of epochs.
        :param patience: Patience.
        :param dev_path: Path to the development set. If None, use training set as dev set.
        :param decode_threads: How many processes to use for decoding.
        :param reload: Whether to start train the pretrained model from the previous epoch.
        :param timeout: Time for training: The neural model will not be trained more than the `timeout` hours.
        :param dynet_seed: Random seed, for model initialization.
        """
        channel_data_dir = os.path.join(self.em_data_dir,
                                        self.channel_data_template.format(epoch_number))
        channel_train_data_fn = os.path.join(channel_data_dir, 'train.tsv')

        os.makedirs(channel_data_dir)
        with open(channel_train_data_fn, 'w', encoding=ENCODING) as w:
            if weights is None:
                # @TODO Better: remove --sample-weights flag of the trainer shell script
                rows = ((s, t, NT_DEFAULT_FEATURE, 1.) for s, t in zip(sources, targets))
            else:
                rows = ((s, t, NT_DEFAULT_FEATURE, w) for s, t, w in zip(sources, targets, weights))
            csv.writer(w, delimiter='\t').writerows(rows)
        print(f'Wrote weighted hypotheses for M-step with neural transducer to "{channel_train_data_fn}".')

        if dev_path is None:
            # use train
            dev_path = os.path.join(channel_data_dir, 'dev.tsv')
            shutil.copyfile(channel_train_data_fn, dev_path)
        else:
            assert os.path.exists(dev_path), ('File does not exist: ', dev_path)

        # set some environment variables for training
        os.environ["PATIENCE"] = str(patience)
        os.environ["EPOCHS"] = str(epochs)
        os.environ["SEED"] = str(dynet_seed)
        # @TODO "il-k": if model performance is not improving, aggressive sampling from the model through low k
        # @TODO         (e.g. 12) and inverse sigmoid decay might not be a good choice. Increase k to e.g. 24.
        os.environ["ILK"] = str(kwargs.get('il_k', 12))
        # @TODO "batch-size": batch size 1 used to work well in morphology experiments ...
        os.environ["TRAIN_BATCH_SIZE"] = str(kwargs.get('batch_size', 20))
        # @TODO add more principled way of parsing kwargs parameters for neural channel
        os.environ["PICK_LOSS"] = '--pick-loss' if kwargs.get('pick_loss') else ''
        os.environ["TRAIN"] = channel_train_data_fn
        os.environ["DEV"] = str(dev_path)
        os.environ["RESULTS"] = channel_data_dir
        os.environ["LOGFILE"] = os.path.join(channel_data_dir, 'train.log')
        if reload:
            os.environ["TRAIN_RELOAD"] = self.reload_path  # start training with a pretrained model from this path
            if self.reload_path:
                print('Will continue training a pretrained model from: ', self.reload_path)
            else:
                print('Reload path does not exist. Will train a new model.')
        else:
            os.environ["TRAIN_RELOAD"] = str(self.em_data_dir)  # '' i.e. no model reload, only vocab reload
            print('Will train a new model.')

        # TRAINING
        print('Training the neural channel for {} epochs with patience {} and scoring candidates via {}.'.format(
            epochs, patience,self.PATH2NEURAL_MSTEP))
        start_train = time.time()
        return_code = subprocess.call(["timeout", "{}h".format(timeout), "bash", "-x", self.PATH2NEURAL_MSTEP])
        finish_train = (time.time() - start_train) / 60
        if return_code == 0:
            print('Finished training in {:.1f} min.'.format(finish_train))
            new_best = os.path.join(channel_data_dir, 'f.model')
            checksum = subprocess.check_output(["md5sum", new_best]).split()[0]
            print('checksums: prior={}, now={}'.format(self.checksum, checksum))
            if self.checksum is not None and self.checksum == checksum:
                print('Training resulted in the same model. Restarting from uninformed parameters...')
                self.checksum = None
                reload = False  # do not reload from self.reload_path, previous epoch's channel directory
                shutil.move(channel_data_dir, channel_data_dir + '_restarted')
                # restart
                self.update_model(sources, targets, weights, epoch_number, epochs, patience,
                                  dev_path, decode_threads, reload, timeout, dynet_seed, **kwargs)
            else:
                self.checksum = checksum
                # this has been training from scratch or improvement on reloaded model
                print('Decoding...')
        else:
            print('\n\n*** Training terminated with error or by user. Trained for {:.1f} min. ***.'
                  '\n\nDecoding...'.format(finish_train))

        # DECODING
        self._rescoring(channel_data_dir, channel_data_dir, channel_train_data_fn, decode_threads)

    def _rescoring(self, channel_data_dir: str, reload_path: str, channel_train_data_fn: str, decode_threads: int):
        assert os.path.exists(self.generated_candidates_fn)
        self.path2candidate_scores = os.path.join(channel_data_dir, 'dev_channel.json')
        os.environ["LOGFILE"] = os.path.join(channel_data_dir, 'decoding.log')
        os.environ["TRAIN"] = channel_train_data_fn
        os.environ["RELOAD"] = reload_path

        if decode_threads <= 1:
            os.environ["RESULTS"] = channel_data_dir
            return_code = subprocess.call(["bash", "-x", self.PATH2DECODER])
            assert return_code == 0
        else:
            try:
                scheduler_queue(lockfile_name=os.environ.get("CNNC_MULTIPROCESSING_RUNNING",
                                                             "CNNC_MULTIPROCESSING_RUNNING"))
                print('Decoding using {} threads'.format(decode_threads))
                # the `self.generated_candidates_fn` TSV file needs to be chopped into `decode_threads` pieces.
                # A decoder is launched for each piece. The result is merged back as
                # `os.path.join(channel_data_dir, 'dev_channel.json')`.

                # count the total number of candidates to score
                # and split the candidate file into as many parts as there are threads.
                # @TODO could also be inferred from the number of types add candidates per type
                tmp_dir = os.path.join('/tmp', uuid.uuid4().hex)
                os.makedirs(tmp_dir)
                print('Temporary decoding directory: {}'.format(tmp_dir))
                os.environ["RESULTS"] = tmp_dir
                line_count = 0
                with open(self.generated_candidates_fn, encoding=ENCODING) as f:
                    for _ in f:
                        line_count += 1
                    f.seek(0)
                    step_size = math.ceil(line_count / decode_threads)
                    print('Will split "{}" ({} lines) into {} chunks of size {}'.format(
                        self.generated_candidates_fn, line_count, decode_threads, step_size))
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
                # ... same as shell script:
                # LINE_COUNT=$(cat $CANDIDATES_TSV | wc -l)
                # CHUNK_SIZE=$(expr $LINE_COUNT / $DECODE_THREAD)
                # TEMPDIR=$(mktemp -d /tmp/dev_channel.XXXXXXXX)
                # split $CANDIDATES_TSV $TEMPDIR/dev_channel -d --lines=$CHUNK_SIZE --additional-suffix=.tsv --suffix-length=3
                assert len(tmp_candidate_fns) == decode_threads, (tmp_candidate_fns, '!=', decode_threads)
                os.environ["DECODE_THREAD"] = str(decode_threads)
                results = self._launch_decoder(tmp_candidate_fns)
                # repackage predictions into one dictionary
                big_table = dict()
                for return_code, (tmp_candidate_fn, suffix) in zip(results, tmp_candidate_fns):
                    assert return_code == 0, 'Decoding failed: {}'.format(tmp_candidate_fn)
                    json_output = os.path.join(tmp_dir, 'dev_channel{}.json'.format(suffix))
                    with open(json_output, encoding=ENCODING) as w:
                        table = json.load(w)
                    for key in table:
                        if key in big_table:
                            for v in ('candidates', 'log_prob', 'acts', 'feats'):
                                big_table[key][v].extend(table[key][v])
                        else:
                            big_table[key] = table[key]
                # write the dictionary to file
                with open(self.path2candidate_scores, mode='w', encoding=ENCODING) as w:
                    json.dump(big_table, w, indent=1, ensure_ascii=False)
                # cleanup temporary files
                print(f'Cleaning up temporary directory {tmp_dir}...')
                shutil.rmtree(tmp_dir)
            finally:
                scheduler_dequeue(lockfile_name=os.environ.get("CNNC_MULTIPROCESSING_RUNNING",
                                                               "CNNC_MULTIPROCESSING_RUNNING"))
        print('Finished rescoring.')
        # reset scores
        self.reset_from_path(self.path2candidate_scores)
        self.reload_path = channel_data_dir  # this epoch's results dir

    def save_model(self, path2model: str) -> None:
        """
        :param path2model: Path where to write the model.
        """
        shutil.copyfile(self.path2candidate_scores, path2model)

    def _launch_decoder(self, tmp_candidate_fns: List[Tuple[str, str]]) -> List[int]:
        """
        Launch decoder shell processes in parallel.
        :param tmp_candidate_fns: List of candidate tsv filenames and corresponding decoder batch suffixes.
        :return: List of return codes.
        """
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
