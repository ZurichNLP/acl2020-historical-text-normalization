from contextualized_transduction.sequence_reranker import SequenceReranker

from sacred import Experiment
from sacred import SETTINGS

SETTINGS.CONFIG.READ_ONLY_CONFIG = False

ex = Experiment()


@ex.automain
def main(config):
    sequence_reranker = SequenceReranker(config)
    sequence_reranker.rerank()
