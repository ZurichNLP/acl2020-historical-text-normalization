from contextualized_transduction.direct_candidates import DirectCandidates

from sacred import Experiment
from sacred import SETTINGS

SETTINGS.CONFIG.READ_ONLY_CONFIG = False

ex = Experiment()


@ex.automain
def main(config):
    direct_candidates_generator = DirectCandidates.from_config(config)
    direct_candidates_generator.train_decode()
