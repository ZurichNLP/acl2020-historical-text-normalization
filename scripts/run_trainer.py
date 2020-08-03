from contextualized_transduction.trainer import Trainer

import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s: %(message)s')

from sacred import Experiment
from sacred import SETTINGS

SETTINGS.CONFIG.READ_ONLY_CONFIG = False

ex = Experiment()


@ex.automain
def main(config):
    trainer = Trainer(config)
    trainer.train()
