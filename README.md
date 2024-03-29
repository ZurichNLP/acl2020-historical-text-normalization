# acl2020-historical-text-normalization
System predictions and code for the [ACL 2020 paper "Semi-supervised Contextual Historical Text Normalization" by Peter Makarov and Simon Clematide](https://www.aclweb.org/anthology/2020.acl-main.650).

## Contextualized transduction

This code performs the expectation--maximization training and decoding. It is somewhat agnostic of the channel model, with which it communicates via system calls (`scripts/*sh`).

The kenlm-based decoders use kenlm's Python API and so are not that performant despite vectorized code.

Clone and install the code:

```
cd acl2020-historical-text-normalization
python3 -m venv venv
source venv/bin/activate
pip install .
python contextualized_transduction/tests.py  # works?
```

Additionally, you will need to install the neural transducer (anywhere on your system):

```
git clone https://github.com/peter-makarov/il-reimplementation.git
cd il-reimplementation
git checkout feature/acl2020-htn  # this branch includes changes relevant for this paper
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

(For more details on the installation of the neural transducer, please see https://github.com/peter-makarov/il-reimplementation/tree/feature/acl2020-htn)

## Running the code

An experiment is specified in a json file (e.g. `scripts/example_configs/trainer.de.500-1000.config.json`). Once you have configured the json, run the experiment with

```
python scripts/run_trainer.py with scripts/example_configs/trainer.de.500-1000.config.json
```

## System predictions
All our system predictions referred to in the paper can be found in the subdirectory `paper_results`. 

Corpora acronyms:

 - de-rid German Ridges
 - en-ica English ICAMET (not available due to licencing restrictions)
 - es-psa Spanish 
 - hu-hgd Hungarian
 - is-ice Icelandic
 - pt-psa Portuguese
 - sl-boh Slovene
 - sv-gaw Swedish 
