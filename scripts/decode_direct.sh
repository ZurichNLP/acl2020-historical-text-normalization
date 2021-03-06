source "$PATH2NEURAL_CODE"/venv*/bin/activate

echo "*** BEAM DECODING DIRECT MODEL ***"
/usr/bin/time -v python -u "$PATH2NEURAL_CODE"/run_scripts/decoders.py --dynet-seed 1 --dynet-mem 1000 \
--dynet-autobatch 0  --transducer=haem --sigm2017format \
--input=100 --feat-input=20 --action-input=100 --enc-hidden=200 --dec-hidden=200 --enc-layers=1 --dec-layers=1 \
--mlp=0 --nonlin=ReLU --compact-feat=0 --compact-nonlin=linear --tag-wraps=both --param-tying \
--decoding-mode=$MODE --beam-width=$BEAM_WIDTH --result-fn-suffix=$RESULT_SUFFIX --reload-path=$RELOAD \
$TRAIN  $TYPES2DECODE_TSV  $RESULTS 2>&1 | tee -a $LOGFILE
