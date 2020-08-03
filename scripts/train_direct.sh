source "$PATH2NEURAL_CODE"/venv*/bin/activate

echo "*** TRAINING TRANSDUCER ***" 
echo "ENVIRONMENT"
env

/usr/bin/time -v python3 -u "$PATH2NEURAL_CODE"/run_scripts/run_transducer.py \
	--dynet-seed $SEED --dynet-mem 1000 --dynet-autobatch 0 --transducer=haem --sigm2017format --sample-weights \
	--input=100 --feat-input=20 --action-input=100 --enc-hidden=200 --dec-hidden=200 --enc-layers=1 --dec-layers=1 \
	--mlp=0 --nonlin=ReLU --compact-feat=0 --compact-nonlin=linear --tag-wraps=both --param-tying \
	--il-optimal-oracle --il-loss=nll --il-k=$ILK --il-beta=0.5 --il-global-rollout --verbose=0 --dev-subsample=0 \
	--dropout=0 --optimization=ADADELTA --l2=0 --batch-size=$TRAIN_BATCH_SIZE --patience=$PATIENCE --epochs=$EPOCHS \
	--decbatch-size=25 --reload-path=$TRAIN_RELOAD --mode=il --beam-width=0 --beam-widths=4 $PICK_LOSS \
	$TRAIN  $DEV  $RESULTS 2>&1 | tee $LOGFILE
