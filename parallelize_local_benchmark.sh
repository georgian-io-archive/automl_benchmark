#!/bin/sh

TRAINING_TIME=$1
FORESHADOW=$2
EXP_NOTES=$3

if [ -z "$TRAINING_TIME" ]; then
    echo "Training time (mins) input required."
    exit 1
fi

if [ -z "$FORESHADOW" ]; then
    echo "Running original AutoML benchmarks..."
elif [ "$FORESHADOW" = "--foreshadow" ]; then
    echo "Running foreshadow benchmarks..."
fi

PROCESSES_POSSIBLE=$(($(nproc) - 2))
PROCESS_LIMIT=15
PROCESSES=$((PROCESSES_POSSIBLE > PROCESS_LIMIT ? PROCESS_LIMIT : PROCESSES_POSSIBLE))

python generate_params.py $FORESHADOW | xargs -L 1 -P $PROCESSES python ./main.py local-benchmark -t=$TRAINING_TIME -n=$EXP_NOTES
