#!/bin/bash

# run a test file and compare the result against the expected result
if [ $# -ne 4  ]; then
    echo "Incorrect argument count!"
    echo
    echo "Usage: $0 test_executable.x obj_file output_image reference_image"
    echo
    exit 1
fi

TEST_EXE="$1"

OBJ_FILE="$2"
OUT_FILE="$3"
REF_FILE="$4"


eval "OMP_NUM_THREADS=4 $TEST_EXE $OBJ_FILE $OUT_FILE"

python compare_images.py "$OUT_FILE" "$REF_FILE"

diff=$?

if [ $diff -eq 0 ]; then
    echo "Equality"
    exit 0
else
    echo "Images differ!"
    exit 1
fi
