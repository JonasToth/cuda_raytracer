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

OBJ_BASE="$2"

OUT_BASE="$3"
OUT_FILE="$3.png"

REF_BASE="$4"
REF_FILE="$4.png"


eval "$TEST_EXE $OBJ_BASE $OUT_BASE"

convert "./int_test_output/$OUT_FILE" "./int_test_output/$OUT_BASE.rgba"
convert "./int_test_ref/$REF_FILE" "./int_test_output/$REF_BASE.rgba"
cmp "./int_test_output/$OUT_BASE.rgba" "./int_test_output/$REF_BASE.rgba"

diff=$?

rm "./int_test_output/$OUT_BASE.rgba" "./int_test_output/$REF_BASE.rgba"

if [ $diff -eq 0 ]; then
    echo "Equality"
    exit 0
else
    echo "Images differ!"
    exit 1
fi
