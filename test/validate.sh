#!/bin/bash

# run a test file and compare the result against the expected result
if [ $# -ne 3  ]; then
    echo "Incorrect argument count!"
    echo
    echo "Usage: $0 test_executable.x output_image.png reference_image.rgba"
    echo
    exit 1
fi

TEST_EXE="$1"

OUT_FILE="$2"
OUT_BASE="${OUT_FILE%.*}"

REF_FILE="$3"
REF_BASE="${REF_FILE%.*}"


eval "$TEST_EXE $OUT_FILE"

convert "$OUT_FILE" "$OUT_BASE.rgba"
convert "$REF_FILE" "$REF_BASE.rgba"
cmp "$OUT_BASE.rgba" "$REF_BASE.rgba"

diff=$?

rm "$OUT_BASE.rgba" "$REF_BASE.rgba"

if [ $diff -eq 0 ]; then
    echo "Equality"
    exit 0
else
    echo "Images differ!"
    exit 1
fi
