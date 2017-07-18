#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import Image
import math
import sys


## http://snipplr.com/view/757/compare-two-pil-images-in-python/
def test_equality(path1, path2, threshold):
    h1 = Image.open(path1).histogram()
    h2 = Image.open(path2).histogram()

    rms = math.sqrt(reduce(lambda x,y: x + y, map(lambda a,b: (a-b) ** 2, h1, h2)) / len(h1))

    if rms < threshold:
        return (True, rms)
    else:
        return (False, rms)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Compare two images by histogram')
    parser.add_argument('tested_image', type=str, 
                        help = 'Path of image that shall be tested for its validity')
    parser.add_argument('reference_image', type=str, 
                        help = 'Path of the reference image/expected correct output')
    parser.add_argument('--threshold', type=float, required = False, default = 10.,
                        help = 'Threshold for histogram difference, default = 10.')

    args = parser.parse_args()
    (equals, diff) = test_equality(args.tested_image, args.reference_image, args.threshold)

    if equals:
        print "Images are considered equal, histogramm difference = ", diff
        sys.exit(0)
    else:
        print "Images are considered inequal, histogramm difference = ", diff
        sys.exit(1)

