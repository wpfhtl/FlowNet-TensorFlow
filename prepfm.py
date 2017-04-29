# -*- coding:utf8 -*-
"""
Preprocess pfm files
"""

import os
import re
import sys
import subprocess

import numpy as np

import pfmkit

output_dir = '..\FlowNet-Data\Driving\\bin'
assert(os.path.isdir(output_dir))
dispnoc = []
base1 = '..\FlowNet-Data\Driving\disparity'

print('load 04')
for i in range(400, 402):
    disp, scale = pfmkit.load_pfm(os.path.join(base1, '{}.pfm'.format(i)), True)
    dispnoc.append(disp.astype(np.float32))

for i in range(len(dispnoc)):
    pfmkit.tofile('{}/dispnoc_{:04d}.bin'.format(output_dir, i + 1), dispnoc[i])