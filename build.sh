#!/bin/bash
wc -l test.c backprop.c backprop.h vecmath.c vecmath.h
gcc test.c backprop.c backprop.h vecmath.c vecmath.h -lm
