#!/bin/bash

doxygen Doxyfile
cd out_ddi_doxygen/latex
make
