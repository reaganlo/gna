#!/bin/bash

doxygen Doxyfile
cd out_tlv_doxygen/latex
make
