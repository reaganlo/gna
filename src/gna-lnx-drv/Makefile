# SPDX-License-Identifier: GPL-2.0
#
# Makefile - Intel GMM & Neural Network Accelerator (Intel GNA) Linux driver
# Copyright (c) 2018, Intel Corporation.
#

ccflags-y += -Werror
ccflags-$(CONFIG_INTEL_GNA_DEBUG) += -DDEBUG
gna-y := gna_drv.o \
         gna_pci.o \
         gna_ioctl.o \
         gna_hw.o \
         gna_irq.o \
         gna_mem.o \
         gna_score.o

obj-$(CONFIG_INTEL_GNA) += gna.o
