// SPDX-License-Identifier: GPL-2.0
// Copyright(c) 2017-18 Intel Corporation

#ifndef __GNA_DRV_H__
#define __GNA_DRV_H__

#include <linux/cdev.h>
#include <linux/fs.h>
#include <linux/pci.h>

#include <uapi/misc/gna.h>

#include "gna_hw.h"
#include "gna_mem.h"
#include "gna_score.h"

#define GNA_DRV_NAME "gna"
#define GNA_DRV_VER "1.0"

#define MAX_GNA_DEVICES 16

extern struct class *gna_class;

extern const struct dev_pm_ops gna_pm;

struct gna_driver_private {

	/* device major/minor number facitlities */
	DECLARE_BITMAP(dev_map, MAX_GNA_DEVICES);
	dev_t devt;
	int minor;

	/* protects this structure */
	struct mutex			lock;
};

struct pci_bar {
	resource_size_t		iostart;
	resource_size_t		iosize;
	void __iomem		*mem_addr;
};

struct gna_drv_info {
	const enum gna_device_t		id;
	const u32			max_hw_mem;
	const u32			num_pagedir;
	const u32			num_pagetab;
	const u32			max_layer_count;
	const struct gna_desc_info	desc_info;
};

struct gna_private {

	struct gna_driver_private 	*drv_priv;

	// character device info
	char				name[8];
	int				dev_num;

	/* lock protecting this very structure */
	struct mutex			lock;

	/* spinlock used in interrupt context */
	spinlock_t			irq_lock;

	/* device objects */
	struct pci_dev			*pdev;
	struct device			*parent; /* pdev->dev */
	struct device			dev;
	struct cdev			cdev;

	/* device related resources */
	struct pci_bar			bar0;
	struct gna_drv_info		info;
	unsigned int			irq;

	/* device busy indicator */
	bool				busy;

	/* device functions */
	/* should be called with acquired mutex */
	struct gna_device_operations	*ops;

	/* score related fields */
	atomic_t			request_count;
	struct gna_request		*current_request;
	u64				current_request_id;

	/* score request workqueue */
	struct workqueue_struct		*callback_wq;
	struct work_struct		score_work;

	/* bottom half facilities */
	atomic_t			isr_count;
	struct tasklet_struct 		request_tasklet;
	void				(*request_tasklet_fn)(unsigned long);

	/* interrupt timer */
	struct timer_list		isr_timer;

	/* list of reqs to be processed */
	struct spinlock			reqlist_lock;
	struct list_head		request_list;
};

extern struct gna_driver_private gna_drv_priv;

extern int gna_probe(struct pci_dev *pcidev, const struct pci_device_id *pci_id);

extern void gna_remove(struct pci_dev *pci);

#endif /* __GNA_DRV_H__ */
