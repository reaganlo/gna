// SPDX-License-Identifier: GPL-2.0
// Copyright(c) 2017-18 Intel Corporation

/*
 *  gna_drv.c - GNA Driver
 */

#include <uapi/misc/gna.h>

#include <linux/cdev.h>
#include <linux/module.h>
#include <linux/mm.h>
#include <linux/sched.h>
#include <linux/init.h>
#include <linux/fs.h>
#include <linux/err.h>
#include <linux/types.h>
#include <linux/pci.h>
#include <linux/pagemap.h>
#include <linux/pm_runtime.h>

#include "gna_drv.h"

#include "gna_irq.h"
#include "gna_ioctl.h"

struct class *gna_class;

struct gna_driver_private gna_drv_priv;

static bool msi_enable = true;
module_param(msi_enable, bool, S_IRUGO);
MODULE_PARM_DESC(msi_enable, "Enable MSI interrupts");

/* recovery timeout in ms */
static unsigned int recovery_timeout = 60;
module_param(recovery_timeout, uint, S_IRUGO|S_IWUSR);
MODULE_PARM_DESC(recovery_timeout, "Recovery timeout");

static int gna_suspend(struct device *dev)
{
	dev_dbg(dev, "%s\n", __func__);
	return 0;
}

static int gna_resume(struct device *dev)
{
	dev_dbg(dev, "%s\n", __func__);
	return 0;
}

static int gna_runtime_suspend(struct device *dev)
{
	struct gna_private *gna_priv = dev_get_drvdata(dev);
	void __iomem *addr = gna_priv->bar0.mem_addr;
	u32 val;
	int i = 0;

	dev_dbg(dev, "%s: PM registers\n", __func__);
	dev_dbg(dev, "ISI  %.8x\n", gna_reg_read(addr, GNAISI));
	dev_dbg(dev, "PISV %.8x\n\n", gna_reg_read(addr, GNAPISV));
	dev_dbg(dev, "PM_CS %.8x\n", gna_reg_read(addr, GNA_PCI_PMCS));
	dev_dbg(dev, "PWRCTRL %.8x\n", gna_reg_read(addr, GNAPWRCTRL));

	val = gna_reg_read(addr, GNAD0I3C);
	dev_dbg(dev, "D0I3 %.8x\n", gna_reg_read(addr, GNAD0I3C));

	/* Verify command in progress bit */
	while (i < 100) {
		val = gna_reg_read(addr, GNAD0I3C);
		if ((val & 0x1) == 0)
			break;
		i++;
	}

	if (i == 100) {
		dev_err(dev, "command in progress - try again\n");
		return -EAGAIN;
	}

	/* abort any operations if running */
	gna_abort_hw(gna_priv, addr);

	/* put device in D0i3 */
	val = GNA_D0I3_POWER_OFF;
	gna_reg_write(addr, GNAD0I3C, val);

	dev_dbg(dev, "%s: exit: D0I3 %.8x val 0x%x\n", __func__, gna_reg_read(addr, GNAD0I3C), val);

	return 0;
}

static int gna_runtime_resume(struct device *dev)
{
	struct gna_private *gna_priv = dev_get_drvdata(dev);
	void __iomem *addr = gna_priv->bar0.mem_addr;
	u32 val;

	dev_dbg(dev, "%s:\n", __func__);

	val = gna_reg_read(addr, GNAD0I3C);
	dev_dbg(dev, "D0I3 %.8x\n", gna_reg_read(addr, GNAD0I3C));

	/* put device in active D0 state */
	val = GNA_D0I3_POWER_ON;
	gna_reg_write(addr, GNAD0I3C, val);

	dev_dbg(dev, "D0I3 %.8x val 0x%x\n", gna_reg_read(addr, GNAD0I3C), val);
	return 0;
}

const struct dev_pm_ops gna_pm = {
	.suspend = gna_suspend,
	.resume = gna_resume,
	.runtime_suspend = gna_runtime_suspend,
	.runtime_resume = gna_runtime_resume,
};

static inline struct gna_private *inode_to_gna(struct inode *inode)
{
	return container_of(inode->i_cdev, struct gna_private, cdev);
}

static int gna_open(struct inode *inode, struct file *f)
{
	struct gna_private *gna_priv;
	int ret;
	int id;

	id = iminor(inode);

	gna_priv = inode_to_gna(inode);
	if (!gna_priv)
		return -ENODEV;

	dev_dbg(&gna_priv->dev, "%s: enter id=%d\n", __func__, id);

	mutex_lock(&gna_priv->lock);
	ret = gna_priv->ops->open(gna_priv, f);
	mutex_unlock(&gna_priv->lock);

	dev_dbg(&gna_priv->dev, "%s: exit\n", __func__);

	return ret;
}

static int gna_release(struct inode *inode, struct file *f)
{
	struct gna_private *gna_priv;

	gna_priv = inode_to_gna(inode);
	if (!gna_priv)
		return -ENODEV;

	dev_dbg(&gna_priv->dev, "%s: enter\n", __func__);

	mutex_lock(&gna_priv->lock);
	gna_priv->ops->free(gna_priv, f);
	mutex_unlock(&gna_priv->lock);

	dev_dbg(&gna_priv->dev, "%s: exit\n", __func__);
	return 0;
}


static const struct file_operations gna_file_ops = {
	.owner		=	THIS_MODULE,
	.open		=	gna_open,
	.release	=	gna_release,
	.unlocked_ioctl =	gna_ioctl,
};

static int gna_priv_open(struct gna_private *gna_priv, struct file *fd)
{
	struct gna_file_private *file_priv;

	dev_dbg(&gna_priv->dev, "%s: enter\n", __func__);

	file_priv = kzalloc(sizeof(*file_priv), GFP_KERNEL);
	if (!file_priv)
		return -ENOMEM;

	file_priv->fd = fd;
	file_priv->gna_priv = gna_priv;

	idr_init(&file_priv->memory_idr);
	mutex_init(&file_priv->lock);

	fd->private_data = file_priv;

	dev_dbg(&gna_priv->dev, "%s: exit\n", __func__);

	return 0;
}

static void gna_priv_free(struct gna_private *gna_priv, struct file *fd)
{
	struct gna_file_private *file_priv;
	struct gna_file_private *request_file_priv;
	struct gna_request *score_request;
	unsigned long irq_flags;

	dev_dbg(&gna_priv->dev, "%s\n", __func__);

	file_priv = (struct gna_file_private*) fd->private_data;

	// if current request belongs to the file, wait for it
	// at this point we don't care about request result
	spin_lock_irqsave(&gna_priv->irq_lock, irq_flags);
	score_request = gna_priv->current_request;
	if (score_request != NULL) {
		request_file_priv = gna_priv->current_request->file_priv;
		if (request_file_priv->fd == fd)
			gna_score_wait(score_request, recovery_timeout);
	}
	spin_unlock_irqrestore(&gna_priv->irq_lock, irq_flags);

	// delete all requests that belong to the file
	spin_lock(&gna_priv->reqlist_lock);
	gna_delete_file_requests(fd, &gna_priv->request_list);
	spin_unlock(&gna_priv->reqlist_lock);

	/* free is called when last entity closes the file */
	idr_for_each(&file_priv->memory_idr, gna_priv->ops->unmap, file_priv);
	idr_destroy(&file_priv->memory_idr);

	fd->private_data = NULL;
	kfree(file_priv);
}

static int gna_priv_getcaps(struct gna_private *gna_priv, struct gna_capabilities *caps)
{
	int ret;
	dev_dbg(&gna_priv->dev, "%s: enter\n", __func__);

	caps->device_type = gna_priv->info.id;
	caps->recovery_timeout = recovery_timeout;

	ret = pm_runtime_get_sync(&gna_priv->pdev->dev);
	if (ret < 0) {
		dev_warn_once(&gna_priv->dev,
			"pm_runtime_get_sync() failed: %d\n", ret);
		return ret;
	}

	caps->in_buff_size = gna_reg_read(gna_priv->bar0.mem_addr, GNAIBUFFS);

	ret = pm_runtime_put_sync(&gna_priv->pdev->dev);
	if (ret < 0) {
		dev_warn_once(&gna_priv->dev,
			"pm_runtime_put_sync() failed: %d\n", ret);
	}

	dev_dbg(&gna_priv->dev, "in_buff_size %#x device_type %#x\n", caps->in_buff_size,
		 caps->device_type);

	return 0;
}

static struct gna_device_operations gna_drv_ops = {
	.owner		=	THIS_MODULE,
	.getcaps	=	gna_priv_getcaps,
	.open		=	gna_priv_open,
	.free		=	gna_priv_free,
	.score		=	gna_priv_score,
	.map		=	gna_priv_map,
	.unmap		=	gna_priv_unmap,
};

void gna_dev_release(struct device *dev)
{
	struct gna_private *gna_priv;

	dev_dbg(dev, "%s enter\n", __func__);

	gna_priv = dev_get_drvdata(dev);

	__clear_bit(MINOR(dev->devt), gna_drv_priv.dev_map);
	destroy_workqueue(gna_priv->callback_wq);
	kfree(gna_priv);

	dev_set_drvdata(dev, NULL);
	pci_set_drvdata(gna_priv->pdev, NULL);

	dev_dbg(dev, "%s exit\n", __func__);
}

static int gna_dev_create(struct gna_private *gna_priv)
{
	struct pci_dev *pcidev;
	struct device *dev;
	dev_t gna_devt;
	int dev_num;
	int major;
	int minor;
	int ret;

	pcidev = gna_priv->pdev;

	mutex_lock(&gna_drv_priv.lock);

	dev_num = find_first_zero_bit(gna_drv_priv.dev_map, MAX_GNA_DEVICES);
	if (dev_num == MAX_GNA_DEVICES) {
		dev_err(&pcidev->dev, "number of gna devices reached maximum\n");
		ret = -ENODEV;
		goto err_unlock_drv;
	}

	set_bit(dev_num, gna_drv_priv.dev_map);
	major = MAJOR(gna_drv_priv.devt);
	minor = gna_drv_priv.minor++;

	mutex_unlock(&gna_drv_priv.lock);

	gna_devt = MKDEV(major, minor);
	dev = &gna_priv->dev;
	device_initialize(dev);
	dev->devt = gna_devt;
	dev->class = gna_class;
	dev->parent = gna_priv->parent;
	dev->groups = NULL;
	dev->release = gna_dev_release;
	dev_set_drvdata(dev, gna_priv);
	dev_set_name(dev, "gna%d", dev_num);

	snprintf(gna_priv->name, sizeof(gna_priv->name), "gna%d", dev_num);
	gna_priv->dev_num = dev_num;
	gna_priv->ops = &gna_drv_ops;

	cdev_init(&gna_priv->cdev, &gna_file_ops);
	gna_priv->cdev.owner = THIS_MODULE;

	ret = cdev_device_add(&gna_priv->cdev, &gna_priv->dev);
	if (ret) {
		dev_err(&gna_priv->dev, "could not add gna%d char device\n", dev_num);
		goto err_release_devnum;
	}

	dev_info(&gna_priv->dev, "registered gna%d device: major %d, minor %d\n",
						dev_num, major, minor);

	return 0;

err_release_devnum:
	mutex_lock(&gna_drv_priv.lock);
	__clear_bit(minor, gna_drv_priv.dev_map);

err_unlock_drv:
	mutex_unlock(&gna_drv_priv.lock);

	return ret;
}

static int gna_dev_init(struct gna_private *gna_priv, struct pci_dev *pcidev,
		const struct pci_device_id *pci_id)
{
	struct gna_drv_info *gna_info;
	int ret;

	dev_dbg(&pcidev->dev, "%s: enter\n", __func__);

	pci_set_drvdata(pcidev, gna_priv);

	gna_priv->irq = pcidev->irq;
	gna_priv->parent = &pcidev->dev;
	gna_priv->pdev = pci_dev_get(pcidev);

	gna_info = (struct gna_drv_info *)pci_id->driver_data;
	memcpy(&gna_priv->info, gna_info, sizeof(*gna_info));
	dev_dbg(&pcidev->dev, "hw mem %d num pd %d\n",
			gna_info->max_hw_mem, gna_info->num_pagedir);
	dev_dbg(&pcidev->dev, "desc gna_info %d mmu gna_info %d\n",
			gna_info->desc_info.rsvd_size,
			gna_info->desc_info.mmu_info.vamax_size);

	mutex_init(&gna_priv->lock);

	spin_lock_init(&gna_priv->reqlist_lock);
	spin_lock_init(&gna_priv->irq_lock);

	gna_priv->drv_priv = &gna_drv_priv;

	INIT_LIST_HEAD(&gna_priv->request_list);

	INIT_WORK(&gna_priv->score_work, gna_work);

	timer_setup(&gna_priv->isr_timer, gna_isr_timeout, 0);
	gna_priv->isr_timer.expires = jiffies + msecs_to_jiffies(recovery_timeout);

	atomic_set(&gna_priv->request_count, 0);
	atomic_set(&gna_priv->isr_count, 0);

	gna_priv->request_tasklet_fn = gna_request_tasklet;
	tasklet_init(&gna_priv->request_tasklet, gna_priv->request_tasklet_fn,
			(unsigned long) gna_priv);

	gna_priv->callback_wq = create_singlethread_workqueue("gna_callback_wq");
	if (IS_ERR(gna_priv->callback_wq)) {
		dev_err(&pcidev->dev, "could not create workqueue for gna device\n");
		ret = PTR_ERR(gna_priv->callback_wq);
		goto err_pci_put;
	}

	gna_priv->busy = false;

	ret = gna_dev_create(gna_priv);
	if (ret) {
		dev_err(&pcidev->dev, "could not create gna device\n");
		goto err_del_wq;
	}

	dev_dbg(&pcidev->dev, "%s: exit\n", __func__);

	return 0;

err_del_wq:
	destroy_workqueue(gna_priv->callback_wq);

err_pci_put:
	pci_dev_put(pcidev);
	pci_set_drvdata(pcidev, NULL);

	return ret;
}

int gna_probe(struct pci_dev *pcidev, const struct pci_device_id *pci_id)
{
	struct gna_private *gna_priv;
	int ret;

	dev_dbg(&pcidev->dev, "%s: enter\n", __func__);

	ret = pci_enable_device(pcidev);
	if (ret) {
		dev_err(&pcidev->dev, "pci device can't be enabled\n");
		goto end;
	}

	ret = pci_request_regions(pcidev, GNA_DRV_NAME);
	if (ret)
		goto err_disable_device;

	ret = pci_set_dma_mask(pcidev, DMA_BIT_MASK(64));
	if (ret) {
		dev_err(&pcidev->dev, "pci_set_dma_mask returned error %d\n", ret);
		goto err_release_regions;
	}

	pci_set_master(pcidev);

	/* register for interrupts */
	if (msi_enable) {
		ret = pci_enable_msi(pcidev);
		if (ret) {
			dev_err(&pcidev->dev, "could not enable msi interrupts\n");
			goto err_clear_master;
		}
		dev_info(&pcidev->dev, "msi interrupts enabled\n");
	}

	/* init gna device */
	gna_priv = kzalloc(sizeof(*gna_priv), GFP_KERNEL | GFP_ATOMIC);
	if (!gna_priv) {
		ret = PTR_ERR(gna_priv);
		dev_err(&pcidev->dev, "could not allocate gna private structure\n");
		goto err_disable_msi;
	}

	ret = request_threaded_irq(pcidev->irq, gna_interrupt,
			gna_irq_thread, IRQF_SHARED,
			GNA_DRV_NAME, gna_priv);

	if (ret) {
		dev_err(&pcidev->dev, "could not register for interrupt\n");
		goto err_free_priv;
	}

	dev_dbg(&pcidev->dev, "irq num %d\n", pcidev->irq);

	/* Map BAR0 */
	gna_priv->bar0.iostart = pci_resource_start(pcidev, 0);
	gna_priv->bar0.iosize = pci_resource_len(pcidev, 0);
	gna_priv->bar0.mem_addr = pci_iomap(pcidev, 0, 0);

	dev_dbg(&pcidev->dev, "bar0 io start: %p\n", (void *)gna_priv->bar0.iostart);
	dev_dbg(&pcidev->dev, "bar0 io size: %llu\n", gna_priv->bar0.iosize);
	dev_dbg(&pcidev->dev, "bar0 memory address: %p\n", (void *)gna_priv->bar0.mem_addr);

	ret = gna_dev_init(gna_priv, pcidev, pci_id);
	if (ret) {
		dev_err(&pcidev->dev, "could not initialize gna private structure\n");
		goto err_free_irq;
	}

	/* enable power management callbacks */
	pm_runtime_set_autosuspend_delay(&pcidev->dev, 2000);
	pm_runtime_use_autosuspend(&pcidev->dev);
	pm_runtime_allow(&pcidev->dev);
	pm_runtime_put_noidle(&pcidev->dev);

	dev_dbg(&pcidev->dev, "%s exit: %d\n", __func__, ret);

	return 0;

err_free_irq:
	pci_iounmap(pcidev, gna_priv->bar0.mem_addr);
	free_irq(pcidev->irq, gna_priv);
err_free_priv:
	kfree(gna_priv);
err_disable_msi:
	if (msi_enable)
		pci_disable_msi(pcidev);
err_clear_master:
	pci_clear_master(pcidev);
err_release_regions:
	pci_release_regions(pcidev);
err_disable_device:
	pci_disable_device(pcidev);
end:
	dev_err(&pcidev->dev, "gna probe failed with %d\n", ret);
	return ret;
}

void gna_remove(struct pci_dev *pcidev)
{
	struct gna_private *gna_priv;

	gna_priv = pci_get_drvdata(pcidev);
	if (IS_ERR(gna_priv)) {
		dev_err(&pcidev->dev, "could not get driver data from pci device\n");
		return;
	}

	dev_dbg(&gna_priv->dev, "%s: enter\n", __func__);

	cdev_device_del(&gna_priv->cdev, &gna_priv->dev);

	free_irq(pcidev->irq, gna_priv);

	if (msi_enable)
		pci_disable_msi(pcidev);

	pci_clear_master(pcidev);
	pci_iounmap(pcidev, gna_priv->bar0.mem_addr);
	pci_release_regions(pcidev);
	pci_disable_device(pcidev);
	pci_dev_put(pcidev);

	dev_dbg(&gna_priv->dev, "%s: exit\n", __func__);
}

