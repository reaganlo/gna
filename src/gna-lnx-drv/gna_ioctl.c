// SPDX-License-Identifier: GPL-2.0
// Copyright(c) 2017-18 Intel Corporation

#include "gna_ioctl.h"

#include <linux/uaccess.h>
#include <uapi/misc/gna.h>

#include "gna_drv.h"
#include "gna_mem.h"
#include "gna_score.h"

static int gna_ioctl_score(struct gna_file_private *file_priv, void __user *argptr)
{
	struct gna_score_cfg *full_score_cfg;
	struct gna_score_cfg score_cfg;
	struct gna_private *gna_priv;
	size_t score_cfg_size;
	int ret;

	gna_priv = file_priv->gna_priv;
	if (!gna_priv)
		return -ENODEV;

	if (copy_from_user(&score_cfg, argptr, sizeof(struct gna_score_cfg)))
		return -EFAULT;

	if (score_cfg.flags.copy_whole_descriptors) {
		if(score_cfg.flags.gna_mode == 1)
			score_cfg_size = score_cfg.flags.layer_count * XNN_LYR_DSC_SIZE;
		else
			score_cfg_size = GMM_CFG_SIZE;
	}
	else
		score_cfg_size = gna_calc_ld_buffersize(gna_priv, &score_cfg);

	dev_dbg(&gna_priv->dev, "mode %d cfg size %lu memory_id %llu\n",
			score_cfg.flags.gna_mode, (long unsigned) score_cfg_size,
			score_cfg.memory_id);

	full_score_cfg = kzalloc(score_cfg_size, GFP_KERNEL);
	if (IS_ERR(full_score_cfg)) {
		dev_err(&gna_priv->dev, "could not allocate memory for full score cfg\n");
		return -ENOMEM;
	}

	if (copy_from_user(full_score_cfg, argptr, score_cfg_size)) {
		dev_err(&gna_priv->dev, "could not copy full cfg from user space\n");
		kfree(full_score_cfg);
		return -EFAULT;
	}

	ret = gna_request_enqueue(file_priv, full_score_cfg, score_cfg_size);
	if (ret) {
		dev_err(&gna_priv->dev, "could not enqueue score request\n");
		kfree(full_score_cfg);
		return ret;
	}

	if (copy_to_user(argptr, full_score_cfg, score_cfg_size)) {
		dev_err(&gna_priv->dev, "could not copy score ioctl status to user\n");
		return -EFAULT;
	}

	return 0;
}

static int gna_ioctl_wait(struct file *f, void __user *argptr)
{
	struct gna_file_private *file_priv;
	struct gna_request *score_request;
	struct gna_private *gna_priv;
	struct gna_wait wait_data;
	unsigned long irq_flags;
	long timeout;
	int ret;

	ret = 0;

	if (copy_from_user(&wait_data, argptr, sizeof(wait_data)))
		return -EFAULT;

	file_priv = (struct gna_file_private *) f->private_data;
	if (!file_priv)
		return -ENODEV;

	gna_priv = file_priv->gna_priv;
	if (!gna_priv)
		return -ENODEV;

	dev_dbg(&gna_priv->dev, "%s: enter\n", __func__);

	spin_lock(&gna_priv->reqlist_lock);

	score_request = gna_find_request(wait_data.request_id,
			&gna_priv->request_list);
	if (!score_request) {
		dev_err(&gna_priv->dev, "could not find request with id: %u\n",
			wait_data.request_id);
		return -EINVAL;
	}

	// request exists but does not belong to this file
	// TODO: would EACCES be more suitable for this?
	if (score_request->file_priv->fd != f)
		return -EINVAL;

	spin_unlock(&gna_priv->reqlist_lock);

	if (!score_request) {
		dev_err(&gna_priv->dev, "invalid req id %u\n", wait_data.request_id);
		return -EFAULT;
	}

	dev_dbg(&gna_priv->dev, "found request %u in the queue\n", wait_data.request_id);

	spin_lock_irqsave(&score_request->lock, irq_flags);

	if(score_request->done == true) {
		dev_dbg(&gna_priv->dev, "request already done, excellent\n");
		spin_unlock_irqrestore(&score_request->lock, irq_flags);
		goto copy_request_result;
	}

	spin_unlock_irqrestore(&score_request->lock, irq_flags);

	dev_dbg(&gna_priv->dev, "waiting for request %u for timeout %u\n",
		wait_data.request_id, wait_data.timeout);

	timeout = gna_score_wait(score_request, wait_data.timeout);
	if (timeout > 0)
		goto copy_request_result;

	dev_err(&gna_priv->dev, "request timed out, id: %u\n", wait_data.request_id);
	return -EBUSY;

copy_request_result:
	dev_dbg(&gna_priv->dev, "request wait completed with %d req id %u\n",
		ret, wait_data.request_id);

	spin_lock_irqsave(&score_request->lock, irq_flags);

	dev_dbg(&gna_priv->dev, "request status %d, hw status: %#x\n",
			score_request->status, score_request->hw_status);
	wait_data.hw_perf.total = score_request->hw_perf.total;
	wait_data.hw_perf.stall = score_request->hw_perf.stall;
	wait_data.hw_status = score_request->hw_status;
	ret = score_request->status;

	spin_unlock_irqrestore(&score_request->lock, irq_flags);

	spin_lock(&gna_priv->reqlist_lock);
	gna_delete_request(wait_data.request_id, &gna_priv->request_list);
	spin_unlock(&gna_priv->reqlist_lock);

	if (copy_to_user(argptr, &wait_data, sizeof(wait_data))) {
		dev_err(&gna_priv->dev, "could not copy wait ioctl status to user\n");
		ret = -EFAULT;
	}

	dev_dbg(&gna_priv->dev, "%s: exit\n", __func__);

	return ret;
}

static int gna_ioctl_map(struct gna_file_private *file_priv, void __user *argptr)
{
	struct gna_private *gna_priv;
	struct gna_usrptr usrdata;
	int ret;

	gna_priv = file_priv->gna_priv;
	if (!gna_priv)
		return -EINVAL;

	if (copy_from_user(&usrdata, argptr, sizeof(usrdata)))
		return -EFAULT;

	dev_dbg(&gna_priv->dev, "GNA_MAP_USRPTR len %d ptr %p\n",
			usrdata.length, (void *)usrdata.padd);

	ret = gna_priv->ops->map(file_priv, &usrdata);
	if (ret)
		return ret;

	if (copy_to_user(argptr, &usrdata, sizeof(usrdata)))
		return -EFAULT;

	return 0;
}

static int gna_ioctl_unmap(struct gna_file_private *file_priv, void __user *argptr)
{
	struct gna_memory_ctx *memory_ctx;
	struct gna_private *gna_priv;
	__u64 memory_id;
	int ret;

	gna_priv = file_priv->gna_priv;
	if (!gna_priv)
		return -EINVAL;

	if (copy_from_user(&memory_id, argptr, sizeof(memory_id)))
		return -EFAULT;

	memory_ctx = idr_find(&file_priv->memory_idr, memory_id);
	if (memory_ctx == NULL) {
		dev_err(&gna_priv->dev, "memory id invalid: %llu\n", memory_id);
		return -EINVAL;
	}

	gna_delete_memory_requests(memory_id, file_priv);

	ret = gna_priv->ops->unmap(memory_id, memory_ctx, file_priv);

	idr_remove(&file_priv->memory_idr, memory_id);

	return ret;
}

static int gna_ioctl_cpblts(struct gna_private *gna_priv, void __user *argptr)
{
	struct gna_capabilities devcaps;
	int ret;

	if (copy_from_user(&devcaps, argptr, sizeof(devcaps)))
		return -EFAULT;

	/* do not fail immediately if error is return
	 * function could gather partial info */
	ret = gna_priv->ops->getcaps(gna_priv, &devcaps);

	if (copy_to_user(argptr, &devcaps, sizeof(devcaps)))
		return -EFAULT;

	return ret;
}

long gna_ioctl(struct file *f, unsigned int cmd, unsigned long arg)
{
	struct gna_file_private *file_priv;
	struct gna_private *gna_priv;
	void __user *argptr;
	void *data;
	u64 size;
	int ret;

	argptr = (void __user *) arg;
	data = NULL;
	size = 0;
	ret = 0;

	file_priv = (struct gna_file_private *) f->private_data;
	if (!file_priv)
		return -ENODEV;

	gna_priv = file_priv->gna_priv;
	if (!gna_priv)
		return -ENODEV;

	dev_dbg(&gna_priv->dev, "%s: enter cmd %#x\n", __func__, cmd);

	switch (cmd) {

	case GNA_CPBLTS:

		dev_dbg(&gna_priv->dev, "%s: GNA_CPBLTS command\n", __func__);
		ret = gna_ioctl_cpblts(gna_priv, argptr);
		break;

	case GNA_MAP_USRPTR:

		dev_dbg(&gna_priv->dev, "%s: GNA_MAP_USRPTR command\n", __func__);
		ret = gna_ioctl_map(file_priv, argptr);
		break;

	case GNA_UNMAP_USRPTR:

		dev_dbg(&gna_priv->dev, "%s: GNA_UNMAP_USRPTR command\n", __func__);
		ret = gna_ioctl_unmap(file_priv, argptr);
		break;

	case GNA_SCORE:
		dev_dbg(&gna_priv->dev, "%s: GNA_SCORE command\n", __func__);
		ret = gna_ioctl_score(file_priv, argptr);
		break;

	case GNA_WAIT:
		dev_dbg(&gna_priv->dev, "%s: GNA_WAIT command\n", __func__);
		ret = gna_ioctl_wait(f, argptr);
		break;

	default:
		dev_warn(&gna_priv->dev, "wrong ioctl command: %#x\n", cmd);
		break;
	}

	dev_dbg(&gna_priv->dev, "%s: exit\n", __func__);

	return ret;
}


