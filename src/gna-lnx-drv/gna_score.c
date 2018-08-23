// SPDX-License-Identifier: GPL-2.0
// Copyright(c) 2017-18 Intel Corporation

#include <linux/module.h>
#include <linux/init.h>
#include <linux/err.h>
#include <linux/types.h>
#include <linux/sched.h>
#include <linux/fs.h>
#include <linux/poll.h>
#include <linux/uaccess.h>
#include <linux/device.h>
#include <linux/pm_runtime.h>
#include <linux/time.h>
#include <linux/slab.h>
#include <linux/pci.h>

#include "gna_score.h"

#include "gna_drv.h"

void gna_delete_memory_requests(u64 memory_id, struct gna_file_private *file_priv)
{
	struct gna_request *score_req;
	struct gna_request *temp_req;
	struct gna_private *gna_priv;
	struct file *score_fd;
	u64 score_mem_id;

	gna_priv = file_priv->gna_priv;
	mutex_lock(&gna_priv->lock);

	if (!list_empty(&gna_priv->request_list))
		list_for_each_entry_safe(score_req, temp_req,
				&gna_priv->request_list, req_list) {
			score_fd = score_req->file_priv->fd;
			score_mem_id = score_req->score_cfg->memory_id;
			if (file_priv->fd == score_fd && score_mem_id == memory_id) {
				dev_dbg(&gna_priv->dev, "deleting req %u\n", score_req->request_id);
				list_del(&score_req->req_list);
				kfree(score_req);
				continue;
			}
		}

	mutex_unlock(&gna_priv->lock);
}

static struct gna_request *gna_create_request
	(struct gna_file_private *file_priv,
	 struct gna_score_cfg *score_cfg)
{
	struct gna_request *score_request;
	struct gna_private *gna_priv;

	gna_priv = file_priv->gna_priv;
	if (IS_ERR(gna_priv))
		return NULL;

	dev_dbg(&gna_priv->dev, "%s: enter\n", __func__);

	score_request = kzalloc(sizeof(*score_request), GFP_KERNEL | GFP_ATOMIC);
	if (IS_ERR(score_request))
		PTR_ERR(score_request);

	dev_dbg(&gna_priv->dev, "config_base %d layer_count %d\n",
		score_cfg->flags.config_base, score_cfg->flags.layer_count);

	score_request->request_id = atomic_inc_return(&gna_priv->request_count);
	score_request->score_cfg = score_cfg;
	score_request->file_priv = file_priv;
	score_request->active = false;
	score_request->done = false;

	init_waitqueue_head(&score_request->waitq);

	score_cfg->request_id = score_request->request_id;

	dev_dbg(&gna_priv->dev, "%s: exit\n", __func__);
	return score_request;
}

int gna_request_enqueue(struct gna_file_private *file_priv,
		struct gna_score_cfg *score_cfg, size_t length)
{
	struct gna_request *score_request;
	struct gna_private *gna_priv;
	int ret;

	if (IS_ERR(file_priv))
		return PTR_ERR(file_priv);

	gna_priv = file_priv->gna_priv;

	dev_dbg(&gna_priv->dev, "%s: enter\n", __func__);

	dev_dbg(&gna_priv->dev, "gna_mode %d\n", score_cfg->flags.gna_mode);

	ret = 0;

	score_request = gna_create_request(file_priv, score_cfg);
	if (IS_ERR(score_request)) {
		ret = PTR_ERR(score_request);
		goto end;
	}

	dev_dbg(&gna_priv->dev, "created new request %u\n",
			score_request->request_id);

	spin_lock(&gna_priv->reqlist_lock);
	list_add_tail(&score_request->req_list, &gna_priv->request_list);
	spin_unlock(&gna_priv->reqlist_lock);
	dev_dbg(&gna_priv->dev, "request %u added to req list\n",
			score_request->request_id);

	queue_work(gna_priv->callback_wq, &gna_priv->score_work);

	dev_dbg(&gna_priv->dev, "queued work\n");
end:

	dev_dbg(&gna_priv->dev, "%s: exit\n", __func__);
	return ret;
}

/* should be call with held gna_private reqlist_lock */
struct gna_request *gna_find_request(u64 request_id, struct list_head *list)
{
	struct gna_request *req;

	list_for_each_entry(req, list, req_list) {
		if (request_id == req->request_id)
			return req;
	}

	return NULL;
}

/* should be call with held gna_private reqlist_lock */
struct gna_request *gna_dequeue_request(struct list_head *list)
{
	struct gna_request *req;
	struct gna_request *temp_req;

	if (list_empty(list))
		return NULL;

	list_for_each_entry_safe(req, temp_req,
				 list, req_list) {
		if (req->done == false && req->active == false)
			return req;
	}

	return NULL;
}


/* should be call with held gna_private reqlist_lock */
void gna_delete_request(u64 request_id, struct list_head *list)
{
	struct gna_request *temp_req;
	struct gna_request *req;

	/*Free from the list*/
	if (!list_empty(list)) {
		list_for_each_entry_safe(req, temp_req,
					 list, req_list) {
			if (req->request_id == request_id) {
				list_del(&req->req_list);
				kfree(req->score_cfg);
				kfree(req);
				break;
			}
		}
	}
}

/* should be call with held gna_private reqlist_lock */
void gna_delete_file_requests(struct file *fd, struct list_head *list)
{
	struct gna_request *temp_req;
	struct gna_request *req;

	/*Free from the list*/
	if (!list_empty(list)) {
		list_for_each_entry_safe(req, temp_req,
					 list, req_list) {
			if (req->file_priv->fd == fd) {
				list_del(&req->req_list);
				kfree(req->score_cfg);
				kfree(req);
				break;
			}
		}
	}
}

size_t gna_calc_ld_buffersize (struct gna_private *gna_priv,
			const struct gna_score_cfg *input)
{
	const size_t max_ = (size_t)max
				(sizeof(struct gna_gmm_al_descr),
				sizeof(struct gna_xnn_al_descr));

	size_t sz = sizeof(struct gna_score_cfg);

	dev_dbg(&gna_priv->dev, "buffers_count = %d\n", input->req_cfg_desc.buffer_count);
	dev_dbg(&gna_priv->dev, "nnop_types_count = %d\n", input->req_cfg_desc.nnop_type_count);
	dev_dbg(&gna_priv->dev, "xnn_al_count = %d\n", input->req_cfg_desc.xnn_al_count);
	dev_dbg(&gna_priv->dev, "gmm_al_count = %d\n", input->req_cfg_desc.gmm_al_count);

	sz += sizeof(struct gna_buffer_descr) * input->req_cfg_desc.buffer_count;
	sz += sizeof(struct gna_nnop_descr) * input->req_cfg_desc.nnop_type_count;
	sz += max_ * input->req_cfg_desc.xnn_al_count;
	sz += max_ * input->req_cfg_desc.gmm_al_count;

	dev_dbg(&gna_priv->dev, "buffers config size = %ld\n", sz);

	return ALIGN(sz, sizeof(__u64));
}

static void score_set_done(struct gna_request *score_request, int status)
{
	score_request->status = status;
	score_request->done = true;
	score_request->active = false;
}

void gna_work(struct work_struct *work)
{
	struct gna_request *score_request;
	struct gna_private *gna_priv;
	unsigned long irq_flags;
	u64 *start_hw;
	u64 *score_hw;
	int ret;

	gna_priv = container_of(work, struct gna_private, score_work);
	if (IS_ERR(gna_priv))
		return;

	dev_dbg(&gna_priv->dev, "%s: enter\n", __func__);

	mutex_lock(&gna_priv->lock);

	if (gna_priv->busy == true) {
		dev_dbg(&gna_priv->dev, "gna device is busy, we should come back later\n");
		mutex_unlock(&gna_priv->lock);
		return;
	}

	mutex_unlock(&gna_priv->lock);

	dev_dbg(&gna_priv->dev, "looking for request to work with\n");
	spin_lock(&gna_priv->reqlist_lock);
	score_request = gna_dequeue_request(&gna_priv->request_list);
	spin_unlock(&gna_priv->reqlist_lock);
	if (score_request == NULL) {
		dev_dbg(&gna_priv->dev, "didn't find any request to work with\n");
		return;
	}

	spin_lock_irqsave(&score_request->lock, irq_flags);

	dev_dbg(&gna_priv->dev, "found request to work with: %p, id: %u\n",
			score_request, score_request->request_id);

	ret = pm_runtime_get_sync(&gna_priv->pdev->dev);
	if (ret < 0) {
		dev_warn_once(&gna_priv->dev,
			"pm_runtime_get_sync() failed: %d\n", ret);

		score_set_done(score_request, -ENODEV);

		spin_unlock_irqrestore(&score_request->lock, irq_flags);
		return;
	}


	start_hw = &score_request->drv_perf.start_hw;
	*start_hw = rdtsc();

	/* request dequeued, mark as active */
	score_request->active = true;

	/* mark gna device as busy */
	mutex_lock(&gna_priv->lock);
	gna_priv->current_request = score_request;
	gna_priv->busy = true;

	ret = gna_priv->ops->score(gna_priv, score_request);
	if (ret) {
		gna_priv->current_request = NULL;
		gna_priv->busy = false;

		score_set_done(score_request, ret);

		ret = pm_runtime_put_sync(&gna_priv->pdev->dev);
		if (ret < 0)
			dev_warn_once(&gna_priv->dev,
				"pm_runtime_put_sync() failed: %d\n", ret);
	}
	mutex_unlock(&gna_priv->lock);

	*start_hw = rdtsc() - *start_hw;

	score_hw = &score_request->drv_perf.score_hw;
	*score_hw = rdtsc();

	spin_unlock_irqrestore(&score_request->lock, irq_flags);

	dev_dbg(&gna_priv->dev, "%s: exit", __func__);
}

void gna_request_tasklet(unsigned long priv)
{
	struct gna_request *score_request;
	struct gna_private *gna_priv;
	unsigned long irq_flags;
	void __iomem *addr;
	bool work_queued;
	u64 *intr_proc;
	u64 *score_hw;
	u32 hw_status;
	int isr_left;
	int status;
	int ret;

	gna_priv = (struct gna_private *) priv;
	if (IS_ERR(gna_priv)) {
		return;
	}

	del_timer(&gna_priv->isr_timer);

	dev_dbg(&gna_priv->dev, "%s: enter\n", __func__);

	addr = gna_priv->bar0.mem_addr;

	score_request = gna_priv->current_request;
	if (IS_ERR(score_request)) {
		dev_err(&gna_priv->dev, "gna score request is unavailable\n");
		return;
	}

	gna_priv->current_request = NULL;
	gna_priv->busy = false;

	spin_lock_irqsave(&score_request->lock, irq_flags);

	intr_proc = &score_request->drv_perf.intr_proc;
	*intr_proc = rdtsc();

	score_hw = &score_request->drv_perf.score_hw;
	*score_hw = rdtsc() - *score_hw;

	hw_status = score_request->hw_status;

	score_request->done = true;
	score_request->active = false;

	if (hw_status & GNA_STS_STATISTICS_VALID) {
		dev_dbg(&gna_priv->dev, "GNA statistics calculated successfully\n");
		score_request->hw_perf.total = gna_reg_read(addr, GNAPTC);
		score_request->hw_perf.stall = gna_reg_read(addr, GNAPSC);
		dev_dbg(&gna_priv->dev, "GNAPTC %llu\n", score_request->hw_perf.total);
		dev_dbg(&gna_priv->dev, "GNAPSC %llu\n", score_request->hw_perf.stall);
	} else {
		dev_warn(&gna_priv->dev, "GNA statistics missing\n");
		score_request->hw_perf.total = 0;
		score_request->hw_perf.stall = 0;
	}

	spin_unlock_irqrestore(&score_request->lock, irq_flags);

	spin_lock_irqsave(&gna_priv->irq_lock, irq_flags);
	gna_abort_hw(gna_priv, addr);
	spin_unlock_irqrestore(&gna_priv->irq_lock, irq_flags);

	if (hw_status & GNA_STS_SCORE_COMPLETED) {
		dev_info(&gna_priv->dev, "scoring completed with status %#x\n", hw_status);
		status = 0;
	} else {
		dev_err(&gna_priv->dev, "scoring not completed, status: %#x\n", hw_status);
		status = -EIO;
	}

	if (hw_status & GNA_STS_PARAM_OOR) {
		dev_err(&gna_priv->dev, "scoring error: Param Out Range Error\n");
	} else if (hw_status & GNA_STS_VA_OOR) {
		dev_err(&gna_priv->dev, "scoring error: VA Out of Range Error\n");
	} else if (hw_status & GNA_STS_PCI_MMU_ERR) {
		dev_err(&gna_priv->dev, "scoring error: PCI MMU Error\n");
	} else if (hw_status & GNA_STS_PCI_DMA_ERR) {
		dev_err(&gna_priv->dev, "scoring error: PCI MMU Error\n");
	} else if (hw_status & GNA_STS_PCI_UNEXCOMPL_ERR) {
		dev_err(&gna_priv->dev, "scoring error: PCI Unexpected Completion Error\n");
	} else if (hw_status & GNA_STS_SATURATE) {
		dev_warn(&gna_priv->dev, "scoring error: Saturation Reached !\n");
	}

	spin_lock_irqsave(&score_request->lock, irq_flags);

	score_request->status = status;

	*intr_proc = rdtsc() - *intr_proc;

	spin_unlock_irqrestore(&score_request->lock, irq_flags);

	/* wake up waiting process */
	dev_dbg(&gna_priv->dev, "request done, waking user process\n");

	/* device is ready and request is done, queue new work */
	work_queued = queue_work(gna_priv->callback_wq, &gna_priv->score_work);
	if (work_queued)
		dev_dbg(&gna_priv->dev, "queued new work\n");
	else
		dev_dbg(&gna_priv->dev, "work was already queued\n");

	wake_up_interruptible(&score_request->waitq);
	dev_dbg(&gna_priv->dev, "woke up user process\n");

	/* reschedule itself if another interrupt came in the meantime */
	/* unlikely since tasklet acquires irq spin lock and disables interrupts */
	isr_left = atomic_dec_return(&gna_priv->isr_count);
	if (isr_left) {
		dev_dbg(&gna_priv->dev, "scheduling another tasklet\n");
		tasklet_schedule(&gna_priv->request_tasklet);
	}

	ret = pm_runtime_put(&gna_priv->pdev->dev);
	if (ret < 0)
		dev_warn_once(&gna_priv->dev,
			"pm_runtime_put_sync() failed: %d\n", ret);

	dev_dbg(&gna_priv->dev, "%s: exit\n", __func__);
}

void gna_isr_timeout(struct timer_list *timer)
{
	struct gna_request *score_request;
	struct gna_private *gna_priv;
	unsigned long irq_flags;

	gna_priv = from_timer(gna_priv, timer, isr_timer);

	dev_dbg(&gna_priv->dev, "%s enter\n", __func__);

	spin_lock_irqsave(&gna_priv->irq_lock, irq_flags);
	score_request = gna_priv->current_request;
	spin_unlock_irqrestore(&gna_priv->irq_lock, irq_flags);

	if (score_request) {
		dev_err(&gna_priv->dev, "request id %d timeout\n", score_request->request_id);

		score_request->status = -ETIME;
		score_request->hw_status = gna_reg_read(gna_priv->bar0.mem_addr, GNASTS);

		atomic_inc(&gna_priv->isr_count);
		tasklet_schedule(&gna_priv->request_tasklet);
	}

	dev_dbg(&gna_priv->dev, "%s exit\n", __func__);
}

static void gna_set_ld_parameters(struct gna_private *gna_priv,
		void *cfgdata, struct gna_score_cfg *input,
		struct gna_memory_context *memctx)
{
	struct gna_xnn_al_descr *xnn_al_desc;
	struct gna_gmm_al_descr *gmm_al_desc;
	struct gna_nnop_descr *nnop_type_desc;
	struct gna_buffer_descr *buffer_desc;
	u8 *membase;
	u32 i;

	membase = cfgdata;

	dev_dbg(&gna_priv->dev, "input_cfg_id %d model_cfg_id %lld\n",
		 input->req_cfg_desc.request_cfg_id, memctx->request_config_id);

	if (input->req_cfg_desc.model_id == memctx->model_id &&
	    input->req_cfg_desc.request_cfg_id == memctx->request_config_id) {
		dev_dbg(&gna_priv->dev, "same request config as in previous request\n");
		return;
	}

	memctx->request_config_id = input->req_cfg_desc.request_cfg_id;

	dev_dbg(&gna_priv->dev, "buffers_count %d\n", input->req_cfg_desc.buffer_count);
	/* set buffers according to request config */
	buffer_desc = (struct gna_buffer_descr *)
		       ((u8 *)input + sizeof(struct gna_score_cfg));
	for (i = 0; i < input->req_cfg_desc.buffer_count; ++i) {
		*(uint32_t *)(membase + buffer_desc->offset) =
				buffer_desc->value;
		++buffer_desc;
	}

	dev_dbg(&gna_priv->dev, "nnop_types_count %d\n", input->req_cfg_desc.nnop_type_count);
	/* set nnop type */
	nnop_type_desc = (struct gna_nnop_descr *)buffer_desc;
	for (i = 0; i < input->req_cfg_desc.nnop_type_count; ++i) {
		*(uint8_t *)(membase + nnop_type_desc->offset) =
				nnop_type_desc->value;
		++nnop_type_desc;
	}

	dev_dbg(&gna_priv->dev, "xnn_al_count %d\n", input->req_cfg_desc.xnn_al_count);
	/* set xnn active list params according to request config */
	xnn_al_desc = (struct gna_xnn_al_descr *)nnop_type_desc;
	for (i = 0; i < input->req_cfg_desc.xnn_al_count; ++i) {
		*(uint32_t *)(membase +
				xnn_al_desc->al_buffer_offset) =
				xnn_al_desc->al_buffer_value;
		*(uint32_t *)(membase +
				xnn_al_desc->al_n_elems_offset) =
				xnn_al_desc->al_n_elems_value;
		++xnn_al_desc;
	}

	dev_dbg(&gna_priv->dev, "gmm_al_count %d\n", input->req_cfg_desc.gmm_al_count);
	/* set gmm active list params according to request config */
	gmm_al_desc = (struct gna_gmm_al_descr *)xnn_al_desc;
	for (i = 0; i < input->req_cfg_desc.gmm_al_count; ++i) {
		*(uint32_t *)(membase + gmm_al_desc->asl_addr_offset) =
					gmm_al_desc->asl_addr_value;
		*(uint32_t *)(membase + gmm_al_desc->asl_len_offset) =
					gmm_al_desc->asl_len_value;
		*(uint32_t *)(membase + gmm_al_desc->gmm_scrlen_offset) =
					gmm_al_desc->gmm_scrlen_value;
		++gmm_al_desc;
	}
}

int gna_score_wait(struct gna_request *score_request, unsigned int timeout)
{
	struct timeval time_val;
	time_val.tv_sec = timeout / 1000;
	time_val.tv_usec = (timeout % 1000) * 1000;
	return wait_event_interruptible_timeout
		(score_request->waitq,
		 score_request->done == true,
		 timeval_to_jiffies(&time_val));
}

int gna_priv_score(struct gna_private *gna_priv, struct gna_request *score_request)
{
	struct gna_memory_context *memory_ctx;
	struct gna_file_private *file_priv;
	struct gna_hw_descriptor *hwdesc;
	struct gna_score_cfg *score_cfg;
	unsigned long irq_flags;
	size_t score_cfg_size;
	void __iomem *addr;
	u64 memory_id;
	u32 desc_base;
	void *cfg_data;
	void *gmmdesc;
	void *xnndesc;
	int size;

	if (!gna_priv)
		return -EINVAL;

	dev_dbg(&gna_priv->dev, "%s: enter", __func__);

	if (!score_request) {
		dev_err(&gna_priv->dev, "no score request structure\n");
		return -EINVAL;
	}

	score_cfg = score_request->score_cfg;
	if (!score_cfg) {
		dev_err(&gna_priv->dev, "no score config for request\n");
		return -EINVAL;
	}

	memory_id = score_cfg->memory_id;
	size = 0;

	file_priv = (struct gna_file_private *) score_request->file_priv;
	if (!file_priv) {
		dev_err(&gna_priv->dev, "no app context for request\n");
		return -EINVAL;
	}

	mutex_lock(&file_priv->lock);

	memory_ctx = idr_find(&file_priv->memory_idr, memory_id);
	if (!memory_ctx) {
		dev_err(&gna_priv->dev, "no model context found\n");
		mutex_unlock(&file_priv->lock);
		return -EINVAL;
	}

	mutex_unlock(&file_priv->lock);

	spin_lock_irqsave(&gna_priv->irq_lock, irq_flags);

	/* switch mmu by updating descbase register */
	addr = gna_priv->bar0.mem_addr;
	desc_base = (u32)(memory_ctx->hwdesc_dma >> PAGE_SHIFT);
	gna_reg_write(addr, GNADESBASE, desc_base);

	/* copy the config data */
	cfg_data = memory_ctx->mapped;

	hwdesc = memory_ctx->hwdesc;
	if (score_cfg->flags.gna_mode == 1) {
		dev_dbg(&gna_priv->dev, "xNN mode, labase: %d, lacount: %d\n",
			score_cfg->flags.config_base, score_cfg->flags.layer_count);
		hwdesc->xnn_config.labase = score_cfg->flags.config_base;
		hwdesc->xnn_config.lacount = (u16)score_cfg->flags.layer_count;

		if (score_cfg->flags.copy_whole_descriptors) {
			const void *payload = (uint8_t*)score_cfg + sizeof(struct gna_score_cfg);
			score_cfg_size = score_cfg->flags.layer_count * XNN_LYR_DSC_SIZE;
			xnndesc = (void *)((char*)cfg_data + score_cfg->flags.config_base);
			memcpy(xnndesc, payload, score_cfg_size);
		}
		else
			gna_set_ld_parameters(gna_priv, cfg_data, score_cfg, memory_ctx);

	} else {
		dev_dbg(&gna_priv->dev, "GMM mode, offset: %d\n",
				score_cfg->flags.config_base);
		gmmdesc = (void *)((char*)cfg_data + score_cfg->flags.config_base);
		memcpy((void *)(&hwdesc->xnn_config), gmmdesc, GMM_CFG_SIZE);
	}

	gna_start_scoring(gna_priv, addr, score_cfg);

	spin_unlock_irqrestore(&gna_priv->irq_lock, irq_flags);

	dev_dbg(&gna_priv->dev, "%s: exit\n", __func__);

	return 0;
}

