// SPDX-License-Identifier: GPL-2.0
// Copyright(c) 2017-18 Intel Corporation

/*
 *  gna_score.h - Machine Learning Intel Accelerator Header
 */

#ifndef __GNA_SCORE_H__
#define __GNA_SCORE_H__

#include <uapi/misc/gna.h>

#include "gna_hw.h"

#define MAX_GNA_MEMORIES 32

struct gna_file_private;
struct gna_private;
struct gna_request;

struct gna_device_operations {
	struct module *owner;
	int (*getcaps)(struct gna_private *gna_priv, struct gna_capabilities *caps);
	int (*open)(struct gna_private *gna_priv, struct file *fd);
	void (*free)(struct gna_private *gna_priv, struct file *fd);
	int (*score)(struct gna_private *gna_priv, struct gna_request *score_request);
	int (*map)(struct gna_file_private *file_priv, struct gna_usrptr *usrptr);
	int (*unmap)(int memory_id, void *memory, void *data);
};

struct gna_request {
	struct gna_file_private		*file_priv;

	unsigned			request_id;

	u32				hw_status;

	int				status;

	/* executing or in queue */
	bool				active;

	bool				done;

	wait_queue_head_t		waitq;

	struct gna_hw_perf		hw_perf;

	struct gna_drv_perf		drv_perf;

	struct list_head		req_list;

	struct gna_score_cfg		*score_cfg;

	/* protects this structure */
	struct spinlock			lock;
};

struct gna_file_private {
	struct file			*fd;
	struct gna_private		*gna_priv;
	struct idr			memory_idr;

	//struct gna_memory_context	*mem_ctx_arr[MAX_GNA_MEMORIES];

	struct mutex lock;
};


/* add request to the list */
int gna_request_enqueue(struct gna_file_private *file_priv, struct gna_score_cfg *score_cfg, size_t length);

/* find request by id */
struct gna_request *gna_find_request(u64 request_id, struct list_head *list);

/* get next request that needs work */
struct gna_request *gna_dequeue_request(struct list_head *list);

/* delete request by id */
void gna_delete_request(u64 request_id, struct list_head *list);

/* delete all requests that belong to the file */
void gna_delete_file_requests(struct file *fd, struct list_head *list);

/* delete all requests related to memory resource */
void gna_delete_memory_requests(u64 memory_id, struct gna_file_private *file_priv);

/* dequeues request if any and starts the device */
void gna_work(struct work_struct *work);

/* interrupt related functions */
void gna_isr_timeout(struct timer_list *timer);
void gna_request_tasklet(unsigned long);

/* scoring helper functions */
size_t gna_calc_ld_buffersize (struct gna_private *gna_priv,
			const struct gna_score_cfg *input);
int gna_priv_score(struct gna_private *gna_priv,
		struct gna_request *score_request);

int gna_score_wait(struct gna_request *score_request, unsigned int timeout);


#endif // __GNA_SCORE_H__
