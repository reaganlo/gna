// SPDX-License-Identifier: GPL-2.0
// Copyright(c) 2017-18 Intel Corporation

#ifndef _UAPI_GNA_H_
#define _UAPI_GNA_H_

#if defined(__cplusplus)
extern "C" {
#endif

#include <linux/types.h>
#include <linux/ioctl.h>

/* Request processing flags */
#define GNA_SCORE_COPY_DESCRIPTOR	(1 <<  0)

/* GNA parameters */
#define GNA_PARAM_DEVICE_ID		(1 <<  0)
#define GNA_PARAM_RECOVERY_TIMEOUT	(1 <<  1)
#define GNA_PARAM_IBUFFS		(1 <<  2)

#define GNA_CFG_SIZE 256

#define GNA_STS_SCORE_COMPLETED		(1 <<  0)
#define GNA_STS_STATISTICS_VALID	(1 <<  3)
#define GNA_STS_PCI_MMU_ERR		(1 <<  4)
#define GNA_STS_PCI_DMA_ERR		(1 <<  5)
#define GNA_STS_PCI_UNEXCOMPL_ERR	(1 <<  6)
#define GNA_STS_PARAM_OOR		(1 <<  7)
#define GNA_STS_VA_OOR			(1 <<  8)
#define GNA_STS_OUTBUF_FULL		(1 << 16)
#define GNA_STS_SATURATE		(1 << 17)

#define GNA_ERROR (GNA_STS_PCI_DMA_ERR | \
		GNA_STS_PCI_MMU_ERR | \
		GNA_STS_PCI_UNEXCOMPL_ERR | \
		GNA_STS_PARAM_OOR | \
		GNA_STS_VA_OOR)

/**
 *  Enumeration of device flavors
 *  Hides gna_device_kind
 */
enum gna_device_t {
	GNA_NO_DEVICE	= 0x0000,
	GNA_DEV_CNL	= 0x5A11,
	GNA_DEV_GLK	= 0x3190,
	GNA_DEV_EHL	= 0x4511,
	GNA_DEV_ICL	= 0x8A11,
	GNA_DEV_TGL	= 0x9A11,
	GNA_DEV_ADL	= 0x46AD
};

struct gna_userptr {
	__u64 memory_id;
	__u64 user_address;
	__u32 user_size;
} __attribute__((packed));

struct gna_ctrl_flags {
	__u32 active_list_on:1;
	__u32 gna_mode:2;
	__u32 hw_perf_encoding:8;
	__u32 reserved:21;
} __attribute__((packed));

struct gna_getparam {
	__u64 param;
	__u64 value;
} __attribute__((packed));

/**
 * Structure describes part of memory to be overwritten before starting GNA
 */
struct gna_patch {
	/* offset from targeted memory */
	__u64 offset;

	__u64 size;
	union {
		__u64 value;
		void *user_ptr;
	};
} __attribute__((packed));

struct gna_buffer {
	__u64 memory_id;

	__u64 offset;
	__u64 size;

	__u64 patch_count;
	__u64 patches_ptr;
} __attribute__((packed));

struct gna_drv_perf {
	__u64 start_hw;
	__u64 score_hw;
	__u64 intr_proc;
} __attribute__((packed));

struct gna_hw_perf {
	__u64 total;
	__u64 stall;
} __attribute__((packed));

struct gna_score_cfg {

	/* Flags applied to GNA control register */
	struct gna_ctrl_flags ctrl_flags;

	/* Request processing flags */
	__u64 flags;

	__u64 request_id;

	union gna_desc_cfg {
		__u8 descriptor[GNA_CFG_SIZE];
		struct gna_xnn_config {
			__u32 layer_base;
			__u32 layer_count;
		} xnn_cfg;
	} desc_cfg;

	/* List of GNA memory buffers */
	__u64 buffers_ptr;
	__u64 buffer_count;

} __attribute__((packed));

struct gna_wait {
	/* user input */
	__u64 request_id;
	__u32 timeout;

	/* user output */
	__u32 hw_status;
	struct gna_drv_perf drv_perf;
	struct gna_hw_perf hw_perf;
} __attribute__((packed));

#define GNA_IOCTL_USERPTR	_IOWR('C', 0x01, struct gna_userptr)
#define GNA_IOCTL_FREE		_IOWR('C', 0x01, __u64)
#define GNA_IOCTL_SCORE		_IOWR('C', 0x02, struct gna_score_cfg)
#define GNA_IOCTL_WAIT		_IOWR('C', 0x03, struct gna_wait)
#define GNA_IOCTL_GETPARAM	_IOWR('C', 0x04, struct gna_getparam)

#if defined(__cplusplus)
}
#endif

#endif /* _UAPI_GNA_H_ */

