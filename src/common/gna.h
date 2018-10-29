// SPDX-License-Identifier: GPL-2.0
// Copyright(c) 2017-18 Intel Corporation

#ifndef _UAPI_GNA_H_
#define _UAPI_GNA_H_

#if defined(__cplusplus)
extern "C" {
#endif

#include <linux/types.h>
#include <linux/ioctl.h>

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
	GNA_DEV_TGL	= 0x9A11
};

struct gna_usrptr {
	__u64				memory_id;
	__u64				padd;
	__u32				length;
} __attribute__((packed));

struct gna_capabilities {
	__u32				in_buff_size;
	__u32				recovery_timeout;
	enum gna_device_t		device_type;
} __attribute__((packed));

struct gna_ctrl_flags {
	__u32				active_list_on:1;
	__u32				gna_mode:2;
	__u32				reserved:29;
	__u32				config_base;
	__u32				layer_count;
} __attribute__((packed));

/**
 * Structure describes part of memory to be overwritten before starting GNA
 */
struct gna_memory_patch {
	__u64				offset;
	__u64				size;
	__u8				data[];
} __attribute__((packed));

struct gna_drv_perf {
	__u64				start_hw;
	__u64				score_hw;
	__u64				intr_proc;
} __attribute__((packed));

struct gna_hw_perf {
	__u64				total;
	__u64				stall;
} __attribute__((packed));

struct gna_score_cfg {

	__u64			request_id;
	__u64			memory_id;
	__u64			config_size;
	__u64			patch_count;

	struct gna_ctrl_flags	flags;
	__u8			hw_perf_encoding;

	__u8			patches[];

} __attribute__((packed));

struct gna_wait {
	/* user input */
	__u32			request_id;
	__u32			timeout;

	/* user output */
	__u32			hw_status;
	struct gna_drv_perf	drv_perf;
	struct gna_hw_perf	hw_perf;
} __attribute__((packed));

/**
 * Intel GMM & Neural Network Accelerator ioctl definitions
 *
 * GNA_DRV_MAP_USRPTR: Map the userspace address to physical pages
 * GNA_DRV_UNMAP_USRPTR: Unmap the userspace address
 * GNA_DRV_SCORE: Submit scoring requet/Start scoring operation
 * GNA_DRV_WAIT: Blocking call to query about a specific score request
 * GNA_DRV_CPBLTS: Get the device capabilities
 */
#define GNA_MAP_USRPTR		_IOWR('C', 0x01, struct gna_usrptr)
#define GNA_UNMAP_USRPTR	 _IOR('C', 0x02, __u64)
#define GNA_SCORE		_IOWR('C', 0x03, struct gna_score_cfg)
#define GNA_WAIT		_IOWR('C', 0x04, struct gna_wait)
#define GNA_CPBLTS		_IOWR('C', 0x05, struct gna_capabilities)

#if defined(__cplusplus)
}
#endif

#endif /* _UAPI_GNA_H_ */

