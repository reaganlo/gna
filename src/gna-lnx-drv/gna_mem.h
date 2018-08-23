// SPDX-License-Identifier: GPL-2.0
// Copyright(c) 2017-18 Intel Corporation

#ifndef __GNA_MEM_H__
#define __GNA_MEM_H__

#include <linux/types.h>

#include "gna_hw.h"

struct gna_usrptr;

struct gna_file_private;

struct gna_xnn_descriptor {
    u32 labase;
    u16 lacount;
    u16 _rsvd;
};

struct gna_mmu {
    u32 vamaxaddr;
    u8 __res_204[12];
    u32 pagedir_n[GNA_PGDIRN_LEN];
};

struct gna_hw_descriptor {
    u8 __res_0000[256];
    union {
	    struct gna_xnn_descriptor xnn_config;
    };
    u8 __unused[248];
    struct gna_mmu mmu;
};

struct descriptor_addr {
	struct descriptor	*descla;
	dma_addr_t		descph;
};

struct gna_memory_context {
	u64			memory_id;
	u64			memory_size;

	u64			model_id;
	s64			request_config_id;

	struct gna_hw_descriptor	*hwdesc;
	dma_addr_t			hwdesc_dma;
	u32				*pagedirs;
	dma_addr_t			pagedirs_dma;
	struct sg_table			*sgt;
	struct page			**pages;
	void				*mapped;

	int				num_pages;
	int				num_pagedirs;
	int				pinned;
	int				sgcount;
};

extern int gna_priv_map(struct gna_file_private *file_priv, struct gna_usrptr *data);

extern int gna_priv_unmap(int memory_id, void *memory, void *data);

#endif // __GNA_MEM_H__
