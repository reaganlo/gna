// SPDX-License-Identifier: GPL-2.0
// Copyright(c) 2017-18 Intel Corporation

#include "gna_mem.h"

#include "gna_score.h"

#include "gna_drv.h"

#include <uapi/misc/gna.h>
#include <linux/types.h>
#include <linux/sched.h>
#include <linux/device.h>
#include <linux/mm.h>
#include <linux/swap.h>
#include <linux/init.h>
#include <linux/slab.h>
#include <linux/pci.h>

#define ROUND_UP(x,n) (((x)+(n)-1u) & ~((n)-1u))

void gna_memctx_initialize(struct gna_memory_context *memory_ctx, u64 memory_id,
			struct gna_usrptr *data)
{
	memory_ctx->memory_id = memory_id;
	memory_ctx->model_id = -1;
	memory_ctx->request_config_id = -1;

	memory_ctx->memory_size = data->length;
	memory_ctx->num_pages = ROUND_UP(data->length, PAGE_SIZE) >> PAGE_SHIFT;
	memory_ctx->num_pagedirs = DIV_N_CEIL(memory_ctx->num_pages,
						GNA_PGDIR_ENTRIES);
}

/* descriptor and page tables allocation */
static int gna_mmu_alloc(struct gna_private *gna_priv,
		struct gna_memory_context *memory_ctx)
{
	int pagedir_size;
	int desc_size;
	int cfg_size;

	dev_dbg(&gna_priv->dev, "%s: enter\n", __func__);

	cfg_size = ROUND_UP(gna_priv->info.desc_info.cfg_size, PAGE_SIZE);
	pagedir_size = memory_ctx->num_pagedirs * PAGE_SIZE;
	desc_size = cfg_size + pagedir_size;

	dev_dbg(&gna_priv->dev, "desc size %d\n", desc_size);

	memory_ctx->hwdesc = pci_alloc_consistent(gna_priv->pdev,
						     desc_size,
						     &memory_ctx->hwdesc_dma);
	if (!memory_ctx->hwdesc) {
		dev_err(&gna_priv->dev, "gna descriptor alloc fail\n");
		return -ENOMEM;
	}

	memory_ctx->pagedirs = (u32 *)((u8 *)memory_ctx->hwdesc + PAGE_SIZE);
	memory_ctx->pagedirs_dma = memory_ctx->hwdesc_dma + PAGE_SIZE;

	memset(memory_ctx->hwdesc, 0, gna_priv->info.desc_info.cfg_size);

	memory_ctx->hwdesc->mmu.vamaxaddr = memory_ctx->memory_size;

	dev_dbg(&gna_priv->dev, "%s: exit\n", __func__);

	return 0;
}

static void gna_mmu_free(struct gna_file_private *file_priv,
		struct gna_memory_context *memory_ctx)
{
	struct gna_private *gna_priv;
	int desc_size;
	int cfg_size;
	int pagedir_size;

	gna_priv = file_priv->gna_priv;
	dev_dbg(&gna_priv->dev, "%s enter\n", __func__);

	cfg_size = ROUND_UP(gna_priv->info.desc_info.cfg_size, PAGE_SIZE);
	pagedir_size = memory_ctx->num_pagedirs * PAGE_SIZE;
	desc_size = cfg_size + pagedir_size;

	dev_dbg(&gna_priv->dev, "%s: desc size %d\n", __func__, desc_size);

	pci_free_consistent(gna_priv->pdev, desc_size,
	    memory_ctx->hwdesc, memory_ctx->hwdesc_dma);

	dev_dbg(&gna_priv->dev, "%s: exit\n", __func__);
}

static int gna_map_pages(struct gna_private *gna_priv,
			struct gna_usrptr *data,
			struct gna_memory_context *memory_ctx)
{
	struct mm_struct *mm;
	struct page **pages;
	struct sg_table *sgt;
	int num_pinned;
	int num_pages;
	void *mapped;
	int sgcount;
	void *padd;
	int ret;

	dev_dbg(&gna_priv->dev, "user memory size %d\n", data->length);

	num_pages = memory_ctx->num_pages;
	padd = (void *)data->padd;
	mm = current->mm;

	pages = kmalloc_array(num_pages, sizeof(*pages), GFP_KERNEL);
	if (IS_ERR(pages)) {
		ret = PTR_ERR(pages);
		goto err_exit;
	}

	dev_dbg(&gna_priv->dev, "pages allocated\n");

	ret = get_user_pages_fast(data->padd, num_pages, 1, pages);
	num_pinned = ret;
	if (ret <= 0) {
		dev_err(&gna_priv->dev, "function get_user_pages_fast() failed\n");
		goto err_free_pages;
	}
	if (ret < num_pages) {
		dev_err(&gna_priv->dev, "function get_user_pages_fast() pinned less pages\n");
		goto err_free_pages;
	}

	dev_dbg(&gna_priv->dev, "get user pages success %d\n", ret);

	mapped = vm_map_ram(pages, num_pages, 0, PAGE_KERNEL);
	if (mapped == NULL) {
		dev_err(&gna_priv->dev, "could not map pages into linear");
		goto err_free_pages;
	}

	sgt = kmalloc(sizeof(struct sg_table), GFP_KERNEL);
	if (IS_ERR(sgt)) {
		ret = PTR_ERR(sgt);
		dev_err(&gna_priv->dev, "could not allocate memory for scatter-gather table\n");
		goto err_unmap_ram;
	}

	dev_dbg(&gna_priv->dev, "sgt allocated\n");

	ret = sg_alloc_table_from_pages(sgt, pages, num_pinned, 0, data->length, GFP_KERNEL);
	if (ret) {
		dev_err(&gna_priv->dev, "could not alloc scatter list \n");
		goto err_free_sgt;
	}

	if (IS_ERR(sgt->sgl)) {
		dev_err(&gna_priv->dev, "sgl allocation failed\n");
		ret = PTR_ERR(sgt->sgl);
		goto err_free_sgt;
	}

	dev_dbg(&gna_priv->dev, "sgl allocated\n");

	sgcount = pci_map_sg(gna_priv->pdev, sgt->sgl, sgt->nents, PCI_DMA_BIDIRECTIONAL);
	if (sgcount <= 0) {
		dev_err(&gna_priv->dev, "could not map scatter gather list \n");
		goto err_free_sgl;
	}

	dev_dbg(&gna_priv->dev, "mapped scatter gather list\n");

	memory_ctx->sgt = sgt;
	memory_ctx->sgcount = sgcount;
	memory_ctx->mapped = mapped;
	memory_ctx->pages = pages;
	memory_ctx->pinned = num_pinned;

	return 0;

err_free_sgl:
	sg_free_table(sgt);

err_free_sgt:
	kfree(sgt);

err_unmap_ram:
	vm_unmap_ram(mapped, num_pages);

err_free_pages:
	kfree(pages);

err_exit:
	return ret;
}

static void gna_unmap_pages(struct gna_private *gna_priv,
		struct gna_memory_context *memory_ctx)
{
	struct sg_table *sgt;

	sgt = memory_ctx->sgt;

	pci_unmap_sg(gna_priv->pdev, sgt->sgl, sgt->nents, PCI_DMA_BIDIRECTIONAL);
	sg_free_table(sgt);
	kfree(sgt);
	kfree(memory_ctx->pages);
}

static void gna_mmu_fill_l1(struct gna_private *gna_priv,
		struct gna_memory_context *memory_ctx)
{
	dma_addr_t pgdir_dma;
	u32 *pgdirn;
	int i;

	pgdirn = memory_ctx->hwdesc->mmu.pagedir_n;
	pgdir_dma = memory_ctx->pagedirs_dma;

	for (i = 0; i < memory_ctx->num_pagedirs; i++) {
		pgdirn[i] = pgdir_dma >> PAGE_SHIFT;
		dev_dbg(&gna_priv->dev, "pagedir %#x mapped at %#lx",
					(u32)pgdir_dma, (uintptr_t)&pgdirn[i]);

		pgdir_dma += PAGE_SIZE;
	}

	// mark other page dirs as invalid
	for (; i < GNA_PGDIRN_LEN; i++)
		pgdirn[i] = GNA_PGDIR_INVALID;
}

static void gna_mmu_fill_l2(struct gna_private *gna_priv,
		struct gna_memory_context *memory_ctx)
{
	struct scatterlist *sgl;
	dma_addr_t sg_page;
	int sg_page_len;
	int pd_entries;
	u32 *pagedir;
	u32 mmu_page;
	int sg_pages;

	sgl = memory_ctx->sgt->sgl;
	sg_page = sg_dma_address(sgl);
	sg_page_len = ROUND_UP(sg_dma_len(sgl), PAGE_SIZE) >> PAGE_SHIFT;
	sg_pages = 0;

	pagedir = memory_ctx->pagedirs;
	for (pd_entries = 0; pd_entries < memory_ctx->num_pages; pd_entries++) {
		mmu_page = sg_page >> PAGE_SHIFT;
		pagedir[pd_entries] = mmu_page;

		dev_dbg(&gna_priv->dev, "mapped %u page at %#lx entry\n",
				mmu_page, (uintptr_t)(pagedir + pd_entries));

		sg_page += PAGE_SIZE;
		sg_pages++;
		if (sg_pages == sg_page_len) {
			sgl = sg_next(sgl);
			if (sgl == NULL) {
				dev_warn(&gna_priv->dev, "scatterlist out of entries\n");
				break;
			}

			sg_page = sg_dma_address(sgl);
			sg_page_len = ROUND_UP(sg_dma_len(sgl), PAGE_SIZE) >> PAGE_SHIFT;
			sg_pages = 0;
		}
	}
}

int gna_priv_map(struct gna_file_private *file_priv, struct gna_usrptr *data)
{
	struct gna_memory_context *memory_ctx;
	struct gna_private *gna_priv;
	int memory_id;
	int ret;

	ret = 0;

	mutex_lock(&file_priv->lock);

	gna_priv = file_priv->gna_priv;

	if (data->length <= 0 || data->length > gna_priv->info.max_hw_mem) {
		dev_err(&gna_priv->dev, "invalid user memory size\n");
		ret = -EINVAL;
		goto err_file_unlock;
	}

	if (!data->padd) {
		dev_err(&gna_priv->dev, "invalid user pointer\n");
		ret = -EINVAL;
		goto err_file_unlock;
	}

	memory_ctx = kzalloc(sizeof(*memory_ctx), GFP_KERNEL);
	if (!memory_ctx) {
		dev_err(&gna_priv->dev, "could not allocate memory context\n");
		ret = PTR_ERR(memory_ctx);
		goto err_file_unlock;
	}

	memory_id = idr_alloc(&file_priv->memory_idr, memory_ctx, 1, 0, GFP_KERNEL);
	if (memory_id < 0) {
		dev_err(&gna_priv->dev, "idr allocation for memory failed\n");
		goto err_memctx_release;
	}

	dev_dbg(&gna_priv->dev, "memory id allocated: %d", memory_id);

	gna_memctx_initialize(memory_ctx, memory_id, data);

	ret = gna_mmu_alloc(gna_priv, memory_ctx);
	if (ret) {
		dev_err(&gna_priv->dev, "hw descriptor alloc failed\n");
		goto err_idr_release;
	}

	ret = gna_map_pages(gna_priv, data, memory_ctx);
	if (ret) {
		dev_err(&gna_priv->dev, "mapping pages failed\n");
		goto err_mmu_release;
	}

	/* Populate the pts with the addresses */
	gna_mmu_fill_l1(gna_priv, memory_ctx);
	gna_mmu_fill_l2(gna_priv, memory_ctx);

	mutex_unlock(&file_priv->lock);

	data->memory_id = memory_id;

	return 0;

err_mmu_release:
	gna_mmu_free(file_priv, memory_ctx);

err_idr_release:
	idr_remove(&file_priv->memory_idr, memory_id);

err_memctx_release:
	kfree(memory_ctx);

err_file_unlock:
	mutex_unlock(&file_priv->lock);

	return ret;
}

int gna_priv_unmap(int memory_id, void *memory, void *data)
{
	struct gna_memory_context *memory_ctx;
	struct gna_file_private *file_priv;
	struct gna_private *gna_priv;

	file_priv = (struct gna_file_private *)data;
	gna_priv = file_priv->gna_priv;
	dev_dbg(&gna_priv->dev, "%s\n", __func__);

	memory_ctx = (struct gna_memory_context *)memory;
	gna_unmap_pages(gna_priv, memory_ctx);
	gna_mmu_free(file_priv, memory_ctx);
	kfree(memory_ctx);

	return 0;
}

