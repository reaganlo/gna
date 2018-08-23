// SPDX-License-Identifier: GPL-2.0
// Copyright(c) 2017-18 Intel Corporation

#include "gna_hw.h"
#include "gna_drv.h"

#include <uapi/misc/gna.h>

void gna_start_scoring(struct gna_private *gna_priv, void __iomem *addr,
			      struct gna_score_cfg *score_cfg)
{
	union gna_ctrl_reg ctrl;
	struct gna_ctrl_flags *flags = &score_cfg->flags;

	ctrl.val = gna_reg_read(addr, GNACTRL);

	ctrl.ctrl.start_accel    = 1;
	ctrl.ctrl.compl_int_en   = 1;
	ctrl.ctrl.err_int_en     = 1;
	ctrl.ctrl.comp_stats_en  = score_cfg->hw_perf_encoding & 0xF;
	ctrl.ctrl.active_list_en = flags->active_list_on;
	ctrl.ctrl.gna_mode       = flags->gna_mode;

	gna_reg_write(addr, GNACTRL, ctrl.val);

	add_timer(&gna_priv->isr_timer);

	dev_dbg(&gna_priv->dev, "scoring started...\n");
}

static void gna_clear_saturation(struct gna_private *gna_priv, void __iomem *addr)
{
	u32 val;

	val = gna_reg_read(addr, GNASTS);
	if (val & GNA_STS_SATURATE) {
		dev_dbg(&gna_priv->dev, "saturation reached\n");
		dev_dbg(&gna_priv->dev, "gna status: %#x\n", val);

		val = val & GNA_STS_SATURATE;
		gna_reg_write(addr, GNASTS, val);

		val = gna_reg_read(addr, GNASTS);
		dev_dbg(&gna_priv->dev, "gna modified status: %#x\n", val);
	}
}

void gna_debug_isi(struct gna_private *gna_priv, void __iomem *addr)
{
	u32 isv_lo, isv_hi;

	gna_reg_write(addr, GNAISI, 0x80);
	isv_lo = gna_reg_read(addr, GNAPISV);
	isv_hi = gna_reg_read(addr, GNAPISV + sizeof(__u32));

	dev_dbg(&gna_priv->dev, "labase: %#x\n", isv_lo);
	dev_dbg(&gna_priv->dev, "lacnt: %#x\n", isv_hi);

	gna_reg_write(addr, GNAISI, 0x82);
	isv_lo = gna_reg_read(addr, GNAPISV);
	isv_hi = gna_reg_read(addr, GNAPISV + sizeof(__u32));

	dev_dbg(&gna_priv->dev, "{n_inputs,nnFlags,nnop}: %#x\n", isv_lo);
	dev_dbg(&gna_priv->dev,
	"{inputIteration/nInputConvStride,n_groups,n_outputs}: %#x\n", isv_hi);

	gna_reg_write(addr, GNAISI, 0x83);
	isv_lo = gna_reg_read(addr, GNAPISV);
	isv_hi = gna_reg_read(addr, GNAPISV + sizeof(__u32));

	dev_dbg(&gna_priv->dev,
	"{res,outFbIter,inputInLastIter/nConvFilterSize}: %#x\n", isv_lo);
	dev_dbg(&gna_priv->dev,
	"{outFbInLastIter/poolStride,outFbInFirstIter/nConvFilters: %#x\n", isv_hi);

	gna_reg_write(addr, GNAISI, 0x84);
	isv_lo = gna_reg_read(addr, GNAPISV);
	isv_hi = gna_reg_read(addr, GNAPISV + sizeof(__u32));

	dev_dbg(&gna_priv->dev,
	"{nActListElems/nCopyElems,res,nActSegs/poolSize}: %#x\n", isv_lo);
	dev_dbg(&gna_priv->dev, "reserved: %#x\n", isv_hi);

	gna_reg_write(addr, GNAISI, 0x86);
	isv_lo = gna_reg_read(addr, GNAPISV);
	isv_hi = gna_reg_read(addr, GNAPISV + sizeof(__u32));

	dev_dbg(&gna_priv->dev, "in_buffer: %#x\n", isv_lo);
	dev_dbg(&gna_priv->dev, "out_act_fn_buffer: %#x\n", isv_hi);

	gna_reg_write(addr, GNAISI, 0x87);
	isv_lo = gna_reg_read(addr, GNAPISV);
	isv_hi = gna_reg_read(addr, GNAPISV + sizeof(__u32));

	dev_dbg(&gna_priv->dev, "out_sum_buffer: %#x\n", isv_lo);
	dev_dbg(&gna_priv->dev, "out_fb_buffer: %#x\n", isv_hi);

	gna_reg_write(addr, GNAISI, 0x88);
	isv_lo = gna_reg_read(addr, GNAPISV);
	isv_hi = gna_reg_read(addr, GNAPISV + sizeof(__u32));

	dev_dbg(&gna_priv->dev, "weight/filter buffer: %#x\n", isv_lo);
	dev_dbg(&gna_priv->dev, "bias buffer: %#x\n", isv_hi);

	gna_reg_write(addr, GNAISI, 0x89);
	isv_lo = gna_reg_read(addr, GNAPISV);
	isv_hi = gna_reg_read(addr, GNAPISV + sizeof(__u32));

	dev_dbg(&gna_priv->dev, "indices buffer: %#x\n", isv_lo);
	dev_dbg(&gna_priv->dev, "pwl segments buffer: %#x\n", isv_hi);
}

void gna_abort_hw(struct gna_private *gna_priv, void __iomem *addr)
{
	u32 val;
	int i = 0;

	/* saturation bit in the GNA status register needs
	 * to be expicitly cleared
	 */
	gna_clear_saturation(gna_priv, addr);

	val = gna_reg_read(addr, GNASTS);
	dev_dbg(&gna_priv->dev, "status before abort: %#x\n", val);

	val = gna_reg_read(addr, GNACTRL);

	val |= (1 << GNA_CTRL_ABORT_SHIFT); /* Mask with Abort Mask */

	gna_reg_write(addr, GNACTRL, val);

	while (i < 100) {
		val = gna_reg_read(addr, GNASTS);
		dev_dbg(&gna_priv->dev, "status after abort: %#x\n", val);
		if ((val & 0x1) == 0) {
			break;
		}
		i++;
	}

	if (i == 100)
		dev_err(&gna_priv->dev, "abort did not complete\n");
}

