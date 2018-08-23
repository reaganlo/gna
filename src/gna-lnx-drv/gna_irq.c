// SPDX-License-Identifier: GPL-2.0
// Copyright(c) 2017-18 Intel Corporation

#include "gna_irq.h"
#include "gna_drv.h"

#include <uapi/misc/gna.h>

irqreturn_t gna_interrupt(int irq, void *priv)
{
	struct gna_private *gna_priv;
	void __iomem *addr;
	irqreturn_t ret;
	u32 hw_status;

 	gna_priv = (struct gna_private *)priv;
	dev_dbg(&gna_priv->dev, "%s: enter\n", __func__);

	addr = gna_priv->bar0.mem_addr;
	ret = IRQ_HANDLED;

	spin_lock(&gna_priv->irq_lock);

	hw_status = gna_reg_read(addr, GNASTS);

	dev_dbg(&gna_priv->dev, "received interrupt, device status: 0x%x\n", hw_status);

	/* check if interrupt originated from gna device */
	if (hw_status & GNA_INTERRUPT) {
		dev_dbg(&gna_priv->dev, "interrupt originated from gna device\n");

		if (gna_priv->current_request) {
			gna_priv->current_request->hw_status = hw_status;
			ret = IRQ_WAKE_THREAD;
		} else {
			dev_err(&gna_priv->dev, "no request for an interrupt\n");
			ret = IRQ_HANDLED;
		}
	}

	spin_unlock(&gna_priv->irq_lock);

	dev_dbg(&gna_priv->dev, "%s: exit\n", __func__);

	return ret;
}

irqreturn_t gna_irq_thread(int irq, void *priv)
{
	struct gna_private *gna_priv;

 	gna_priv = (struct gna_private *)priv;
	dev_dbg(&gna_priv->dev, "%s: enter\n", __func__);

	// schedule gna request tasklet
	atomic_inc(&gna_priv->isr_count);
	tasklet_schedule(&gna_priv->request_tasklet);

	dev_dbg(&gna_priv->dev, "%s: exit\n", __func__);

	return IRQ_HANDLED;
}

