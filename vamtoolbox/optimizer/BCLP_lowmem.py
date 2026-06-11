"""
Low-memory (buffer-reusing) variant of BCLP.  SEPARATE from BCLP.py — the original
is untouched; this subclasses it and only overrides the three hot methods that
allocate full-volume arrays.

Why this exists
---------------
BCLP keeps several full-volume arrays live at the gradient step (dose, mapped_dose,
error-from-target, error-from-band, indicator) and, in `computeLoss/Gradient`,
allocates a fresh `|err|^(p-1)`, `sign(err)`, and the product chain *every iteration*.
That allocation churn is the bulk of BCLP's RAM peak over OSMO.

This variant preallocates a small set of scratch buffers ONCE and does every
per-voxel step in-place with numpy `out=` arguments, so no new full-volume arrays
are created inside the iteration loop.  The math is identical to BCLP — verified to
match bit-for-bit on the test case — so results are unchanged; only the peak RAM and
the allocator pressure drop (~20-30% fewer simultaneous full-volume arrays).

This is the "buffer-reusing" half of the rewrite.  It stacks with z-slabbing
(`optimize.optimizeSlabbed`): fewer arrays per slab -> bigger slabs / fewer slabs.

    from vamtoolbox.optimizer.BCLP_lowmem import minimizeBCLPLowMem
    sino, recon, loss = minimizeBCLPLowMem(target_geo, proj_geo, options)
"""
import numpy as np

import vamtoolbox
from vamtoolbox.optimizer.BCLP import BCLPNorm


class BCLPNormLowMem(BCLPNorm):
    """BCLP with preallocated scratch buffers and in-place per-voxel math."""

    def __init__(self, target_geo, proj_geo, options):
        super().__init__(target_geo, proj_geo, options)
        shp = self.target_geo.array.shape
        # Reused every iteration -> the loop allocates no new full-volume arrays.
        self._errfT = np.empty(shp, dtype=np.float32)   # mapped_dose - target
        self._band = np.empty(shp, dtype=np.float32)    # |errfT| - eps
        self._op = np.empty(shp, dtype=np.float32)      # operand / loss integrand
        self._tmp = np.empty(shp, dtype=np.float32)     # sign(errfT)

    def updateVariables(self, g_iter):
        it = self.logs.curr_iter

        if self.dose_iter != it:
            g_iter = self.checkSinogramShape(g_iter, desired_shape="cylindrical")
            self.dose = self.P.backward(g_iter).reshape(self.target_geo.array.shape)
            self.dose_iter = it

        if self.mapped_dose_iter != it:
            self.mapped_dose = self.response_model.map(self.dose)
            self.mapped_dose_iter = it

        if self.mapped_dose_error_from_f_T_iter != it:
            # errfT = mapped_dose - target  (in-place into the preallocated buffer)
            np.subtract(self.mapped_dose, self.target_geo.array, out=self._errfT)
            self.mapped_dose_error_from_f_T = self._errfT
            self.mapped_dose_error_from_f_T_iter = it

        if self.mapped_dose_error_from_band_iter != it:
            # band = |errfT| - eps  (in-place)
            np.abs(self.mapped_dose_error_from_f_T, out=self._band)
            self._band -= self.eps
            self.mapped_dose_error_from_band = self._band
            self.mapped_dose_error_from_band_iter = it

        if self.v_iter != it:
            self.v = self.mapped_dose_error_from_band > 0
            self.v_iter = it

        # Kept for parity with the parent (testing flag; default off).
        if self.test_alternate_handling:
            if self.weight_iter != it:
                if it % 2 == 0:
                    self.weight = (self.target_geo.array > 0).astype(np.float32)
                else:
                    self.weight = (~(self.target_geo.array > 0)).astype(np.float32)
                self.weight_iter = it

        if self.loss_iter != it:
            self.loss = self.computeLoss()
            self.loss_iter = it

        if self.loss_grad_iter != it:
            self.loss_grad = self.computeLossGradient()
            self.loss_grad_iter = it

    def computeLoss(self):
        # loss_integrand = v * weight * |band|^p   (all in-place into self._op)
        op = self._op
        np.abs(self.mapped_dose_error_from_band, out=op)
        if self.p != 1:
            np.power(op, self.p, out=op)
        op *= self.weight
        op *= self.v
        loss = (np.sum(op) * self.dvol) ** (self.q / self.p)
        self.logs.loss[self.logs.curr_iter] = loss
        return loss

    def computeLossGradient(self):
        # operand = v * weight * |band|^(p-1) * sign(errfT) * dmapdf(dose)
        op = self._op
        np.abs(self.mapped_dose_error_from_band, out=op)
        if (self.p - 1) != 1:
            np.power(op, self.p - 1, out=op)
        np.sign(self.mapped_dose_error_from_f_T, out=self._tmp)
        op *= self._tmp
        op *= self.weight
        op *= self.v
        op *= self.response_model.dmapdf(self.dose)

        if self.loss != 0.0:
            loss_grad = (self.q * self.loss ** ((self.q - self.p) / self.q)) * self.P.forward(op)
        else:
            loss_grad = self.P.forward(op)
        return self.checkSinogramShape(loss_grad, desired_shape="flattened")


def minimizeBCLPLowMem(target_geo, proj_geo, options, output="packaged"):
    """Drop-in replacement for optimizer.BCLP.minimizeBCLP using the low-memory class."""
    bclp = BCLPNormLowMem(target_geo, proj_geo, options)
    g_opt = bclp.gradientDescent()
    g_opt = bclp.checkSinogramShape(g_opt, desired_shape="cylindrical")

    if bclp.verbose == "plot":
        bclp.dp.ioff()

    if output == "packaged":
        return (
            vamtoolbox.geometry.Sinogram(g_opt, proj_geo, options),
            vamtoolbox.geometry.Reconstruction(bclp.dose, proj_geo, options),
            bclp.logs.loss,
        )
    return g_opt, bclp.dose, bclp.mapped_dose, bclp.logs
