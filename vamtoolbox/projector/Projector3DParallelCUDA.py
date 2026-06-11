import astra

try:
    import tigre
except:
    ImportError("Tigre toolbox is either not installed or installed incorrectly.")

import numpy as np

import vamtoolbox


class Projector3DParallelCUDAAstraChunked:
    """astra CUDA parallel-3D projector that streams the volume to the GPU in
    blocks of z-slices, so peak VRAM is bounded by the chunk size instead of the
    full N^3 volume.  Parallel-beam geometry is z-separable (each z-slice -> one
    detector row), so chunked == non-chunked numerically.

    chunk size is auto-sized to VRAM_BUDGET_BYTES.  When the whole volume fits in
    one chunk this is identical to Projector3DParallelCUDAAstra (no overhead).
    Only the non-inclined parallel3d case is chunked.
    """

    VRAM_BUDGET_BYTES = 4_000_000_000  # ~4 GB working set per chunk on the GPU

    def __init__(self, target_geo, proj_geo):
        self.target_geo = target_geo
        self.proj_geo = proj_geo
        self.nX, self.nY, self.nZ = target_geo.nX, target_geo.nY, target_geo.nZ
        self.nT = target_geo.nX
        self.nA = proj_geo.angles.size
        self.angles_rad = np.deg2rad(proj_geo.angles)
        if self.proj_geo.absorption_coeff is not None:
            # store in astra order (nZ, nY, nX) to match the transposed volume
            self.proj_geo.absorption_mask = np.transpose(self.proj_geo.absorption_mask)

        # per-z-slice GPU cost: volume slice + its sinogram row + ~2x texture
        per_slice = self.nX * self.nY * 4 * 3 + self.nA * self.nX * 4
        self.z_chunk = int(max(1, min(self.nZ, self.VRAM_BUDGET_BYTES // per_slice)))

    def _geoms(self, n_rows):
        vol_geom = astra.create_vol_geom(self.nX, self.nY, n_rows)
        proj_geom = astra.create_proj_geom(
            "parallel3d", 1.0, 1.0, n_rows, self.nT, self.angles_rad
        )
        return vol_geom, proj_geom

    def forward(self, target):
        """b = Ax, streamed over z-chunks."""
        x = np.transpose(vamtoolbox.util.data.clipToCircle(target))   # (nZ, nY, nX)
        if self.proj_geo.absorption_coeff is not None:
            x = self.proj_geo.absorption_mask * x
        b = np.empty((self.nZ, self.nA, self.nX), dtype=np.float32)
        for z0 in range(0, self.nZ, self.z_chunk):
            z1 = min(z0 + self.z_chunk, self.nZ)
            vol_geom, proj_geom = self._geoms(z1 - z0)
            xc = np.ascontiguousarray(x[z0:z1], dtype=np.float32)
            b_id, tmp_b = astra.create_sino3d_gpu(xc, proj_geom, vol_geom)
            b[z0:z1] = tmp_b
            astra.data3d.delete(b_id)
        return np.transpose(b, (2, 1, 0))                              # (nX, nA, nZ)

    def backward(self, sinogram):
        """x = A^T b, streamed over z-chunks."""
        b = sinogram
        if self.proj_geo.zero_dose_sino is not None:
            b[self.proj_geo.zero_dose_sino] = 0.0
        tmp = np.transpose(b, (2, 1, 0))                               # (nZ, nA, nX)
        x = np.empty((self.nZ, self.nY, self.nX), dtype=np.float32)
        for z0 in range(0, self.nZ, self.z_chunk):
            z1 = min(z0 + self.z_chunk, self.nZ)
            vol_geom, proj_geom = self._geoms(z1 - z0)
            tc = np.ascontiguousarray(tmp[z0:z1], dtype=np.float32)
            x_id, xc = astra.creators.create_backprojection3d_gpu(tc, proj_geom, vol_geom)
            x[z0:z1] = xc
            astra.data3d.delete(x_id)
        if self.proj_geo.absorption_coeff is not None:
            x = self.proj_geo.absorption_mask * x
        x = np.transpose(x)                                            # (nX, nY, nZ)
        return vamtoolbox.util.data.clipToCircle(x)


class Projector3DParallelCUDAAstra:
    def __init__(self, target_geo, proj_geo):
        self.target_geo = target_geo
        self.proj_geo = proj_geo
        self.nT = target_geo.nX
        self.angles_rad = np.deg2rad(proj_geo.angles)

        if self.proj_geo.absorption_coeff is not None:
            self.proj_geo.absorption_mask = np.transpose(self.proj_geo.absorption_mask)

        self.vol_geom = astra.create_vol_geom(
            target_geo.nX, target_geo.nY, target_geo.nZ
        )

        if proj_geo.inclination_angle is None or proj_geo.inclination_angle == 0:
            self.proj_geom = astra.create_proj_geom(
                "parallel3d", 1.0, 1.0, target_geo.nZ, self.nT, self.angles_rad
            )

        else:
            self.angles_vector = vamtoolbox.projector.genVectorsAstra.genVectorsAstra(
                proj_geo.angles, proj_geo.inclination_angle
            )
            self.proj_geom = astra.create_proj_geom(
                "parallel3d_vec", target_geo.nZ, self.nT, self.angles_vector
            )

    def forward(self, target):
        """Forward projector operation (b = Ax)"""
        x = np.transpose(vamtoolbox.util.data.clipToCircle(target))
        if self.proj_geo.absorption_coeff is not None:
            x = self.proj_geo.absorption_mask * x

        b_id, tmp_b = astra.create_sino3d_gpu(x, self.proj_geom, self.vol_geom)

        b = np.transpose(tmp_b, (2, 1, 0))

        astra.data3d.delete(b_id)

        return b

    def backward(self, sinogram):
        """Backward projector operation (x = A^Tb)"""
        b = sinogram
        if self.proj_geo.zero_dose_sino is not None:
            b[self.proj_geo.zero_dose_sino] = 0.0

        tmp_b = np.transpose(b, (2, 1, 0))

        x_id, x = astra.creators.create_backprojection3d_gpu(
            tmp_b, self.proj_geom, self.vol_geom
        )

        if self.proj_geo.absorption_coeff is not None:
            x = self.proj_geo.absorption_mask * x
        x = np.transpose(x)

        astra.data3d.delete(x_id)

        return vamtoolbox.util.data.clipToCircle(x)


class Projector3DParallelCUDATigre:
    def __init__(self, target_geo, proj_geo, optical_params=None):

        self.angles_rad = np.deg2rad(proj_geo.angles)

        try:
            self.attenuation = np.swapaxes(
                proj_geo.attenuation.astype(np.float32), 2, 0
            )
            self.attenuation = np.ascontiguousarray(self.attenuation)
        except:
            self.attenuation = None

        # setup fixed coordinate grid for backprojection and dimensions of projections
        self.radius = target_geo.nY // 2
        self.y, self.x = np.mgrid[: target_geo.nY, : target_geo.nY] - self.radius
        self.center = target_geo.nY // 2
        self.proj_t = np.arange(target_geo.nY) - target_geo.nY // 2

        self.geo = tigre.geometry(
            mode="parallel",
            nVoxel=np.array([target_geo.nZ, target_geo.nY, target_geo.nX]),
        )
        self.geo.dDetector = np.array([1, 1])  # size of each pixel            (mm)
        self.geo.sDetector = self.geo.dDetector * self.geo.nDetector
        self.geo.accuracy = 1
        self.geo.vialRadius = 1
        self.geo.maxIntensity = 1

    def forward(self, target):

        x = vamtoolbox.util.data.clipToCircle(target.astype(np.float32))
        x = np.swapaxes(x, 2, 0)
        x = np.ascontiguousarray(x)

        b = tigre.Ax(
            x,
            self.geo,
            self.angles_rad,
            projection_type="interpolated",
            img_att=self.attenuation,
        )
        b = np.transpose(b, (2, 0, 1))

        return b

    def backward(self, projections):
        b = projections.astype(np.float32)

        b = np.ascontiguousarray(np.transpose(b, (1, 2, 0)))
        if self.attenuation is not None:
            tmp_attenuation = np.ascontiguousarray(np.swapaxes(self.attenuation, 1, 2))
            x = tigre.Atb(
                projections, self.geo, self.angles_rad, img_att=tmp_attenuation
            )
        else:
            x = tigre.Atb(projections, self.geo, self.angles_rad)

        # print(tmp_attenuation.shape)

        x = np.swapaxes(x, 0, 2)

        return vamtoolbox.util.data.clipToCircle(x)
