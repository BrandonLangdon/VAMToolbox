from functools import partial

try:
    import astra  # type: ignore
except ImportError:
    astra = None
import numpy as np
from skimage.transform import radon, iradon, warp
try:
    from joblib import Parallel, delayed as jdelayed, effective_n_jobs
    _N_JOBS = -1   # use all logical CPUs
except ImportError:
    Parallel = None


def _radon_chunk(x_chunk, angles):
    """Worker: radon-transform a contiguous block of z-slices.
    x_chunk: (k, nX, nY) -> returns (k, nX, n_angles).

    loky memmaps chunks above ~1 MB as READ-ONLY; skimage's Cython radon needs a
    writable buffer, so copy the chunk to writable once (cheap memcpy per chunk)."""
    x_chunk = np.array(x_chunk)  # writable copy in the worker
    return np.stack(
        [radon(x_chunk[i], theta=angles, circle=True) for i in range(x_chunk.shape[0])],
        axis=0,
    )


def _iradon_chunk(b_chunk, angles):
    """Worker: unfiltered back-project a contiguous block of z-slices.
    b_chunk: (k, nX, n_angles) -> returns (k, nX, nY)."""
    b_chunk = np.array(b_chunk)  # writable copy in the worker (see _radon_chunk)
    return np.stack(
        [iradon(b_chunk[i], theta=angles, filter_name=None, circle=True)
         for i in range(b_chunk.shape[0])],
        axis=0,
    )


import vamtoolbox


class Projector3DParallelAstra:
    def __init__(self, target_geo, proj_geo):
        self.target_geo = target_geo
        self.proj_geo = proj_geo
        self.nT = target_geo.nX

        self.angles_rad = np.deg2rad(proj_geo.angles)

        self.proj_geom = astra.create_proj_geom(
            "parallel", 1.0, self.nT, self.angles_rad
        )
        self.vol_geom = astra.create_vol_geom(target_geo.nY, target_geo.nX)
        self.proj_id = astra.create_projector("line", self.proj_geom, self.vol_geom)

    def forward(self, x):
        """Forward projector operation (b = Ax)"""
        x = vamtoolbox.util.data.clipToCircle(x)
        b = np.zeros((self.nT, self.proj_geo.n_angles, self.target_geo.nZ))
        if self.proj_geo.absorption_coeff is not None:
            x = self.proj_geo.absorption_mask * x

        for z_i in range(self.target_geo.nZ):
            b_id, tmp_b = astra.create_sino(x[:, :, z_i], self.proj_id)
            b[:, :, z_i] = np.transpose(tmp_b)
            astra.data3d.delete(b_id)

        return b

    def backward(self, b):
        """Backward projector operation (x = A^Tb)"""

        x = np.zeros((self.target_geo.nX, self.target_geo.nY, self.target_geo.nZ))
        for z_i in range(self.target_geo.nZ):
            if self.proj_geo.zero_dose_sino is not None:
                b[self.proj_geo.zero_dose_sino] = 0.0
            x_id, tmp_x = astra.creators.create_backprojection(
                np.transpose(b[:, :, z_i]), self.proj_id
            )
            x[:, :, z_i] = tmp_x
            astra.data3d.delete(x_id)

        if self.proj_geo.absorption_coeff is not None:
            x = self.proj_geo.absorption_mask * x

        return vamtoolbox.util.data.clipToCircle(x)


class Projector3DParallelPython:
    def __init__(self, target_geo, proj_geo):
        self.target_geo = target_geo
        self.proj_geo = proj_geo
        self.angles = proj_geo.angles

        # setup fixed coordinate grid for backprojection and dimensions of projections
        self.radius = target_geo.nY // 2
        self.y, self.x = np.mgrid[: target_geo.nY, : target_geo.nY] - self.radius
        self.center = target_geo.nY // 2
        self.proj_t = np.arange(target_geo.nY) - target_geo.nY // 2

        # Occlusion sinogram is computed in init because it will remain the same for a
        # given instance of the class; all forward/backprojections with the instance
        # assume that the occlusion does not change
        if self.proj_geo.attenuation_field is not None:
            # TODO make independent of infinite value of insert, e.g. attenuated forward and backward projection
            insert = np.where(self.proj_geo.attenuation_field > 0, 1, 0).astype(int)
            self.occ_sinogram = self.generateOccSinogram(insert)

    def generateOccSinogram(self, occ_array):
        """
        Create sinogram containing minimum values of 's' within the occlusion map

        Returns
        ---------------
        occ_sinogram : nd_array
        Npixels x Nangles x Nslices

        """

        occ_sinogram = np.zeros(
            (self.target_geo.nY, self.angles.shape[0], self.target_geo.nZ)
        )

        for z_i in range(self.target_geo.nZ):
            for i, angle in enumerate(np.deg2rad(self.angles)):
                cos_a, sin_a = np.cos(angle), np.sin(angle)

                R = np.array(
                    [
                        [cos_a, sin_a, -self.center * (cos_a + sin_a - 1)],
                        [-sin_a, cos_a, -self.center * (cos_a - sin_a - 1)],
                        [0, 0, 1],
                    ]
                )

                rotated_occlusion = warp(occ_array[:, :, z_i], R, clip=True)
                s_occ = np.where(rotated_occlusion > 0, self.y, np.nan)

                # disp.view_plot(s_occ,'S')

                occ_sinogram[:, i, z_i] = np.nanmin(s_occ, axis=0)

        return occ_sinogram

    def forward(self, target):
        """
        Computes forward Radon transform of the target space object accounting for
        reduced projection contribution due to occlusion shadowing

        Inputs
        ---------------
        target : nd_array
        Npixels x Npixels x Npixels array that contains the target space object

        Returns
        ---------------
        projection : nd_array
        Npixels x Nangles x Nslices array of forward Radon transform with occlusion shadowing

        """
        projection = np.zeros(
            (self.target_geo.nY, self.angles.shape[0], self.target_geo.nZ)
        )
        for z_i in range(self.target_geo.nZ):
            for i, angle in enumerate(np.deg2rad(self.angles)):

                cos_a, sin_a = np.cos(angle), np.sin(angle)

                R = np.array(
                    [
                        [cos_a, sin_a, -self.center * (cos_a + sin_a - 1)],
                        [-sin_a, cos_a, -self.center * (cos_a - sin_a - 1)],
                        [0, 0, 1],
                    ]
                )

                rotated = warp(target[:, :, z_i], R, clip=True)

                if self.proj_geo.attenuation_field is not None:
                    curr_occ = self.occ_sinogram[:, i, z_i]
                    if np.count_nonzero(curr_occ) - np.sum(np.isnan(curr_occ)) != 0:

                        occ_shadow = self.y > curr_occ[np.newaxis, :]

                        rotated = np.multiply(rotated, np.logical_not(occ_shadow))

                projection[:, i, z_i] = rotated.sum(0)
        return projection

    def backward(self, projection):
        """
        Computes inverse Radon transform of projection accounting for reduced dose
        deposition due to occlusion shadowing

        Inputs
        ---------------
        projection : nd_array
        Npixels x Nangles x Nslices array that contains the projection space sinogram of the target

        Returns
        ---------------
        reconstruction : nd_array
        Npixels x Npixels x Npixels array of inverse Radon transform with occlusion shadowing

        """

        reconstruction = np.zeros(self.target_geo.array.shape)
        for z_i in range(self.target_geo.nZ):
            reconstructed = np.zeros((reconstruction.shape[0], reconstruction.shape[1]))
            for i, (curr_proj, angle) in enumerate(
                zip(projection[:, :, z_i].T, np.deg2rad(self.angles))
            ):

                cos_a, sin_a = np.cos(angle), np.sin(angle)

                t = self.x * cos_a - self.y * sin_a
                s = self.x * sin_a + self.y * cos_a

                interpolant = partial(
                    np.interp, xp=self.proj_t, fp=curr_proj, left=0, right=0
                )
                curr_backproj = interpolant(t)

                if self.proj_geo.attenuation_field is not None:
                    curr_occ = self.getOccShadow(i, z_i, angle, t, s)
                    if np.count_nonzero(curr_occ) - np.sum(np.isnan(curr_occ)) != 0:
                        reconstructed += np.multiply(
                            curr_backproj, np.logical_not(curr_occ)
                        )
                    else:
                        reconstructed += curr_backproj
                else:
                    reconstructed += curr_backproj

                # plt.imshow(np.multiply(curr_backproj,np.logical_not(curr_occ)),cmap='CMRmap')
                # plt.show()
            reconstruction[:, :, z_i] = reconstructed

        return vamtoolbox.util.data.clipToCircle(reconstruction)

    def getOccShadow(self, i, j, angle, t, s):

        curr_occ = self.occ_sinogram[:, :, j]
        interpolant = partial(
            np.interp, xp=self.proj_t, fp=curr_occ[:, i], left=np.nan, right=np.nan
        )

        return s > np.floor(interpolant(t))

    # def calcVisibility(self):
    #     tmp = np.zeros((self.target_obj.nY,self.target_obj.nX,self.angles.shape[0]))
    #     vis = np.zeros(self.target_obj.target.shape)
    #     projection = np.ones((self.target_obj.nY,self.angles.shape[0]))

    #     for i, (curr_proj, angle) in enumerate(zip(projection.T, np.deg2rad(self.angles))):

    #         cos_a, sin_a = np.cos(angle), np.sin(angle)

    #         t = self.x * cos_a - self.y * sin_a
    #         s = self.x * sin_a + self.y * cos_a

    #         interpolant = partial(np.interp, xp=self.proj_t, fp=curr_proj*self.angles[i], left=0, right=0)
    #         curr_backproj = interpolant(t)

    #         curr_occ = self.getOccShadow(i,angle,t,s)

    #         tmp[..., i] = np.multiply(curr_backproj,np.logical_not(curr_occ))

    #     for k in range(self.target_obj.nY):
    #         for j in range(self.target_obj.nX):
    #             q = np.unique(tmp[k,j,:]%(self.angles.shape[0]//2))

    #             vis[k,j] = q.shape[0]

    #     vis = np.multiply(vis,self.target_obj.target)
    #     vis = vis/(self.angles.shape[0]//2)
    #     vis = np.where(vis >= 1, 1, vis)


class Projector3DParallelSkimage:
    """CPU parallel-beam 3D projector using scikit-image radon/iradon.
    Drop-in replacement for Projector3DParallelAstra when astra is unavailable.
    """

    def __init__(self, target_geo, proj_geo):
        self.target_geo = target_geo
        self.proj_geo = proj_geo
        self.angles_deg = proj_geo.angles  # degrees

    @staticmethod
    def _n_chunks(nZ):
        """Group z-slices into ~2x the worker count: cuts loky dispatch/pickle
        count from nZ to a few dozen while keeping good load balance."""
        if Parallel is None:
            return nZ
        n_workers = max(1, effective_n_jobs(_N_JOBS))
        return int(min(nZ, n_workers * 2))

    def forward(self, target):
        """Forward projector: b = Ax (Radon transform per z-slice, parallelised over z).

        Lever 4: transpose to z-first ONCE so each slice is a contiguous view
        (no per-slice np.ascontiguousarray copy), and dispatch contiguous CHUNKS
        of slices to workers instead of 318 individual tasks.
        """
        x = vamtoolbox.util.data.clipToCircle(target)
        # Apply Beer-Lambert absorption mask (matches Projector3DParallelCUDAAstra).
        # The mask is rotationally symmetric so no transpose is needed here.
        if self.proj_geo.absorption_coeff is not None:
            x = self.proj_geo.absorption_mask * x
        nZ = self.target_geo.nZ
        angles = self.angles_deg

        if Parallel is not None and nZ > 1:
            # one contiguous z-first copy: (nX, nY, nZ) -> (nZ, nX, nY)
            xz = np.ascontiguousarray(np.moveaxis(x, 2, 0))
            chunks = np.array_split(xz, self._n_chunks(nZ), axis=0)
            results = Parallel(n_jobs=_N_JOBS)(
                jdelayed(_radon_chunk)(c, angles) for c in chunks
            )
            out = np.concatenate(results, axis=0)   # (nZ, nX, n_angles)
            return np.moveaxis(out, 0, 2)           # (nX, n_angles, nZ)

        # fallback: serial
        n_angles = self.proj_geo.n_angles
        b = np.zeros((self.target_geo.nX, n_angles, nZ), dtype=np.float64)
        for z_i in range(nZ):
            b[:, :, z_i] = radon(x[:, :, z_i], theta=angles, circle=True)
        return b

    def backward(self, b):
        """Backward projector: x = A^T b (unfiltered backprojection per z-slice).

        Lever 4: same z-first contiguous chunking as forward().
        """
        nZ = self.target_geo.nZ
        if self.proj_geo.zero_dose_sino is not None:
            b[self.proj_geo.zero_dose_sino] = 0.0
        angles = self.angles_deg

        if Parallel is not None and nZ > 1:
            bz = np.ascontiguousarray(np.moveaxis(b, 2, 0))   # (nZ, nX, n_angles)
            chunks = np.array_split(bz, self._n_chunks(nZ), axis=0)
            results = Parallel(n_jobs=_N_JOBS)(
                jdelayed(_iradon_chunk)(c, angles) for c in chunks
            )
            out = np.concatenate(results, axis=0)   # (nZ, nX, nY)
            x = np.moveaxis(out, 0, 2)              # (nX, nY, nZ)
        else:
            x = np.zeros((self.target_geo.nX, self.target_geo.nY, nZ), dtype=np.float64)
            for z_i in range(nZ):
                x[:, :, z_i] = iradon(b[:, :, z_i], theta=angles, filter_name=None, circle=True)
        # Apply Beer-Lambert absorption mask on the back-projection (matches CUDA projector).
        if self.proj_geo.absorption_coeff is not None:
            x = self.proj_geo.absorption_mask * x
        return vamtoolbox.util.data.clipToCircle(x)

    #     return vamtoolbox.util.data.clipToCircle(vis)


def _spmm_worker(data, indices, indptr, shape, Xc):
    """Worker: reconstruct a CSR view from memmapped component arrays (read-only,
    shared across workers) and multiply by a column block of the dense operand."""
    import scipy.sparse as sp
    M = sp.csr_matrix((data, indices, indptr), shape=shape, copy=False)
    return M @ Xc


class Projector3DParallelSparse:
    """CPU parallel-beam 3D projector using a precomputed 2D sparse system matrix
    (lever 6).  All z-slices share the same 2D parallel-beam geometry, so we build
    ONE matrix A (flattened nX*nY slice -> flattened n_angles*nX sinogram) once and
    apply it to every slice and every iteration as a sparse matmul, parallelised by
    splitting the z-columns across workers (the matrix is shared read-only).

    backward() uses A.T, the EXACT adjoint of forward() (unlike radon/iradon).

    Geometry note: the matrix comes from astra's 'line' parallel projector, whose
    angle/orientation convention differs from skimage radon.  forward/backward are a
    consistent transpose pair so OSMO converges correctly, but if the optimized
    sinogram is fed to the physical projector the angle convention should be
    calibrated (or rebinFanBeam adjusted) the same way it is for the astra projectors.
    """

    _cache = {}  # (nX, nY, nA, a0, a1) -> (A, AT)

    # Calibration so this matches the astra-CUDA-3D (GPU production) convention.
    # Determined empirically (_calibrate_sparse.py): astra's 2D 'line' parallel
    # geometry matches parallel3d at angles+90deg with the detector axis flipped.
    ANGLE_OFFSET_DEG = 90.0
    DET_FLIP = True

    def __init__(self, target_geo, proj_geo):
        import scipy.sparse as sp
        import astra
        self.target_geo = target_geo
        self.proj_geo = proj_geo
        nX, nY, nZ = target_geo.nX, target_geo.nY, target_geo.nZ
        self.nX, self.nY, self.nZ = nX, nY, nZ
        angles_rad = np.deg2rad(proj_geo.angles + self.ANGLE_OFFSET_DEG)
        self.nA = angles_rad.size
        self.n_jobs = effective_n_jobs(_N_JOBS) if Parallel is not None else 1

        key = (nX, nY, self.nA, float(angles_rad[0]), float(angles_rad[-1]),
               self.DET_FLIP)
        if key in Projector3DParallelSparse._cache:
            self.A, self.AT = Projector3DParallelSparse._cache[key]
        else:
            vol_geom = astra.create_vol_geom(nY, nX)
            proj_geom = astra.create_proj_geom("parallel", 1.0, nX, angles_rad)
            proj_id = astra.create_projector("line", proj_geom, vol_geom)
            # float32: halves matrix memory and keeps the sparse matmul output
            # float32 (avoids upcasting the whole volume to float64 in OSMO).
            A = sp.csr_matrix(astra.matrix.get(astra.projector.matrix(proj_id))).astype(np.float32)
            astra.projector.delete(proj_id)
            if self.DET_FLIP:
                # rows are angle-major (a*nX + d); flip detector index d -> nX-1-d
                perm = (np.arange(self.nA)[:, None] * nX
                        + (nX - 1 - np.arange(nX))[None, :]).ravel()
                A = sp.csr_matrix(A[perm, :])
            AT = sp.csr_matrix(A.T)
            self.A, self.AT = A, AT
            Projector3DParallelSparse._cache[key] = (A, AT)

    def _spmm(self, M, X):
        """Parallel sparse @ dense over column blocks of X (matrix shared read-only)."""
        if Parallel is None or self.n_jobs <= 1:
            return M @ X
        cols = np.array_split(X, self.n_jobs, axis=1)
        res = Parallel(n_jobs=self.n_jobs)(
            jdelayed(_spmm_worker)(M.data, M.indices, M.indptr, M.shape, c) for c in cols
        )
        return np.concatenate(res, axis=1)

    def forward(self, target):
        """b = A x.  (nX, nY, nZ) volume -> (nX, n_angles, nZ) sinogram."""
        x = vamtoolbox.util.data.clipToCircle(target)
        if self.proj_geo.absorption_coeff is not None:
            x = self.proj_geo.absorption_mask * x
        # flatten each slice to astra vol order (nX==nY here, XY-padded square): (nX*nY, nZ)
        X2d = np.ascontiguousarray(x.reshape(self.nX * self.nY, self.nZ))
        B2d = self._spmm(self.A, X2d)                       # (nA*nX, nZ)
        # (nA, nX, nZ) -> (nX, nA, nZ)
        return B2d.reshape(self.nA, self.nX, self.nZ).transpose(1, 0, 2)

    def backward(self, b):
        """x = A^T b.  (nX, n_angles, nZ) sinogram -> (nX, nY, nZ) volume."""
        if self.proj_geo.zero_dose_sino is not None:
            b = b.copy()
            b[self.proj_geo.zero_dose_sino] = 0.0
        B2d = np.ascontiguousarray(b.transpose(1, 0, 2).reshape(self.nA * self.nX, self.nZ))
        X2d = self._spmm(self.AT, B2d)                      # (nX*nY, nZ) astra vol order
        x = X2d.reshape(self.nX, self.nY, self.nZ)          # (nX==nY square)
        if self.proj_geo.absorption_coeff is not None:
            x = self.proj_geo.absorption_mask * x
        return vamtoolbox.util.data.clipToCircle(x)
