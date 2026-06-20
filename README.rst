.. image:: /docs/_static/logos/logo_bone.png
   :height: 200px
----

.. |conda| image:: https://anaconda.org/vamtoolbox/vamtoolbox/badges/version.svg
   :target: https://anaconda.org/vamtoolbox/vamtoolbox

.. |rtd| image:: https://readthedocs.org/projects/vamtoolbox/badge/?version=latest
   :target: https://vamtoolbox.readthedocs.io/en/latest/?badge=latest

.. |zen| image:: https://zenodo.org/badge/500715593.svg
   :target: https://zenodo.org/badge/latestdoi/500715593

+----------------------+-----------+
| Deployment           | |conda|   |
+----------------------+-----------+
| Documentation        | |rtd|     |
+----------------------+-----------+
| Citation             | |zen|     |
+----------------------+-----------+

VAMToolbox is a Python library to support the generation of the light projections and the control of a DLP projector for tomographic volumetric additive manufacturing. It provides visualization, various optimization techniques, and flexible projection geometries to assist in the creation of sinograms and reconstructions for simulated VAM.

Prefer a graphical workflow? `Tomo <https://github.com/computed-axial-lithography/tomo>`_ is a standalone Windows desktop application that wraps VAMToolbox end-to-end (load STL → voxelize → optimize → preview/export) with no Python setup required. See the `Tomo desktop application`_ section below.

VAMToolbox 3.0.0 Release Notes:
-------------------------------
This release adds a high-level pipeline API, a desktop GUI (Tomo), large-part/high-resolution performance work, resolution-aware physics, and a simplified ASTRA-based install. Changed/added modules are listed (per the license's "mark your changes" requirement; changes made 2026-06).

1. Tomo desktop application

   `Tomo <https://github.com/computed-axial-lithography/tomo>`_, a standalone Windows GUI built on VAMToolbox, is now available. See the `Tomo desktop application`_ section below.

2. High-level pipeline API (new ``vamtoolbox/pipeline.py``)

   A clean, GUI-facing API — ``PrintConfig`` (a JSON-serializable dataclass of all job parameters, with ``.validate()``) and ``VAMPipeline`` (stateful: ``detect_hardware`` / ``voxelize`` / ``optimize`` / ``rebin`` / ``save_video`` / ``run``, with staged progress callbacks, ETA estimation, and cancellation). No environment variables or globals are required to drive a full job. Re-exported at the top level as ``vamtoolbox.PrintConfig`` / ``VAMPipeline`` / ``run_print``. Example: ``examples/gui_integration_example.py``.

3. Optimization and large-volume performance

   Per-iteration progress callbacks added to the OSMO and BCLP loops (for live GUI progress). Memory-scalable optimization for billion-voxel volumes via z-slab chunking in the 3D projectors (``Projector3DParallel`` / ``Projector3DParallelCUDA``), a new GPU ray-density projector (``projector/ProjectorRayDensityGPU.py``), and a low-memory BCLP variant (``optimizer/BCLP_lowmem.py``). On the CPU path, a sparse-matrix projector substantially reduces per-iteration cost.

4. Resolution-aware physics corrections

   Absorption (``geometry.py``) and diffusion (``response.py``) corrections now use the real physical voxel pitch rather than pixel units, so results are consistent across resolutions.

5. Hardware detection (new ``vamtoolbox/util/hardware.py``)

   CUDA capability detection and auto-tuning of run parameters, with a CPU fallback.

6. Native install, ASTRA backend, and a one-command installer (conda deprecated)

   VAMToolbox uses the `ASTRA Toolbox <https://astra-toolbox.com>`_ as its CUDA tomography backend for the projection/reconstruction operators. **conda is now deprecated** as the install path; VAMToolbox installs natively into a plain ``pip``/venv environment on **Python 3.13**. ASTRA is installed from the standalone, CUDA-bundled download on the `ASTRA Toolbox site <https://astra-toolbox.com/downloads/>`_ rather than a conda channel. A new ``install.ps1`` script at the repo root does the whole setup in one command — creates the venv, downloads and installs ASTRA (and its VC++ redistributable), installs ``requirements-py313.txt`` and VAMToolbox, and verifies CUDA. This is also how the bundled Tomo runtime ships ASTRA. See `Installation`_.

7. Cleanup and packaging

   ``torch`` is now an optional dependency — the imports in ``vamtoolbox/__init__.py`` and ``vamtoolbox/projector/__init__.py`` are guarded so the OSMO/BCLP + ASTRA/sparse path runs without torch installed (enabling a much slimmer bundled runtime). Added Python 3.13 support (``requirements-py313.txt``), a ``pytest`` test suite under ``tests/``, and additional usage examples under ``examples/``.


VAMToolbox 2.0.0 Release Notes:
------------
This major release includes a number of new features and improvements. The major changes are listed below. For more details, please refer to the documentation.

1. General loss function to formulate optimization of grayscale response profile with high tunability

   Added a formulation of the general optimization problem called Band-Contraint-Lp-norm (BCLP) minimization. This loss function formulation generalizes three existing optimization schemes and is capable to optimize for grayscale target values.
   Added material response model for capturing non-linear relationships between optical dose and the desired response (such as conversion). This allows us to optimize response profile instead of dose profile.
   Refer to the "Band-Contraint-Lp-norm" section for details of BCLP and material response.

2. Ray tracing propagator

   Added a ray tracing propagator to model light attenuation and refraction in medium with gradient refractive index. This ray tracer is written with pyTorch and compatible with python CUDA libraries like cuPy. This ray tracing is performed on GPU.
   Added a models of spatial variant attenuation coefficient, absorption coefficient and refractive index to support ray tracing operations and optimization. 
   Refer to the "Ray tracing propagator and algebraic propagator" section for details of ray tracing.

3. OpenGL voxelizer

   Major performance improvement over pure python voxelizers. This is important for voxelizing large objects at high resolution. Please refer to `voxelizestl <https://github.com/computed-axial-lithography/VAMToolbox/blob/main/examples/voxelizestl.py>`_ for example usage. This is also available as a standalone package `OpenGL Voxelizer <https://github.com/computed-axial-lithography/OpenGL-voxelizer>`_.

4. 3MF import (with beam-lattice support)

   Import 3MF files — including the **beam-lattice extension** (strut graphs with per-end radii and ball nodes, as produced by lattice generators such as the VolumeFillingLattice add-on + Blender 3MF Exporter) — directly as voxel targets. Solid meshes and beam lattices are read via the lib3mf SDK; beam lattices are voxelized analytically (capsule signed-distance), which scales to dense lattices and needs no GPU/OpenGL. Usage: ``TargetGeometry(threemffilename="lattice.3mf", resolution=100)`` (a ``.3mf`` passed to ``stlfilename`` is auto-routed). ``lib3mf`` is an optional dependency (``pip install lib3mf``).


Band-Contraint-Lp-norm (BCLP) minimization (contribution from `LDCT-VAM <https://github.com/facebookresearch/LDCT-VAM>`_)
------------
This new loss function unified the optimization for both real-valued (grayscale) and binary targets. It is a generalization of three existing projection optimization schemes.
The target tomogram can now be specified in physical unit of response (such as degree-of-conversion, elastic modulus, or refractive index).
BCLP uses a material response model to capture the non-linear relationship between the response and optical dose. 
This response model is consistently implemented throughout initialization, optimization and evaluation. 
The physical unit of projection parameters (sinogram), optical dose and material response are all preserved during optimization for experimental calibration purposes.
One major benefit of the BCLP formulation is that it allows numerous optimization features to be implemented in a unified framework.
The BCLP loss function provides control over local response tolerance, local weighting and global error sparsity.
For details of the BCLP formulation, please refer to our arXiv publication "Tomographic projection optimization for volumetric additive manufacturing with general band constraint Lp-norm minimization".


Ray tracing propagator and algebraic propagator (contribution from `LDCT-VAM <https://github.com/facebookresearch/LDCT-VAM>`_)
------------
The light propagation model is one of the most critical elements in projection optimization. A ray tracing propagator is coded in pyTorch to models light attenuation, absorption and refraction in medium with gradient refractive index.
The light attenuation, absorption and refraction is simulated based on a spatial description of the simulation medium.
Additionally, the ray tracer can generate an algebraic representation of the propagation such that various algebraic techniques in tomography can be applied.
LCDT-VAM provides algebraic propagators (one in scipy and one in pyTorch) to compute light propagation via matrix-vector multiplication.
The memory-intensive algebraic represntation is only practical for 2D problems or 3D shift-invariant problems (shift-invariant in z direction, along the rotation axis).
However, when the propagation can be performed algebraically, the computation is much faster than the ray tracing propagator.
For futher details of the algebraic representation, refers to the supplementary of the BCLP publication above.


Installation
------------

*NOTE: This toolbox is currently only compatible with Windows OS, and requires Python 3.13.*

As of 3.0.0, VAMToolbox installs natively into a plain Python virtual environment — **conda is no longer required** (see `Deprecated: conda`_ below).

**Recommended: one-command install**

From a checkout of this repository, run::

   powershell -ExecutionPolicy Bypass -File install.ps1

``install.ps1`` does everything end-to-end: it finds your Python 3.13, creates a virtual environment (``.venv``), downloads the standalone CUDA-bundled ASTRA build from the `ASTRA Toolbox downloads <https://astra-toolbox.com/downloads/>`_, installs it (plus the bundled VC++ redistributable), installs all Python requirements, installs VAMToolbox itself, and verifies that ``astra.use_cuda()`` works. Re-running it reuses the existing environment.

Useful flags: ``-SkipTorch`` (smaller install — ``torch`` is only needed for the pyTorch ray-tracing / algebraic propagators), ``-AstraZip <path>`` (use an already-downloaded ASTRA zip for offline installs), and ``-VenvPath <dir>``.

When it finishes, activate the environment with ``.venv\Scripts\Activate.ps1``.

**Manual install**

If you prefer to do it by hand:

1. Create and activate a Python 3.13 virtual environment::

      python -m venv .venv
      .venv\Scripts\activate

2. Install ASTRA. Download ``astra-toolbox-2.4.1-python313-win-x64.zip`` from the `ASTRA Toolbox downloads <https://astra-toolbox.com/downloads/>`_, unpack it, run the included ``vc_redist.x64.exe``, and pip-install the ``.whl`` inside::

      pip install astra_toolbox-2.4.1-cp313-cp313-win_amd64.whl

   *Note:* ``pip install astra-toolbox`` from PyPI is **not** sufficient on Windows (PyPI ships Linux-only wheels) — use the standalone wheel from the ASTRA download above.

3. Install the remaining requirements and VAMToolbox::

      pip install -r requirements-py313.txt
      pip install -e .

``astra.use_cuda()`` should return ``True`` on a CUDA-capable NVIDIA GPU.

Deprecated: conda
~~~~~~~~~~~~~~~~~~
The previous conda packages are deprecated and no longer the supported install path. The native install above replaces::

   conda install vamtoolbox -c vamtoolbox -c conda-forge -c astra-toolbox

For more information, refer to the `installation documentation <https://vamtoolbox.readthedocs.io/en/latest/_docs/gettingstarted.html>`_.


Resources
---------
View the `documentation <https://vamtoolbox.readthedocs.io/en/latest/_docs/intro.html>`_ site.


Tomo desktop application
------------------------
`Tomo <https://github.com/computed-axial-lithography/tomo>`_ is a standalone Windows desktop GUI built on top of VAMToolbox. It exposes the full pipeline — load one or more STLs, voxelize on the GPU, optimize the projections (OSMO or BCLP), then preview and export a print-ready projection video — through a guided four-stage interface (Prep → Voxelize → Optimize → Preview), with live 3D previews, absorption/diffusion correction toggles, hardware auto-tuning, and a z-slab memory mode for very large parts.

Tomo is a thin front end: all voxelization, optimization, and physics are performed by VAMToolbox via the high-level ``vamtoolbox.pipeline`` API (see `VAMToolbox 3.0.0 Release Notes`_). It ships as a single NSIS installer with a self-contained Python/CUDA runtime bundled in (including ASTRA), so end users do not need to set up Python, conda, or CUDA. It is released under the same UC Regents license as VAMToolbox.


License
Copyright © 2026 The Regents of the University of California. All Rights Reserved.
This software is source-available under a custom academic license developed by UC Berkeley. It is not OSI-approved open source.
Free use is permitted for:

Educational, research, and non-profit entities for noncommercial purposes, including distribution of modifications — provided that any distributed modifications or derivative works are released under this same license, accompanied by source code, and marked with the changes made
Commercial entities for internal use only (no distribution, no productization, no hosted/SaaS offerings)

A commercial license is required for:

Incorporating this software into a commercial product
Distributing the software commercially
Offering the software as a hosted or SaaS service
Any other use whose value derives substantially from this software

No patent rights are granted under the source-available license; patent rights are licensed separately as part of a commercial license.
Redistribution and Modifications
If you redistribute this software, or distribute modifications or derivative works of it, under the noncommercial terms above, you must:

License your modifications under this same license. Modified or derivative works must carry the same terms; you may not relicense them under a different license, including more permissive ones.
Include the complete source code of your modifications alongside any distribution.
Mark your changes. Modified files must carry prominent notices stating that you changed the file and the date of the change.
Preserve this license. The original copyright notice and the full text of this LICENSE file must be included in every copy and modification.
Stay within the noncommercial scope. Redistribution is permitted only by educational, research, and non-profit entities for noncommercial purposes. Any commercial redistribution — including bundling into a commercial product or offering as a hosted/SaaS service — requires a commercial license from OTL.

Commercial entities operating under the internal-use terms may modify the software for their own internal evaluation and use, but may not distribute the original software, modified versions, or derivative works in any form without first obtaining a commercial license.
Commercial Licensing
Commercial licensing for this software is administered by UC Berkeley's Office of Technology Licensing (OTL) at the Berkeley Intellectual Property & Industry Research Alliances (IPIRA).

Email: otl@berkeley.edu
Web: https://ipira.berkeley.edu
