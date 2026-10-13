# morphodomains

Analysis code and figure generation for Leroy, van Leen et al., *Morphodynamic domains enable integration of live morphometrics and spatial transcriptomics*, Cell Reports (2026). Code and data: [10.5281/zenodo.19633285](https://doi.org/10.5281/zenodo.19633285).

Morphodynamic domains are regions of tissue with coherent cell behaviours. Here they are defined from live imaging of the *Drosophila* pupal notum, linked to a reconstructed spatial transcriptome, and used to compare genotypes. The tissue measurements come from [AnimalProcessing](https://github.com/BellaicheLab/AnimalProcessing) and the spatial transcriptome from [vgep](https://github.com/BellaicheLab/vgep).

## Walkthrough notebooks

The four notebooks follow the analysis of the paper on the notum data:

1. {doc}`Morphodynamic domains <notebooks/01_morphodynamic_domains>`: PCA and spatially constrained clustering of 30 morphogenetic features.
2. {doc}`Spatial transcriptome <notebooks/02_spatial_transcriptome_reconstruction>`: mapping scRNA-seq onto a 24-gene reference atlas to reconstruct 4,312 gene expression patterns with vGEP.
3. {doc}`Gene regulatory networks <notebooks/03_grn_analysis>`: tau specificity scoring and GENIE3-based GRN inference for each domain.
4. {doc}`Comparing genotypes <notebooks/04_morphodynamics_comparison>`: global and local Moran's I to find when and where a perturbation changes tissue flow, deformation and cell dynamics.

## Installation

Python 3.10 or later is required.

```bash
pip install git+https://github.com/BellaicheLab/AnimalProcessing.git
pip install git+https://github.com/BellaicheLab/vgep.git
git clone https://github.com/BellaicheLab/morphodomains.git
pip install -e morphodomains
```

PERLER, used in the benchmark, is not on PyPI. Install it without its outdated scipy pin:

```bash
pip install "git+https://github.com/yasokochi/Perler.git" --no-deps
```

Then change line 348 of `perler/perler_class.py` to `DM = distance.cdist(self.__mu, self.__r, metric="mahalanobis", VI=VI)`. PERLER passes this matrix as `V=`, which SciPy 1.7 and later reject.

## Data

Download and extract the data from Zenodo ([10.5281/zenodo.19633285](https://doi.org/10.5281/zenodo.19633285)), then point `MORPHODOMAINS_DATA` to the directory that contains `data/`:

```bash
export MORPHODOMAINS_DATA="/path/to/extracted_archive"
```

## Related repositories

[AnimalProcessing](https://github.com/BellaicheLab/AnimalProcessing) quantifies tissue mechanics from movies. [vgep](https://github.com/BellaicheLab/vgep) reconstructs the spatial transcriptome and infers gene regulatory networks, and [vgep_gui](https://github.com/BellaicheLab/vgep_gui) is the web app to browse it ([live app](https://ybellaichelab.shinyapps.io/NotumvGEP/)).

```{toctree}
:maxdepth: 1
:caption: Walkthrough
:hidden:

notebooks/01_morphodynamic_domains
notebooks/02_spatial_transcriptome_reconstruction
notebooks/03_grn_analysis
notebooks/04_morphodynamics_comparison
```
