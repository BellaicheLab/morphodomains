# morphodomains

Figure generation and walkthrough notebooks for Leroy, van Leen *et al.*, *Morphodynamic domains enable integration of live morphometrics and spatial transcriptomics*, Cell Reports (2026).

The scripts produce the panels of the paper from the data of the Zenodo archive. The figures and panels to generate are set in `scripts/figures/figure_config.yaml`. Four notebooks walk through the main analyses.

Documentation: [https://morphodomains.readthedocs.io](https://morphodomains.readthedocs.io)

## Installation

```bash
conda create --name morphodomains python=3.10
conda activate morphodomains

pip install git+https://github.com/BellaicheLab/AnimalProcessing.git
pip install git+https://github.com/BellaicheLab/vgep.git

# perler is not on PyPI; --no-deps skips its outdated scipy pin
pip install "git+https://github.com/yasokochi/Perler.git" --no-deps
# then edit perler/perler_class.py line 348 to
#   DM = distance.cdist(self.__mu, self.__r, metric="mahalanobis", VI=VI)
# (perler passes V=, which SciPy >= 1.7 rejects)

git clone https://github.com/BellaicheLab/morphodomains.git
pip install -e morphodomains
```

## Data

Download and extract the data from Zenodo ([10.5281/zenodo.19633285](https://doi.org/10.5281/zenodo.19633285)). Inside the extracted archive nothing needs configuring. Otherwise, point `MORPHODOMAINS_DATA` to the directory that contains `data/`:

```bash
export MORPHODOMAINS_DATA="/path/to/extracted_archive"
```

## Walkthrough notebooks

```bash
cd notebooks && jupyter lab
```

1. `01_morphodynamic_domains`: PCA and spatially constrained clustering of 30 morphogenetic features into morphodynamic domains.
2. `02_spatial_transcriptome_reconstruction`: vGEP reconstruction of 4,312 genes from scRNA-seq and a 24-gene reference atlas, validated on 20 genes left out of the atlas (13 smiFISH and 7 protein reporters).
3. `03_grn_analysis`: tau specificity scoring and GENIE3-based GRN inference for each domain.
4. `04_morphodynamics_comparison`: global and local Moran's I to find when and where *TolloRNAi* differs from the *vRNAi* control.

## Generating figures

```bash
python scripts/figures/generate.py                          # all enabled figures
python scripts/figures/generate.py --figure grn             # one figure
python scripts/figures/generate.py --figure flow_deformation --panel velocity_maps
python scripts/figures/generate.py --list                   # list available panels
python scripts/figures/generate.py --figure flow_deformation --display-all  # keep viewers open
```

Outputs go to `figures/fig_<name>/`. Use `--display-all` on a single figure only, as it keeps every napari viewer open.

Figure modules by manuscript figure:

- Figure 1: `figure_flow_deformation` (velocity and deformation tensor maps).
- Figure 2: `figure_morphodynamic_domains` (PCA, clustering and spatial tensor maps).
- Figure 3: `figure_spatial_transcriptome` (spatial transcriptome reconstruction and benchmark).
- Figure 4: `figure_grn` (TF specificity and gene regulatory networks).
- Figure 5: `figure_morphodynamics_comparison` (Moran's I, *vRNAi* vs *TolloRNAi*), `figure_apoptosis_compression` (apoptosis and compression over time) and `figure_delamination_analysis` (apoptosis in the Lagrangian frame).
- Supplementary figures: `figure_tollo_gfp` (Tollo:GFP at four time points), `figure_clustering_stability`, `figure_clustering_temporal` and `figure_clustering_spatiotemporal` (robustness of the clustering), `figure_atlas_size` (reconstruction vs atlas size), `figure_gene_atlas` (gene atlas on a common grid) and `figure_neck_archetype_spread` (gene atlas with and without the neck landmark).

Run `figure_morphodynamic_domains` before `figure_spatial_transcriptome` and `figure_grn`, and `figure_morphodynamics_comparison` before `figure_apoptosis_compression`.

## Related repositories

[AnimalProcessing](https://github.com/BellaicheLab/AnimalProcessing) quantifies tissue mechanics and computes the spatial statistics. [vgep](https://github.com/BellaicheLab/vgep) reconstructs the spatial transcriptome and infers gene regulatory networks, and [vgep_gui](https://github.com/BellaicheLab/vgep_gui) is the web app to browse it ([live app](https://ybellaichelab.shinyapps.io/NotumvGEP/)).
