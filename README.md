# AIDD8122 final project: induction chemotherapy response in nasopharyngeal carcinoma

**Student:** CEN FENG  
**Course:** AIDD8122, Advanced Topics in Artificial Intelligence & Drug Discovery

This repository contains the analysis code, public-data inputs, and numerical outputs for an exploratory study of whether pretreatment tumour gene expression predicts short-term response to induction chemotherapy in nasopharyngeal carcinoma (NPC). The study uses [GEO GSE132112](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE132112): 95 samples, comprising 71 complete/partial responders and 24 patients with stable disease. The endpoint is **response**, not long-term survival or the causal benefit of one treatment over another.

## Main finding

The ten-gene model's mean training-fold AUROC was 0.974, but its mean out-of-fold AUROC was 0.412 across ten repeats of stratified five-fold cross-validation. The PCA model also did not outperform a prevalence-only baseline. Among four prespecified pathway models, the EMT-only model had the highest mean out-of-fold AUROC (0.568); a permutation test accounting for selection of the best of four models gave **p = 0.415**. At the fixed 0.5 response-probability threshold, it identified none of the 24 nonresponders. These results do **not** establish a clinically useful predictor or support individual treatment decisions.

## Repository contents

| Path | Purpose |
| --- | --- |
| `GSE132112_extract_samples.py` | Extract response labels and sample IDs from the public GEO Series Matrix. |
| `GSE132112_build_gene_matrix.py` | Convert GEO probe-level expression to a gene-level matrix without using response labels. |
| `GSE132112_baseline_cv.py` | Ten-gene, PCA, and prevalence-only baselines with repeated stratified cross-validation. |
| `GSE132112_stage4_pathway_cv.py` | Three fixed Hallmark pathway models, their combination, and the permutation test. |
| `GSE132112_samples.csv` | Public GEO sample IDs and response labels. |
| `GSE132112_gene_expression.csv.gz.part00`–`part07` | Eight pieces of the reproducible 23,934-gene × 95-sample matrix derived from GEO. |
| `assemble_gene_matrix.py` | Reassemble the pieces and verify the matrix SHA-256 checksum. |
| `GSE132112_stage4_hallmark_sets.gmt` | Fixed DNA repair, hypoxia, and EMT Hallmark gene lists. |
| `*_metrics.json`, `*_oof_predictions.csv` | Out-of-fold predictions and full numerical summaries, including the exploratory three-sample sensitivity analysis. |

The 950 prediction rows in each main out-of-fold file represent **95 patients × 10 resampling repeats**, not 950 independent patients.

## Reproduce the models

Tested with Python 3.12.14, NumPy 2.3.5, and pandas 2.2.3. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
mkdir -p reproduced
python assemble_gene_matrix.py --output reproduced/GSE132112_gene_expression.csv.gz

python GSE132112_baseline_cv.py \
  --matrix reproduced/GSE132112_gene_expression.csv.gz \
  --labels GSE132112_samples.csv \
  --predictions reproduced/baseline_oof_predictions.csv \
  --summary reproduced/baseline_metrics.json

python GSE132112_stage4_pathway_cv.py \
  --matrix reproduced/GSE132112_gene_expression.csv.gz \
  --labels GSE132112_samples.csv \
  --sets GSE132112_stage4_hallmark_sets.gmt \
  --predictions reproduced/stage4_oof_predictions.csv \
  --summary reproduced/stage4_metrics.json \
  --permutations 199
```

The second command repeats model fitting for 199 shuffled-label permutations and may take substantially longer. The baseline script also supports `--exclude-samples LA-NPC_78 LA-NPC_69 LA-NPC_47` for the exploratory sensitivity analysis. The main analysis retains all 95 samples.

## Rebuild the public-data inputs

The eight included matrix parts and labels allow model reproduction without downloading the original 283 MB compressed probe-level file. Run the assembly command above before fitting the models; it verifies the reconstructed file checksum. For a full rebuild, obtain the [GSE132112 Series Matrix](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE132nnn/GSE132112/matrix/GSE132112_series_matrix.txt.gz), [processed probe-level expression](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE132nnn/GSE132112/suppl/GSE132112_nor_data_probelevel.txt.gz), and [GPL17585 platform annotation](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GPL17585). Verify their SHA-256 values before use:

| Source | SHA-256 |
| --- | --- |
| Series Matrix | `9c005399455cc91fd0e2505d27bc30d7d124721ad3cc20b303ba09b771425f53` |
| Probe-level expression | `12ae0d9d3b2b4fe2032c2bd49ba747ba53ee615e034b8bec1fe2756f2c17fbc3` |
| GPL17585 annotation text | `a434780c57154c39c29ee4ed38b0835cc4a8bd518a9f69699c9e924cde2f4afb` |

Run these commands after naming the downloaded files as shown:

```bash
python GSE132112_extract_samples.py \
  --series-matrix raw/GSE132112_series_matrix.txt.gz \
  --output reproduced/GSE132112_samples.csv

python GSE132112_build_gene_matrix.py \
  --expression raw/GSE132112_nor_data_probelevel.txt.gz \
  --annotation raw/GPL17585_annotation.txt \
  --output reproduced/GSE132112_gene_expression.csv.gz
```

The gene-matrix script retains PSR probes that map to exactly one gene symbol and takes the per-sample median across probes for each gene. The resulting matrix has SHA-256 `8f45417fbbd407fdbb93b3f6c64a596a4d78235d10e3d194c52b0c6ef89d3f70`. This is a simplified representation and can obscure transcript-specific effects.

The Hallmark sets were fixed before fitting the pathway models. Gene lists and descriptions are attributed to [MSigDB Hallmark DNA Repair](https://www.gsea-msigdb.org/gsea/msigdb/human/geneset/HALLMARK_DNA_REPAIR.html), [Hypoxia](https://www.gsea-msigdb.org/gsea/msigdb/human/geneset/HALLMARK_HYPOXIA.html), and [Epithelial Mesenchymal Transition](https://www.gsea-msigdb.org/gsea/msigdb/human/geneset/HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION.html); see the [MSigDB license terms](https://www.gsea-msigdb.org/gsea/msigdb_license_terms.jsp).

## Interpretation limits

This is one selected public cohort with only 24 stable-disease cases and no external test cohort. The observed response proportion is affected by the original study's sampling design. Repeated cross-validation measures internal stability; it cannot show transportability to other hospitals or treatment regimens. The exploratory EMT signal was not significant after the stated four-model comparison and should not be used for patient stratification.

**Primary dataset publication:** Lei et al., 2021, *Journal of the National Cancer Institute*, [PMCID: PMC8023816](https://pmc.ncbi.nlm.nih.gov/articles/PMC8023816/).
