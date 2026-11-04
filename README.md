# Drug-Target Affinity Prediction

A machine-learning project that predicts how strongly a small molecule binds to a protein target (pKi), using only the molecule's SMILES string and the protein's amino-acid sequence.

The pipeline cleans BindingDB data, calculates molecular features with RDKit and protein features from the sequence, combines them into one feature vector and trains a PyTorch neural network to predict pKi.

**Status:** work in progress. The full workflow from raw BindingDB data to a trained network runs. I'm now rebuilding the evaluation, because the original test setup isn't rigorous enough to show how well the model works on molecules or proteins it hasn't seen.

## Project at a glance

- **Task:** regression, predicting pKi for a (molecule, protein) pair
- **Data:** BindingDB Ki values, 602,793 training rows and 120,559 test rows
- **Input:** 3,429 features per pair (molecule + protein)
- **Model:** feed-forward neural network (MLP), 2048 → 1024 → 512 → 256 → 1
- **Libraries:** PyTorch, RDKit, pandas, NumPy, Biopython, propy3
- **Initial result:** Pearson ≈ 0.90 on the original test set (preliminary, see [Evaluation](#evaluation-and-limitations))
- **Current focus:** reproducible preprocessing and a more rigorous evaluation

## Pipeline

```text
                    BindingDB (Ki values)
                            |
                            v
                      Data cleaning
                            |
            +---------------+---------------+
            |                               |
            v                               v
       Drug SMILES                  Protein sequence
            |                               |
            v                               v
   Morgan fingerprint              Sequence-derived
 + molecular descriptors          protein descriptors
            |                               |
            +---------------+---------------+
                            |
                            v
                Combined feature vector (3,429)
                            |
                            v
                      Standardisation
                            |
                            v
                        PyTorch MLP
                            |
                            v
                       Predicted pKi
```

The target is

```text
pKi = 9 - log10(Ki in nM)
```

so a higher pKi means tighter binding.

## Data

The data comes from `BindingDB_All.tsv` from [BindingDB](https://www.bindingdb.org). The pipeline reads three columns: ligand SMILES, Ki (nM) and BindingDB Target Chain Sequence 1. Only the first chain of each target is used, so protein complexes are represented by chain 1 alone. The raw file isn't included in the repo because of its size.

Cleaning steps:

- drop rows with no Ki value;
- cut each SMILES at the first `|` (BindingDB sometimes adds extra information after it) and skip any SMILES RDKit can't parse;
- skip sequences shorter than 11 residues or containing anything other than the 20 standard amino acids;
- strip `<`, `>` and `=` from Ki values, use the number as if it were exact, and mark those rows with an `Inaccurate` flag;
- convert Ki to pKi (Ki = 0 is currently mapped to pKi = 0, which is a temporary workaround).

The final processed tables have 3,430 columns: 3,428 descriptors, the `Inaccurate` flag and the pKi target.

**Known issue:** the `Inaccurate` flag is currently also used as a model input. That's a mistake, because it describes how the label was measured, which you wouldn't know for a new pair. It will be removed in the rebuilt pipeline.

## Features

### Molecule

A **2,048-bit Morgan fingerprint** (radius 2, chirality included) plus 19 descriptors:

- exact molecular weight, Crippen LogP, TPSA, formal charge;
- hydrogen-bond donors and acceptors;
- sp, sp2 and sp3 atom counts;
- single, double, triple and aromatic bond counts;
- rotatable bonds, conjugated bonds and conjugation ratio, bonds in rings;
- aromatic heterocycle and carbocycle counts.

### Protein

- amino-acid composition and dipeptide composition;
- CTD (composition, transition, distribution);
- PAAC and APAAC (λ = 10, weight = 0.05);
- Moreau-Broto, Moran and Geary autocorrelation;
- molecular weight, aromaticity, isoelectric point, instability index and GRAVY.

`Properties.md` is my original feature plan. It lists an aliphatic index for proteins that I haven't implemented.

### Standardisation

`Standardising.ipynb` z-scores every column with more than two distinct values (except pKi). Binary columns, including the fingerprint bits, are left unchanged.

## Model and training

| | |
|---|---|
| Architecture | 4 × (Linear → BatchNorm → ReLU → Dropout 0.1) with widths 2048, 1024, 512, 256, then Linear(256, 1) |
| Loss | Huber loss |
| Optimiser | Adam, lr = 1e-4, weight decay = 2e-5, gradient clipping at 1.0 |
| Scheduler | ReduceLROnPlateau (patience 3), currently watching training loss |
| Batch size / epochs | 128 / 50 |
| Validation set | none yet |

`model_weights1.pth` contains weights for this architecture. I haven't recorded evaluation metrics for it yet.

## Evaluation and limitations

### Initial results

The saved output of `Code/Testing.ipynb` shows:

| Metric | Initial result |
|---|---:|
| MAE | 0.487 |
| RMSE | 0.696 |
| Pearson correlation | 0.898 |
| Spearman correlation | 0.889 |

These numbers show the pipeline learns something useful, but I treat them as **preliminary**, not as an honest estimate of performance. There are four reasons:

1. **They come from an earlier model.** The output was produced with `model_weights2.pth`, a 1024 → 512 → 256 → 128 network with dropout 0.2. That file isn't in the repo and isn't the architecture the notebooks define now.
2. **The split isn't documented.** No code in the repo creates `testing_data.parquet`, so I can't show how the split was made or whether the same molecules and proteins appear in both sets. BindingDB has many near-duplicate compounds and repeated targets, so random splits tend to look better than they should.
3. **Test scaling doesn't match training.** The test set was standardised with its own mean and standard deviation instead of the training statistics, and those statistics weren't saved.
4. **Metrics were averaged per batch.** The notebook calculates each metric per batch of 128 rows and averages them. That isn't the same as calculating them once over the whole test set, and the spread between batches isn't model uncertainty.

### Planned evaluation

I'm rebuilding the evaluation with one seeded train/validation/test split, the scaler fitted on training data only, and three increasingly difficult splits:

1. **Random split:** a reproducible baseline.
2. **Scaffold split:** test molecules have scaffolds not seen in training.
3. **Target-cluster split:** test proteins are separated from training proteins by sequence clustering.

Each result will report MAE, RMSE, Pearson and Spearman, calculated over the whole test set and across several training seeds.

| Model | Random split | Scaffold split | Target-cluster split |
|---|---|---|---|
| Mean predictor | TBD | TBD | TBD |
| Molecule features only | TBD | TBD | TBD |
| Protein features only | TBD | TBD | TBD |
| Random forest / gradient boosting | TBD | TBD | TBD |
| MLP (all features) | TBD | TBD | TBD |

The molecule-only and protein-only rows matter most. If either does nearly as well as the full model, the network is mostly learning average affinities for each molecule or protein rather than anything about how they interact.

## Repository structure

```text
Code/
  Fetching_data.ipynb            Loads BindingDB, drops rows with no Ki, writes CSV chunks
  Properties_application.ipynb   Calculates features, pKi and the Inaccurate flag
  Standardising.ipynb            Standardises the feature columns
  Training.ipynb                 Defines and trains the MLP
  Testing.ipynb                  Evaluates saved weights on the test set
  model_weights1.pth             Weights for the current architecture
Properties.md                    Original feature plan
```

The repo also contains some exploratory notebooks from earlier stages (descriptor examples, RDKit drawing, a brief look at other datasets). They aren't part of the main pipeline. `website/` is an early prediction prototype and doesn't give meaningful predictions yet, because inference doesn't apply the training standardisation.

## Running it

The project currently uses a notebook-based workflow rather than a one-command setup. The notebooks were last run on Python 3.14. Package versions aren't pinned yet:

```bash
pip install pandas numpy pyarrow scipy torch rdkit biopython propy3 matplotlib
```

Order:

1. Download `BindingDB_All.tsv` and set its path in `Fetching_data.ipynb`.
2. Run `Fetching_data.ipynb` to clean the data and split it into chunks.
3. Run `Properties_application.ipynb` to calculate features (create a `Dataframes/` folder first and set the chunk ranges to match your chunk count).
4. Run `Standardising.ipynb`.
5. Run `Training.ipynb` (add `torch.save(model.state_dict(), "model_weights1.pth")` to keep the weights).
6. Run `Testing.ipynb` with the path to the test file and the weights.

Some notebooks still have hardcoded paths, chunk ranges and `input()` prompts. Replacing these with one config file is on the roadmap.

## Roadmap

**Evaluation**

- [ ] One explicit, seeded train/validation/test split
- [ ] Combine repeated measurements of the same (molecule, protein) pair
- [ ] Fit the scaler on training data only, save it and reuse it everywhere
- [ ] Use the validation set for the scheduler and early stopping
- [ ] Calculate metrics over the whole test set, with variation across seeds
- [ ] Add scaffold and target-cluster splits and fill in the baseline table

**Pipeline**

- [ ] Remove `Inaccurate` from the model inputs (drop censored rows or handle them separately)
- [ ] Replace hardcoded paths, chunk ranges and prompts with one config file
- [ ] Move feature code and the model class into a shared module
- [ ] Save weights and log training loss from `Training.ipynb`
- [ ] Record the BindingDB release, row counts after each cleaning step and the number of distinct molecules and proteins

**Repository**

- [ ] Fix the prediction prototype (load the saved scaler, use relative paths) or remove it
- [ ] Remove notebooks not related to the project
- [ ] Move model weights to a GitHub Release
- [ ] Add a pinned `requirements.txt` and a licence
- [ ] Update this README with the final results

## Where it stands

The workflow runs from raw BindingDB rows to a trained network: data cleaning, feature calculation, standardisation and PyTorch training all work. What I haven't shown yet is how well the model predicts for molecules and proteins it hasn't seen. That's what I'm working on now, and I'd rather get an honest number, even if it turns out lower, than keep a good-looking one I can't back up.

Once the splits and baselines exist, I want to compare this feature-based MLP with a sequence-based CNN or a graph neural network, so the comparison is actually fair.

## Licence and credits

I haven't chosen a licence yet, so by default the code can't be reused. The project uses data from [BindingDB](https://www.bindingdb.org) and the libraries [RDKit](https://www.rdkit.org), [Biopython](https://biopython.org), [propy3](https://pypi.org/project/propy3/) and [PyTorch](https://pytorch.org). Their own licence and data-use terms apply.
