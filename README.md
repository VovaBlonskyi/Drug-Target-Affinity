# Drug–Target Affinity Prediction

> **Status: work in progress.** I'm trying to predict how tightly a small molecule binds to a protein (a number called pKi) using only the molecule's SMILES string and the protein's amino-acid sequence. The whole thing runs from raw data to a trained network, but I've only evaluated it once, on a train/test split I never documented properly. So I don't know yet how well it works on molecules or proteins it hasn't seen. Please don't treat it as a real drug-discovery tool.

## What it does

For each (molecule, protein) pair I calculate a lot of hand-made features for both sides (molecular weight, LogP, a fingerprint for the molecule, amino-acid composition for the protein, and so on), join them into one long vector, standardise it, and feed it into a feed-forward neural network in PyTorch. The network predicts pKi, where `pKi = 9 - log10(Ki in nM)`, so a higher number means tighter binding.

At a glance:

- **Data:** BindingDB Ki values, 602,793 training rows and 120,559 test rows
- **Input:** 3,429 features per pair
- **Model:** a plain MLP, 2048 → 1024 → 512 → 256 → 1
- **Libraries:** PyTorch, RDKit, Biopython, propy
- **Best number so far:** Pearson ≈ 0.90, but I don't fully trust it (see [Evaluation](#evaluation))

## Data

I used `BindingDB_All.tsv` from [BindingDB](https://www.bindingdb.org) and only read three columns: `Ligand SMILES`, `Ki (nM)` and `BindingDB Target Chain Sequence 1`. That last one means I only use the first chain of each target, so a protein complex is represented by chain 1 alone. The data isn't in the repo (it's far too big), and I didn't write down which BindingDB release I downloaded, so I can't say exactly which snapshot it was.

I also had a look at ChEMBL (`chembl_36.db`) in `Importing_db.ipynb`, but I only read the first 100 rows of the `activities` table. ChEMBL isn't used for training.

What my code does to the data:

1. Drops rows with no Ki value (`Fetching_data.ipynb`).
2. Cuts each SMILES at the first `|` (BindingDB sometimes adds extra info after it) and skips any SMILES that RDKit can't parse.
3. Skips sequences shorter than 11 residues, or containing anything other than the 20 standard amino acids.
4. Strips `>`, `<` and `=` from Ki values, uses the number as if it were exact, and sets an `Inaccurate` flag to 1 for those rows.
5. Turns `Ki = 0` into `pKi = 0`, which is a hack.

I didn't record how many rows survive each step. The final tables, from the saved notebook outputs, are:

| Table | Rows | Columns | Where I got the number |
|---|---:|---:|---|
| Training (`training_data.parquet`) | 602,793 | 3,430 | `Properties_application.ipynb` |
| Test (`standardised_testing_data.parquet`) | 120,559 | 3,430 | `Testing.ipynb` |

Each table has 3,428 descriptor columns, the `Inaccurate` flag and the `pKi` target. I also never counted how many distinct molecules and proteins there are.

One thing I'm not happy about: right now the `Inaccurate` flag is also fed to the model as an input. That's a mistake, because it describes how the label was measured, so you'd never know it for a new pair. I want to remove it.

## Features

**Molecule** (19 single-number descriptors plus a 2,048-bit Morgan fingerprint): exact molecular weight, Crippen LogP, hydrogen-bond donors and acceptors, TPSA, aromatic heterocycle and carbocycle counts, formal charge, sp/sp2/sp3 atom counts, single/double/triple/aromatic bond counts, rotatable bonds, conjugated bonds and conjugation ratio, bonds in rings, and the fingerprint (radius 2, chirality included).

**Protein:** amino-acid composition, dipeptide composition, CTD (composition, transition, distribution), PAAC and APAAC (λ = 10, weight = 0.05), Moreau–Broto, Moran and Geary autocorrelation, plus molecular weight, aromaticity, isoelectric point, instability index and GRAVY.

`Properties.md` is my original plan for the features. It lists an aliphatic index for proteins that I never actually implemented.

**Standardisation.** In `Standardising.ipynb` I z-score every column that has more than two distinct values (except `pKi`). Binary columns, including the Morgan bits and `Inaccurate`, are left alone. There are two problems with how I did it: I didn't save the means and standard deviations, and I standardised the test set using its own statistics instead of the training ones.

## Model and training

| | |
|---|---|
| Architecture | `Linear → BatchNorm → ReLU → Dropout(0.1)` four times (2048, 1024, 512, 256), then `Linear(256, 1)` |
| Loss | Huber loss |
| Optimiser | Adam, lr = 1e-4, weight decay = 2e-5, gradient clipping at 1.0 |
| Scheduler | `ReduceLROnPlateau(patience=3)`, watching the **training** loss |
| Batch size / epochs | 128 / 50 |
| Validation set | none |

`model_weights1.pth` has this architecture (input width 3,429). `Training.ipynb` doesn't contain a `torch.save` call, so the notebook alone doesn't produce the weights file. I never recorded any metrics for `model_weights1.pth` and didn't save a training log. The file is 39 MB, which probably belongs in a GitHub Release rather than in the repo.

## Evaluation

### The result I have (can't be reproduced from this repo)

The saved output in `Code/Testing.ipynb` shows:

| Metric | Result |
|---|---:|
| Mean absolute error | 0.4868 ± 0.0447 |
| Root mean squared error | 0.6964 ± 0.0836 |
| Pearson correlation | 0.8977 ± 0.0272 |
| Spearman correlation | 0.8886 ± 0.0269 |

I'd love to say that means it works, but there are four problems with it:

- **It's from a different model.** That output came from loading `model_weights2.pth`, a 1024 → 512 → 256 → 128 network with dropout 0.2. That file isn't in this repo, and it isn't the architecture my notebooks define now.
- **The ± isn't model uncertainty.** The notebook loops 50 times over the same shuffled test set and calculates each metric per batch of 128 rows. The ± is the spread of those per-batch numbers, not a spread across seeds or independent runs. The average of per-batch correlations also isn't the same as the correlation over the whole test set.
- **I don't know the split.** There's no code in this repo that makes `testing_data.parquet`, so I can't show how the split was done or whether the same molecules and proteins appear in both train and test. BindingDB has lots of near-duplicate compounds and repeated targets, so random splits tend to look better than they should.
- **The scaling doesn't match.** The test set was standardised with its own mean and standard deviation instead of the training ones.

So all this really shows is that the pipeline learns something. It isn't an honest estimate of how good the model is.

### What I'm going to do about it (not run yet)

I want to re-evaluate on a proper held-out set, with the scaler fitted on training data only. Each cell will report MAE, RMSE, Pearson and Spearman.

| Model | Random split | Scaffold split (unseen molecules) | Target-cluster split (unseen proteins) |
|---|---|---|---|
| Mean predictor | TBD | TBD | TBD |
| Molecule features only | TBD | TBD | TBD |
| Protein features only | TBD | TBD | TBD |
| Random forest / gradient boosting, all features | TBD | TBD | TBD |
| My MLP | TBD | TBD | TBD |

The molecule-only and protein-only rows matter most. If the molecule-only model does nearly as well as the full one, my network isn't really learning how molecules and proteins interact, it's just learning averages for each molecule or protein.

## What's in the repo

```text
Code/
  Fetching_data.ipynb              Loads BindingDB, writes 100,000-row CSV chunks, drops rows with no Ki
  Importing_db.ipynb               Quick look at BindingDB and ChEMBL (first rows only)
  Properties_application.ipynb     Calculates the features, pKi and Inaccurate; writes training_data.parquet
  Standardising.ipynb              Z-scores the feature columns
  Training.ipynb                   Trains the MLP (doesn't save the weights)
  Testing.ipynb                    Evaluates a saved model on the standardised test set
  Formating_Proteins.ipynb         Side experiment: I used the OpenAI API to tidy protein names for a UniProt
                                   lookup. Not used by the main pipeline, and it needs an API key
  Drug_Properties_Example.ipynb    One example per molecule descriptor (currently has bugs)
  Protein_Properties_Example.ipynb One example per protein descriptor
  Drawing_guide.ipynb              RDKit molecule drawing examples
  Training_Example.ipynb           UNRELATED: credit-card fraud detection (I'm going to delete this)
  Testing_Example.ipynb            UNRELATED: credit-card fraud detection (I'm going to delete this)
  model_weights1.pth               Weights for the 2048-1024-512-256 architecture

Properties.md                      My original feature plan
tables.md                          Notes on the ChEMBL tables (explored, not used for training)
website/public/predict.py          Single-pair prediction prototype (doesn't work properly, see below)
website/public/test.ts             Bun script that calls predict.py
```

**`predict.py` doesn't give meaningful predictions yet.** It feeds raw, unscaled features into a model that was trained on standardised ones, and I never saved the training means and standard deviations, so I can't fix it without regenerating them. It also loads the weights from a hardcoded absolute path. Ignore its output for now.

## Running it

There's no one-command setup yet, sorry. According to the notebook metadata I last ran the data-preparation notebooks on Python 3.14.5 and the standardising, training and testing notebooks on Python 3.14.7. I didn't record package versions, so this install line is unpinned:

```bash
pip install pandas numpy pyarrow scipy torch rdkit biopython propy3 matplotlib
# only for the side notebooks:
pip install pyopsin openai
```

You'll need to edit a few things before the notebooks run on another machine:

1. **`Fetching_data.ipynb`:** change the path to `BindingDB_All.tsv` (it's an absolute Windows path). It asks for a trial number and writes `chunks/chunk_{i}_{trial}.csv`.
2. **`Properties_application.ipynb`:** create a `Dataframes/` folder first. The processing cell is hardcoded to chunks 21–30 (`range(10)` with `i + 21`) and the concatenation cell reads chunks 0–30, so change both to match your chunk count. It reads columns by position (SMILES = 1, Ki = 2, sequence = 3), which only works because `Fetching_data.ipynb` adds an extra index column when it rewrites the chunks. It writes `training_data.parquet`.
3. **`Standardising.ipynb`:** writes `standardised_training_data.parquet` and the testing equivalent. It expects a `testing_data.parquet`, but nothing in this repo creates one.
4. **`Training.ipynb`:** trains the model. Add `torch.save(model.state_dict(), "model_weights1.pth")` if you want to keep the result.
5. **`Testing.ipynb`:** change the hardcoded test-file path and enter the weights trial number when it asks.

`Formating_Proteins.ipynb` needs an OpenAI API key. Put it in an environment variable and never commit it.

## What I still need to do

The plan is to find out whether the result actually holds up, write it down properly, and then call the project finished.

**Evaluation**
- [ ] Combine repeated measurements of the same (molecule, protein) pair
- [ ] Write code that makes one explicit, seeded train/validation/test split
- [ ] Add a scaffold split and a target-cluster split
- [ ] Fit the scaler on training data only, save it, and reuse it everywhere
- [ ] Use a validation set for the scheduler and for early stopping
- [ ] Fix `Testing.ipynb` so the metrics are calculated once over the whole test set, with the variation reported across seeds
- [ ] Run the baselines from the [table above](#what-im-going-to-do-about-it-not-run-yet) and fill it in

**Pipeline**
- [ ] Stop feeding `Inaccurate` to the model (drop the censored rows or handle them separately)
- [ ] Replace hardcoded paths, chunk ranges and `input()` prompts with one config file
- [ ] Put the feature code and the `Model` class in one shared module that training, testing and prediction all use
- [ ] Add `torch.save` to `Training.ipynb` and log the training loss
- [ ] Record the BindingDB release date, the row counts after each cleaning step, and the number of distinct molecules and proteins

**Repo**
- [ ] Fix `predict.py` (load the saved scaler, use a relative weights path), or delete `website/` and say prediction isn't supported
- [ ] Delete the two fraud-detection notebooks and fix the bugs in `Drug_Properties_Example.ipynb`
- [ ] Move the weights to a GitHub Release
- [ ] Commit a pinned `requirements.txt`
- [ ] Add a `LICENSE` file
- [ ] Update this README with the final results

## Where it stands

The workflow works from raw BindingDB rows to a trained network: the feature calculation, caching, standardising and PyTorch training all run. What I haven't shown is how well the model actually predicts. I can't reproduce my one result from this repo, and I haven't tested it on molecules or proteins it hasn't seen. I'd rather get an honest number, even if it turns out to be low, than keep a good-looking one I don't trust.

A convolutional or graph neural network would be a natural next step, but only once the splits and baselines above exist so the comparison is fair.

## Licence and credits

I haven't chosen a licence yet, so by default nobody has permission to reuse the code. I need to add a `LICENSE` file. BindingDB and ChEMBL have their own licence terms, so check them before sharing any data made from them. The project uses [RDKit](https://www.rdkit.org), [Biopython](https://biopython.org), [propy](https://pypi.org/project/propy3/) and [PyTorch](https://pytorch.org).
