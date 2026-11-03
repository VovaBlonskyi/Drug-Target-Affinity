# Drug–Target Affinity Prediction

This project explores whether a neural network can predict drug–target binding affinity from molecular structures and protein sequences. I built a notebook-based workflow to prepare binding data, calculate features for both molecules and proteins, train a PyTorch model, and evaluate its predictions.

It is an experimental project, not a validated drug-discovery tool. The notebooks and results document the current approach and provide a starting point for more rigorous testing.

## Approach

Each example combines a ligand, represented by its SMILES string, with a protein amino-acid sequence. The feature pipeline calculates descriptors for both and uses them to predict pKi.

**Molecule features** include molecular weight, LogP, hydrogen-bond donors and acceptors, TPSA, aromatic ring counts, bond and conjugation information, and a 2,048-bit Morgan fingerprint.

**Protein features** include amino-acid and dipeptide composition, CTD, PAAC/APAAC, autocorrelation descriptors, and sequence properties such as molecular weight, aromaticity, isoelectric point, instability index, and GRAVY.

Ki values in nM are converted to pKi using `pKi = 9 - log10(Ki)`. The feature-generation notebook also marks measurements with inequality-style values using an `Inaccurate` flag.

## Workflow

The project developed from exploring binding data to building a feature pipeline and training a regression model. `Code/Fetching_data.ipynb` and `Code/Importing_db.ipynb` show data exploration and preparation. The main feature engineering is in `Code/Properties_application.ipynb`: it processes records in chunks, caches repeated molecule and protein descriptors, and assembles the feature dataset. `Code/Training.ipynb` trains a feed-forward PyTorch network, and `Code/Testing.ipynb` evaluates its predictions.

The repository also includes [`Properties.md`](Properties.md), which records the feature plan, and [`tables.md`](tables.md), a reference to ChEMBL database tables explored during the project.

## Recorded evaluation results

One run in `Code/Testing.ipynb` reports the following average metrics and standard deviations:

| Metric | Result |
|---|---:|
| Mean absolute error (MAE) | 0.4868 ± 0.0447 |
| Root mean squared error (RMSE) | 0.6964 ± 0.0836 |
| Pearson correlation | 0.8977 ± 0.0272 |
| Spearman correlation | 0.8886 ± 0.0269 |

These are results recorded in the notebook, not independently reproduced benchmarks. The current evaluation does not document a strict scaffold-based or target-based split, so it does not establish performance on entirely unseen compounds or proteins. The split strategy and evaluation procedure should be strengthened before drawing conclusions about generalisation.

## Repository structure

```text
Code/
  Fetching_data.ipynb               Reads and chunks binding data
  Importing_db.ipynb                Explores BindingDB and ChEMBL data
  Properties_application.ipynb      Builds molecule and protein features
  Standardising.ipynb                Standardises generated datasets
  Training.ipynb                     Trains the affinity model
  Testing.ipynb                      Evaluates model predictions
  Drug_Properties_Example.ipynb      Examples of molecule descriptors
  Protein_Properties_Example.ipynb   Examples of protein descriptors
  Drawing_guide.ipynb                RDKit molecule drawing examples
  Training_Example.ipynb             Example training notebook
  Testing_Example.ipynb              Example testing notebook
  model_weights1.pth                 Saved model weights

Properties.md                        Feature notes
tables.md                            ChEMBL table reference
website/public/predict.py            Experimental single-pair prediction script
website/public/test.ts               Small script that calls the predictor
```

## Running the notebooks

This repository is notebook-based and does not currently include a dependency file or a complete, one-command setup. The required source datasets are not included, and some notebooks use local machine-specific paths. Before running the workflow, install the packages used by the notebooks, obtain the data, and update paths for your environment. Dependencies used include pandas, NumPy, PyTorch, RDKit, Biopython, `propy`, SciPy, and PyArrow.

The general order is:

1. Explore and prepare data with `Fetching_data.ipynb` and `Importing_db.ipynb`.
2. Generate descriptors and assemble the dataset with `Properties_application.ipynb`.
3. Standardise the data with `Standardising.ipynb`.
4. Train with `Training.ipynb` and evaluate with `Testing.ipynb`.

This describes the intended workflow, but paths, expected files, and preprocessing assumptions may need adjustment. `website/public/predict.py` is an early prediction prototype and also requires its model path and environment to be configured locally.

## Limitations and next steps

The current notebooks are exploratory and have assumptions that make the results difficult to reproduce on another machine. Important next steps include adding dependency management, replacing hardcoded paths, defining reproducible train/validation/test splits, and evaluating with scaffold- and target-based holdouts. Feature scaling should be fitted on training data and then reused for validation and test data; the handling of censored measurements also needs clearer documentation.

## Conclusion

This project brought together molecular descriptor generation, protein sequence features, data processing, and neural-network regression in an end-to-end affinity-prediction workflow. The recorded results are promising as an initial experiment, but further work is needed to make the pipeline reproducible and to measure its ability to generalise to unseen compounds and targets.

For future projects, I would also like to investigate convolutional neural networks (CNNs) and compare them with this descriptor-based approach. A CNN may be useful when the input representation contains local patterns, but whether it is a better choice depends on the data and representation. A fair comparison would help determine whether it offers a real advantage for this problem.
