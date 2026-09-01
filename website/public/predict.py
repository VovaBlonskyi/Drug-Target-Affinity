import sys
import numpy as np
from rdkit import Chem,DataStructs
from rdkit.Chem import  Descriptors, Crippen, Lipinski, MolSurf
from rdkit.Chem import rdFingerprintGenerator
from propy.PyPro import GetProDes
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import torch 
import torch.nn as nn
import pandas as pd

SMILES = sys.argv[1]
sequence = sys.argv[2]

def protein_properties(sequence):

    sequence = sequence.upper()
    
    allowed = set("ARNDCEQGHILKMFPSTWYV")
     
    if set(sequence) - allowed:
        return False
 
    des = GetProDes(sequence)
    analysis = ProteinAnalysis(sequence)
  
    protein_features = {}

    
    AAC = des.GetAAComp()
    DPC = des.GetDPComp()
    CTD = des.GetCTD()
    PAAC = des.GetPAAC(lamda=10, weight=0.05)
    APAAC = des.GetAPAAC(lamda=10, weight=0.05)
    moreau_broto = des.GetMoreauBrotoAuto()
    moran = des.GetMoranAuto()
    geary = des.GetGearyAuto()
    Mr = analysis.molecular_weight()
    aromaticity = analysis.aromaticity()
    IP = analysis.isoelectric_point()
    II = analysis.instability_index()
    GRAVY = analysis.gravy()

    protein_features.update(AAC)
    protein_features.update(DPC)
    protein_features.update(CTD)
    protein_features.update(PAAC)
    protein_features.update(APAAC)
    protein_features.update(moreau_broto)
    protein_features.update(moran)
    protein_features.update(geary)
    protein_features.update({"Mr" : Mr,
                            "Aromaticity" : aromaticity,
                            "Isoelectric point" : IP,
                            "Instability index" : II,
                            "GRAVY" : GRAVY})
 
    return protein_features

def drug_properties(smiles):

    smiles = str(smiles)
        
    if "|" in smiles:
        smiles = smiles.split("|")[0].strip()
    
    mol = Chem.MolFromSmiles(smiles)
    
    if mol is None:
        return False
    #Bond features

    single = 0
    double = 0
    triple = 0
    aromatic = 0
    conjugated = 0
    total_bonds = mol.GetNumBonds()
    ringBonds = 0
    
    for bond in mol.GetBonds():
        if bond.IsInRing():
            ringBonds +=1
        btype = bond.GetBondType()
        if btype == Chem.rdchem.BondType.SINGLE:
            single += 1
        elif btype == Chem.rdchem.BondType.DOUBLE:
            double += 1
        elif btype == Chem.rdchem.BondType.TRIPLE:
            triple += 1
        elif btype == Chem.rdchem.BondType.AROMATIC:
            aromatic += 1
        
        if bond.GetIsConjugated():
            conjugated += 1

    
   #Hybridization
    
    SP3 = 0
    SP2 = 0
    SP = 0
    
    for i in mol.GetAtoms():
        hybridization = i.GetHybridization()
        if hybridization == Chem.rdchem.HybridizationType.SP3:
            SP3 +=1
        elif hybridization == Chem.rdchem.HybridizationType.SP2:
            SP2 +=1
        elif hybridization == Chem.rdchem.HybridizationType.SP:
            SP +=1


    #formalCharge
    
    overall_charge = 0
    for i in mol.GetAtoms():
        overall_charge += i.GetFormalCharge()


    #Aromatic Cycles
    heteroAromatic = Lipinski.NumAromaticHeterocycles(mol)
    carboAromatic = Lipinski.NumAromaticCarbocycles(mol)

    
    #TPSA
    
    Tpsa = MolSurf.TPSA(mol)


    #Hydrogen Bonds
    
    Hdonors = Lipinski.NumHDonors(mol)
    Hacceptors = Lipinski.NumHAcceptors(mol)

    
    #logP
    
    logp = Crippen.MolLogP(mol)

    
    #Mr
    
    weight = Descriptors.ExactMolWt(mol)

    
    #Morgan fingerprint
    morgan_fp_gen = rdFingerprintGenerator.GetMorganGenerator(includeChirality=True, radius=2, fpSize=2048,countSimulation=False)
    
    fp = morgan_fp_gen.GetFingerprint(mol)
    arr = np.zeros((2048,), dtype=int)
    
    DataStructs.ConvertToNumpyArray(fp, arr)
    
    output = {"Molecular weight": weight,
             "LogP": logp,
             "Hydrogen bond Donors": Hdonors,
             "Hydrogen bond Acceptors": Hacceptors,
             "TPSA": Tpsa,
             "Aromatic Heterocycles": heteroAromatic,
             "Aromatic Carbocycles": carboAromatic,
             "Overall charge": overall_charge,
             'SP3' : SP3,
             'SP2' : SP2,
             'SP' : SP,
             "Single bonds": single,
             "Double bonds": double,
             "Triple bonds": triple,
             "Aromatic bonds": aromatic,
             "Rotatable bonds": Lipinski.NumRotatableBonds(mol),
             "Conjugated bonds": conjugated,
             "Conjugation ratio": conjugated / total_bonds if total_bonds else 0,
             "Bonds in ring": ringBonds
            }
    for i, bit in enumerate(arr):
        output[f"drug_FP_{i}"] = int(bit)
    return output

class Model(nn.Module):
  def __init__(self, input_size, layerSize = [ 2048, 1024, 512, 256]):
    super(Model, self).__init__()
    layers = []
    c = input_size
      
    for s in layerSize:
        layers.append(nn.Linear(c, s))
        layers.append(nn.BatchNorm1d(s))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.1))
        c = s
    layers.append(nn.Linear(c, 1))
    self.model = nn.Sequential(*layers)

  def forward(self, x):
    return self.model(x)

SMILES_properties = drug_properties(SMILES)
sequence_properties = protein_properties(sequence)

properties_unformatted = {**SMILES_properties, **sequence_properties, 'Inaccurate': 0}
df = pd.DataFrame([properties_unformatted])
properties = torch.tensor( df.to_numpy(), dtype=torch.float32 )



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Model(properties.shape[1]).to(device)

model.load_state_dict(torch.load(r"C:\Users\Volodymyr Blonskyi\Chem_AI\Drug-Target\Code\model_weights1.pth", map_location=device))
model.eval()

prediction = model(properties).squeeze().cpu()

print(prediction.item())