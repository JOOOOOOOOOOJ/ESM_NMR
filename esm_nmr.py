#!/usr/bin/env python
from string import ascii_uppercase, ascii_lowercase
import hashlib, re, os
import numpy as np
import torch
from jax.tree_util import tree_map
import jax.numpy as jnp
import jax
from scipy.special import softmax
import gc
import random
import esm
from esm.esmfold.v1.functions import NMR
from esm.esmfold.v1.misc import output_to_pdbh_download 

version = "1" 
model_name = "esmfold_v0.model" if version == "0" else "esmfold.model"

def get_hash(x): return hashlib.sha1(x.encode()).hexdigest()
alphabet_list = list(ascii_uppercase+ascii_lowercase)

'''
JOJO: Sequences of interest
'''
#JO: KaiB (RS)
sequence = "GAMGRRLVLYVAGQTPKSLAAISNLRRICEENLPGQYEVEVIDLKQNPRLAKEHSIVAIPTLVRELPVPIRKIIGDLSDKEQVLVNLKMDME"

jobname = f"test" #@param {type:"string"}
jobname = re.sub(r'\W+', '', jobname)[:50]

sequence = re.sub("[^A-Z:]", "", sequence.replace("/",":").upper())
sequence = re.sub(":+",":",sequence)
sequence = re.sub("^[:]+","",sequence)
sequence = re.sub("[:]+$","",sequence)
copies = 1 #@param {type:"integer"}
if copies == "" or copies <= 0: copies = 1
sequence = ":".join([sequence] * copies)
num_recycles = 3 #@param ["0", "1", "2", "3", "6", "12", "24"] {type:"raw"}
chain_linker = 25

ID = jobname+"_"+get_hash(sequence)[:5]
seqs = sequence.split(":")
lengths = [len(s) for s in seqs]
length = sum(lengths)
print("length",length)

u_seqs = list(set(seqs))
if len(seqs) == 1: mode = "mono"
elif len(u_seqs) == 1: mode = "homo"
else: mode = "hetero"

if "model" not in dir() or model_name != model_name_:
  if "model" in dir():
    # delete old model from memory
    del model
    gc.collect()
    if torch.cuda.is_available():
      torch.cuda.empty_cache()
  print('nihao')
  model = esm.pretrained.esmfold_v1()
  print('nihao')
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = model.to(device)
  model.eval().cuda().requires_grad_(False)
  model_name_ = model_name

# optimized for Tesla T4
if length > 700:
  model.set_chunk_size(64)
else:
  model.set_chunk_size(128)

file_path = "chemicalshift_gs"

'''
These are for omega definition of loss function
'''
def extract_exp_data(file_path):
    data_dict = {}

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            row = line.strip().split()
            if len(row) >= 12: 
                col7 = row[6]  #Residue Number
                col9 = row[8]  #Atom Name
                col12 = row[11]  #Chemical Shift
               # residue_list = [5, 6, 7, 8, 10, 14, 17, 18, 20, 21, 23, 25, 26, 27, 28, 29, 30, 31, 32, 33, 35, 36, 38, 39, 40, 41, 42, 43, 44, 45, 46, 52, 53, 54, 72, 80, 81, 84, 87, 88] 
                residue_list = [i for i in range(1,93)]
                #Only use C-terminal residues' chemical shift changes (51-90)
                if col7.isdigit() and int(col7) in residue_list and col9 == 'N':
                    key = int(col7)  # 组合键
                    data_dict[key] = col12  # 存入字典

    return data_dict
#Chemical Shift from Ground State
file_path = "chemicalshift_gs"
result_dict = extract_exp_data(file_path)

#Change to tensor
chemicalshift_gs = {key: torch.tensor(float(value)).unsqueeze(0) for key, value in result_dict.items()}

print("chemicalshift_gs:",chemicalshift_gs)

exp = torch.cat(list(chemicalshift_gs.values()),dim=0)

exp = exp.to(device)

'''
These are for delta omega definition of loss function
'''
#result_dict = {54: 4.696083031347655, 72: 6.343315140039067, 80: 4.482551608007796, 87: 5.66843416289062, 88: 11.166122429589848}

#chemicalshift_gs = {key: torch.tensor(float(value)).unsqueeze(0) for key, value in result_dict.items()}

#print("chemicalshift_gs:",chemicalshift_gs)

#exp = torch.cat(list(chemicalshift_gs.values()),dim=0)

#exp = exp.to(device)

'''
Training Part
'''

from openfold.utils.feats import atom14_to_atom37
import torch.optim as optim
import torch.nn as nn
import sys
import torch.nn.functional as F
import pickle
'''
JOJO: Parse the output
'''

#JO: Get the pLDDT from output
def parse_output(output):
    plddt = output["plddt"][0, :, 1]
    plddt_t = output["plddt"][0, :, :]
    o = {
        "plddt": plddt[mask],
        "plddt_t": plddt_t
    }

    return o

#Define the loss function combining both plddt and chemical loss
def esm_loss(exp, pre, pre_std, output):
    cs_loss = torch.mean((exp - pre) ** 2 / pre_std ** 2)
    cs_total_loss = torch.sqrt(torch.mean(cs_loss**2))
    output_parsed = parse_output(output)
    plddt_logits = output_parsed["plddt_t"]
    plddt_ca = torch.mean(plddt_logits, dim=-1).mean()
    plddt_loss = -10 * torch.log(torch.sigmoid(plddt_ca - 50))
    total_loss = plddt_loss + cs_total_loss
    aux = (plddt_loss.item(), plddt_ca.item(), cs_total_loss.item())
    return total_loss, aux


torch.cuda.empty_cache()
torch.manual_seed(1)

esm_s, aa, B, L, residx, mask, num_recycles, linker_mask, chain_index = model.infer(sequence,
                    num_recycles=num_recycles,
                    chain_linker="X"*chain_linker,
                    residue_index_offset=512)

linear_transform = nn.Linear(esm_s.shape[-1], esm_s.shape[-1], bias=True, device="cuda")

#For record
#weight_data = torch.ones(esm_s.shape[-1], esm_s.shape[-1], device=linear_transform.weight.device)
weight_data = torch.eye(esm_s.shape[-1], esm_s.shape[-1], device=linear_transform.weight.device)

bias_data = torch.zeros_like(linear_transform.bias)

#initial_matrix = torch.ones(esm_s.shape[-1], esm_s.shape[-1], device=linear_transform.weight.device)
identity_matrix = torch.eye(esm_s.shape[-1], esm_s.shape[-1], device=linear_transform.weight.device)

linear_transform.weight = nn.Parameter(initial_matrix)

linear_transform.bias = nn.Parameter(torch.zeros_like(linear_transform.bias))

for param in linear_transform.parameters():
  param.requires_grad = True

losses = []
plddt_losses = []
cs_losses = []
n_steps = 500

optimizer = optim.Adam(linear_transform.parameters(), lr=0.000001)

for i in range(n_steps):

    esm_s_transformed = linear_transform(esm_s)

    output = model.infer_structure(
            esm_s_transformed, aa, B, L, residx, mask, num_recycles, linker_mask, chain_index
        )   

    aa_seq, coord, coord_h, cs_prediction_type, cs_prediction = model.output_to_pdb(output)
    
    indices = torch.tensor(list(chemicalshift_gs.keys())) - 1 
    
    print("indices:", indices)
    print(cs_prediction["CHEMICAL_SHIFT"])   
 
    pre = cs_prediction["CHEMICAL_SHIFT"][indices]

    pre_std = cs_prediction["CHEMICAL_SHIFT_STD"][indices]

    print(exp)
    print(pre)
    print(pre_std)

    '''
    ###For delta omega definition of loss 

    if i == 0:
        
        initial_pre = pre.clone().detach()
    
    print("pre:", pre)

    print("exp:", exp)

    pre = torch.abs(pre - initial_pre)
    '''
    loss, aux = esm_loss(exp, pre, pre_std, output)

    if not loss.requires_grad:
        raise ValueError("Loss does not require gradients. Check the computation graph.")

    loss.backward()

    #Check if the weight and bias have changed by backpropagation
    are_weights_equal = torch.equal(linear_transform.weight, weight_data)

    print(f"Are the tensors equal? {are_weights_equal}")

    are_bias_equal = torch.equal(linear_transform.bias, bias_data)

    print(f"Are the tensors equal? {are_bias_equal}")

    optimizer.step()
    losses.append(float(loss.item()))
    plddt_losses.append(float(aux[0]))
    cs_losses.append(float(aux[2]))
    optimizer.zero_grad()
    
    print(
        f"Step: {i+1} | loss: {loss.item():.4f} | best loss: {min(losses):.4f} | "
        f"plddt_loss: {aux[0]:.4f} | best plddt_loss: {min(plddt_losses):.4f} | plddt: {aux[1]:.4f} | "
        f"cs_loss: {aux[2]:.4f} | best cs_loss: {min(cs_losses):.4f}"
    )

    sys.stdout.flush()

    output_copy = {}

    for key, value in output.items():
        output_copy[key] = value.detach().clone()

    pdb_str = model.output_to_pdb_download(output_copy)[0]

    prefix = f"step_{i+1}_loss_{loss.item():.3f}"

    with open(f"{prefix}.pdb","w") as out:

      out.write(pdb_str)

    coord_h_copy = coord_h.detach().clone()

    output_to_pdbh_download(aa_seq, coord_h_copy, f"{prefix}_h.pdb") 
