from string import ascii_uppercase, ascii_lowercase
import hashlib, re, os
import numpy as np
import torch
from jax.tree_util import tree_map
import matplotlib.pyplot as plt
from scipy.special import softmax
import gc
import random
import esm
###JOJO: Random Number and Random Seed Generator
###JOJO: Generate a random number between 1 and 10000 (inclusively)
random_int = random.randint(1, 10000)
torch.cuda.manual_seed(random_int)
def parse_output(output):
  pae = (output["aligned_confidence_probs"][0] * np.arange(64)).mean(-1) * 31
  plddt = output["plddt"][0,:,1]

  bins = np.append(0,np.linspace(2.3125,21.6875,63))
  sm_contacts = softmax(output["distogram_logits"],-1)[0]
  sm_contacts = sm_contacts[...,bins<8].sum(-1)
  xyz = output["positions"][-1,0,:,1]
  mask = output["atom37_atom_exists"][0,:,1] == 1
  o = {"pae":pae[mask,:][:,mask],
       "plddt":plddt[mask],
       "sm_contacts":sm_contacts[mask,:][:,mask],
       "xyz":xyz[mask]}
  return o

def get_hash(x): return hashlib.sha1(x.encode()).hexdigest()
alphabet_list = list(ascii_uppercase+ascii_lowercase)

###JOJO: Generate the mask sequence from NMR data
NMR_mask = ".AAAAA......AA.AAAAAAAAAAAAAAAAAAAAAA.AA.AAAAAAAAAAAAAAA.AAAAAAAA.AAA.A..AA.AA.AA......AA.AA.AAAAA"

def generate_masking_pattern(input_string):
    """
    Generate a masking pattern tensor from the input string.

    Args:
        input_string (str): A string where 'A' indicates no mask (0) and '.' indicates mask (1).

    Returns:
        torch.Tensor: A binary tensor where masked positions are 1 and unmasked positions are 0.
    """
    # Convert the string to a list of binary values
    mask = [1 if char == '.' else 0 for char in input_string]
    # Convert the list to a torch.Tensor
    mask_tensor = torch.tensor(mask, dtype=torch.int)
    return mask_tensor

for seed in range(0, 1):

  jobname = f"test" #@param {type:"string"}
  jobname = re.sub(r'\W+', '', jobname)[:50]


  ###JOJO: Flavodoxin
  sequence = "SLEDRGPMYDDPTLPEGWTRKLKQRKSGRSAGKYDVYLINPQGKAFRSKVELIAYFEKVGDTSLDPNDFDFTVTGRGSPSRREQKPPKKPKSPKATSH" #@param {type:"string"}
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
    model.train().cuda().requires_grad_(False)

  # optimized for Tesla T4
  if length > 700:
    model.set_chunk_size(64)
  else:
    model.set_chunk_size(128)

  '''
  JOJO: unsqueeze(0) here is to make sure the shape of NMR masking
  sequence here is the same as the sequence we want to predict. And
  the shape should be like (types, sequences)
  '''
  NMR_mask_pattern = generate_masking_pattern(NMR_mask).unsqueeze(0)
  torch.cuda.empty_cache()
  torch.manual_seed(seed)
  model.train(True)
  output = model.infer(sequence,
                      masking_pattern=NMR_mask_pattern,
                      num_recycles=num_recycles,
                      chain_linker="X"*chain_linker,
                      residue_index_offset=512)
  pdb_str = model.output_to_pdb(output)[0]
  output = tree_map(lambda x: x.cpu().numpy(), output)
  ptm = output["ptm"][0]
  plddt = output["plddt"][0,...,1].mean()
  O = parse_output(output)
  print(f'ptm: {ptm:.3f} plddt: {plddt:.3f}')
  os.system(f"mkdir -p {ID}")
  prefix = f"{ID}/ptm{ptm:.3f}_r{num_recycles}_default_{seed}"
  # np.savetxt(f"{prefix}.pae.txt",O["pae"],"%.3f")
  with open(f"{prefix}.pdb","w") as out:
    out.write(pdb_str)

  #Plot confidence plot
  #@title plot confidence (optional)

  dpi = 100 #@param {type:"integer"}

  def plot_ticks(Ls):
    Ln = sum(Ls)
    L_prev = 0
    for L_i in Ls[:-1]:
      L = L_prev + L_i
      L_prev += L_i
      plt.plot([0,Ln],[L,L],color="black")
      plt.plot([L,L],[0,Ln],color="black")
    ticks = np.cumsum([0]+Ls)
    ticks = (ticks[1:] + ticks[:-1])/2
    plt.yticks(ticks,alphabet_list[:len(ticks)])

  def plot_confidence(O, Ls=None, dpi=100):
    if "lm_contacts" in O:
      plt.figure(figsize=(20,4), dpi=dpi)
      plt.subplot(1,4,1)
    else:
      plt.figure(figsize=(15,4), dpi=dpi)
      plt.subplot(1,3,1)

    plt.title('Predicted lDDT')
    plt.plot(O["plddt"])
    if Ls is not None:
      L_prev = 0
      for L_i in Ls[:-1]:
        L = L_prev + L_i
        L_prev += L_i
        plt.plot([L,L],[0,100],color="black")
    plt.xlim(0,O["plddt"].shape[0])
    plt.ylim(0,100)
    plt.ylabel('plDDT')
    plt.xlabel('position')
    plt.subplot(1,4 if "lm_contacts" in O else 3,2)

    plt.title('Predicted Aligned Error')
    Ln = O["pae"].shape[0]
    plt.imshow(O["pae"],cmap="bwr",vmin=0,vmax=30,extent=(0, Ln, Ln, 0))
    if Ls is not None and len(Ls) > 1: plot_ticks(Ls)
    plt.colorbar()
    plt.xlabel('Scored residue')
    plt.ylabel('Aligned residue')

    if "lm_contacts" in O:
      plt.subplot(1,4,3)
      plt.title("contacts from LM")
      plt.imshow(O["lm_contacts"],cmap="Greys",vmin=0,vmax=1,extent=(0, Ln, Ln, 0))
      if Ls is not None and len(Ls) > 1: plot_ticks(Ls)
      plt.subplot(1,4,4)
    else:

      plt.subplot(1,3,3)
    plt.title("contacts from Structure Module")
    plt.imshow(O["sm_contacts"],cmap="Greys",vmin=0,vmax=1,extent=(0, Ln, Ln, 0))
    if Ls is not None and len(Ls) > 1: plot_ticks(Ls)
    return plt

  plot_confidence(O, Ls=lengths, dpi=dpi)
  # plt.savefig(f'{prefix}.png',bbox_inches='tight')
  plt.show()
