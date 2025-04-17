import os
import time
import collections
import re
import math
import torchani
import torch
import argparse
import pandas as pd
import numpy as np
import mdtraj as md
from pathlib import Path
from esm.esmfold.v1.functions import NMR, ens_means, ens_stdevs
from typing import List, Optional
from torchani.aev import AEVComputer, ANIAngular, ANIRadial

class EntryPDB:
    """
    Class to hold the molecular data for one PDB.

    Attributes:
        species: list of atomic species
        coordinates: torch tensor of atomic coordinates
        res_idx: torch tensor of residue indices
        indices: a dictionary holding the indices for each atom type
    """
    SUPPORTED_SPECIES = ["H", "C", "N", "O", "S"]
    ALLOWED_HETFLAG = ["", "W", "H_DOD"]
    #JO: Here the structure is a instance of Bio.PDBParser
    def __init__(self, sequence, coordinates, device, interested_atypes):
        
        AA3to1 = {
            'ALA': 'A',
            'ARG': 'R',
            'ASN': 'N',
            'ASP': 'D',
            'CYS': 'C',
            'GLN': 'Q',
            'GLU': 'E',
            'GLY': 'G',
            'HIS': 'H',
            'ILE': 'I',
            'LEU': 'L',
            'LYS': 'K',
            'MET': 'M',
            'PHE': 'F',
            'PRO': 'P',
            'SER': 'S',
            'THR': 'T',
            'TRP': 'W',
            'TYR': 'Y',
            'VAL': 'V'
        }
        AA1to3 = {v: k for k, v in AA3to1.items()}
        atom_name_dict = {
            'A': [
                'N', 'CA', 'C', 'O', 'OXT', 'H', 'H2', 'H3', 'CB', 'HA', 'HB1', 'HB2', 'HB3',
                'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD',
                'PAD', 'PAD', 'PAD'
            ],
            'R': [
                'N', 'CA', 'C', 'O', 'OXT', 'H', 'H2', 'H3', 'CB', 'CG', 'CD', 'NE', 'CZ', 'NH1',
                'NH2', 'HA', 'HB2', 'HB3', 'HG2', 'HG3', 'HD2', 'HD3', 'HE', 'HH11', 'HH12',
                'HH21', 'HH22'
            ],
            'N': [
                'N', 'CA', 'C', 'O', 'OXT', 'H', 'H2', 'H3', 'CB', 'CG', 'OD1', 'ND2', 'HA',
                'HB2', 'HB3', 'HD21', 'HD22', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD',
                'PAD', 'PAD', 'PAD'
            ],
            'D': [
                'N', 'CA', 'C', 'O', 'OXT', 'H', 'H2', 'H3', 'CB', 'CG', 'OD1', 'OD2', 'HA',
                'HB2', 'HB3', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD',
                'PAD', 'PAD', 'PAD'
            ],
            'C': [
                'N', 'CA', 'C', 'O', 'OXT', 'H', 'H2', 'H3', 'CB', 'SG', 'HA', 'HB2', 'HB3', 'HG',
                'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD',
                'PAD', 'PAD'
            ],
            'Q': [
                'N', 'CA', 'C', 'O', 'OXT', 'H', 'H2', 'H3', 'CB', 'CG', 'CD', 'OE1', 'NE2', 'HA',
                'HB2', 'HB3', 'HG2', 'HG3', 'HE21', 'HE22', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD',
                'PAD', 'PAD'
            ],
            'E': [
                'N', 'CA', 'C', 'O', 'OXT', 'H', 'H2', 'H3', 'CB', 'CG', 'CD', 'OE1', 'OE2', 'HA',
                'HB2', 'HB3', 'HG2', 'HG3', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD',
                'PAD', 'PAD'
            ],
            'G': [
                'N', 'CA', 'C', 'O', 'OXT', 'H', 'H2', 'H3', 'HA2', 'HA3', 'PAD', 'PAD', 'PAD',
                'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD',
                'PAD', 'PAD', 'PAD'
            ],
            'H': [
                'N', 'CA', 'C', 'O', 'OXT', 'H', 'H2', 'H3', 'CB', 'CG', 'ND1', 'CE1', 'NE2',
                'CD2', 'HA', 'HB2', 'HB3', 'HD1', 'HE1', 'HD2', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD',
                'PAD', 'PAD'
            ],
            'I': [
                'N', 'CA', 'C', 'O', 'OXT', 'H', 'H2', 'H3', 'CB', 'CG1', 'CD1', 'CG2', 'HA',
                'HB', 'HG12', 'HG13', 'HD11', 'HD12', 'HD13', 'HG21', 'HG22', 'HG23', 'PAD',
                'PAD', 'PAD', 'PAD', 'PAD'
            ],
            'L': [
                'N', 'CA', 'C', 'O', 'OXT', 'H', 'H2', 'H3', 'CB', 'CG', 'CD1', 'CD2', 'HA',
                'HB2', 'HB3', 'HG', 'HD11', 'HD12', 'HD13', 'HD21', 'HD22', 'HD23', 'PAD', 'PAD',
                'PAD', 'PAD', 'PAD'
            ],
            'K': [
                'N', 'CA', 'C', 'O', 'OXT', 'H', 'H2', 'H3', 'CB', 'CG', 'CD', 'CE', 'NZ', 'HA',
                'HB2', 'HB3', 'HG2', 'HG3', 'HD2', 'HD3', 'HE2', 'HE3', 'HZ1', 'HZ2', 'HZ3',
                'PAD', 'PAD'
            ],
            'M': [
                'N', 'CA', 'C', 'O', 'OXT', 'H', 'H2', 'H3', 'CB', 'CG', 'SD', 'CE', 'HA', 'HB2',
                'HB3', 'HG2', 'HG3', 'HE1', 'HE2', 'HE3', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD',
                'PAD', 'PAD'
            ],
            'F': [
                'N', 'CA', 'C', 'O', 'OXT', 'H', 'H2', 'H3', 'CB', 'CG', 'CD1', 'CE1', 'CZ',
                'CE2', 'CD2', 'HA', 'HB2', 'HB3', 'HD1', 'HE1', 'HZ', 'HE2', 'HD2', 'PAD', 'PAD',
                'PAD', 'PAD'
            ],
            'P': [
                'N', 'CA', 'C', 'O', 'OXT', 'H', 'H2', 'H3', 'CB', 'CG', 'CD', 'HA', 'HB2', 'HB3',
                'HG2', 'HG3', 'HD2', 'HD3', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD',
                'PAD', 'PAD'
            ],
            'S': [
                'N', 'CA', 'C', 'O', 'OXT', 'H', 'H2', 'H3', 'CB', 'OG', 'HA', 'HB2', 'HB3', 'HG',
                'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD',
                'PAD', 'PAD'
            ],
            'T': [
                'N', 'CA', 'C', 'O', 'OXT', 'H', 'H2', 'H3', 'CB', 'OG1', 'CG2', 'HA', 'HB',
                'HG1', 'HG21', 'HG22', 'HG23', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD',
                'PAD', 'PAD', 'PAD'
            ],
            'W': [
                'N', 'CA', 'C', 'O', 'OXT', 'H', 'H2', 'H3', 'CB', 'CG', 'CD1', 'NE1', 'CE2',
                'CZ2', 'CH2', 'CZ3', 'CE3', 'CD2', 'HA', 'HB2', 'HB3', 'HD1', 'HE1', 'HZ2', 'HH2',
                'HZ3', 'HE3'
            ],
            'Y': [
                'N', 'CA', 'C', 'O', 'OXT', 'H', 'H2', 'H3', 'CB', 'CG', 'CD1', 'CE1', 'CZ', 'OH',
                'CE2', 'CD2', 'HA', 'HB2', 'HB3', 'HD1', 'HE1', 'HH', 'HE2', 'HD2', 'PAD', 'PAD',
                'PAD'
            ],
            'V': [
                'N', 'CA', 'C', 'O', 'OXT', 'H', 'H2', 'H3', 'CB', 'CG1', 'CG2', 'HA', 'HB',
                'HG11', 'HG12', 'HG13', 'HG21', 'HG22', 'HG23', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD',
                'PAD', 'PAD', 'PAD'
            ]
        }
        self.species = []
        self.coordinates = []
        self.res_idx = []
        self.unsupported = False
        self.indices = collections.defaultdict(lambda: [])
        self.str2int = torchani.utils.ChemicalSymbolsToInts(
            self.SUPPORTED_SPECIES
        )
        resname_dict = self.get_resname_dict()
        i = 0
        # No 0 or nan in the coordinates
        mask = ((coordinates != 0) & (~torch.isnan(coordinates))).any(dim=-1)  
        print("Does the mask take effect?",mask)
        for k in range(coordinates.shape[0]):  
            for j in range(coordinates.shape[1]):  
                if mask[k, j]:  
                    atype = atom_name_dict[sequence[k]][j]
                    if atype in interested_atypes:
                        self.indices[atype].append(i)
                    self.res_idx.append(resname_dict[AA1to3[sequence[k]]])
                    element = self.convert_element(atype[0])
                    self.species.append(element)
                    i += 1

        # Assume coordinates has a shape of [L, A, 3] and requires_grad=True
        L, A = coordinates.shape[:2]

        # 1. Reshape
        # Reshape the coordinates tensor to [L*A, 3]. This is a view operation, 
        # meaning it shares the underlying memory with the original tensor.
        reshaped_coordinates = coordinates.reshape(L * A, 3)

        # 2. Mask
        # Create a boolean mask indicating which rows in reshaped_coordinates 
        # contain at least one non-zero and non-NaN value.
        mask = ((reshaped_coordinates != 0) & (~torch.isnan(reshaped_coordinates))).any(dim=-1)

        # 3. Index select
        # Create an empty tensor with the same shape as reshaped_coordinates.
        # This is crucial for preserving the gradient information.
        self.coordinates = torch.empty_like(reshaped_coordinates)

        # Use index_select to copy the rows from reshaped_coordinates that correspond 
        # to True values in the mask into self.coordinates.
        self.coordinates = torch.index_select(reshaped_coordinates, 0, mask.nonzero(as_tuple=False).squeeze())

        # 4. Reshape
        # Add a new dimension at the beginning to make the shape [1, N, 3], 
        # where N is the number of selected rows.
        self.coordinates = self.coordinates.unsqueeze(0)
        
        print("if self.coordinates has gradient:",self.coordinates.requires_grad)
        
        self.res_idx = torch.tensor(self.res_idx, dtype=torch.long)
        self.species = self.str2int(self.species)
        for atype in self.indices:
            self.indices[atype] = torch.tensor(self.indices[atype], dtype=torch.long)
       
        self.res_idx = self.res_idx.to(device)
        self.species = self.species.to(device)
        for atype in self.indices:
            self.indices[atype] = self.indices[atype].to(device)

    @staticmethod
    def convert_element(element):
        if element in EntryPDB.SUPPORTED_SPECIES:
            return element
        if element == "D":
            return "H"
        raise ValueError("Bad element type")

    @staticmethod
    def get_resname_dict():
        string = '''ala arg asn asp cys glu gln gly his ile leu lys met phe pro ser thr trp tyr val hoh dod'''.split()
        otherlist = [item.upper() for item in string]
        return {name: j for j, name in enumerate(otherlist)}

class ChemicalShiftPredictor(torch.nn.Module):
    """
    Class to predict the chemical shift of the atoms in a molecular simulation.

    Attributes:
        models: a ModuleDict of models for each atom type
        aev_computer: computes atomic environment vectors (AEVs)
        ens_means: ensemble mean chemical shifts
        ens_stdevs: ensemble standard deviations of chemical shifts
    """

    def __init__(self, model_paths, interested_atypes):
        super(ChemicalShiftPredictor, self).__init__()
        self.interested_atypes = interested_atypes
        # Load all models for different atom types
        self.models = self._load_models(model_paths)
        for param in self.models.parameters():
            param.requires_grad = False
        radial_terms = ANIRadial.like_2x()
        angular_terms = ANIAngular.like_2x()
        self.aev_computer = AEVComputer(
            angular=angular_terms,
            radial=radial_terms,
            num_species=5,
            strategy="auto", #selects "cuaev" if CudaAEV extensions are available, pyaev if not
            neighborlist='cell_list' #del if too slow #could try use_cuaev_interface=True above
        )

        # Mean and Standard Deviation values for normalizing the output
        self.ens_means = ens_means
        self.ens_stdevs = ens_stdevs

    @staticmethod
    def _load_models(model_paths):
        # Create a dictionary to store all models by atom type
        models = {}
        for atom_type, paths in model_paths.items():
            models[atom_type] = torch.nn.ModuleList(
                [torch.load(path, map_location="cpu") for path in paths]
            )
        return torch.nn.ModuleDict(models)

    def forward(self, entry, batch_size=100):
        """
        Predicts the chemical shift for the given entry and batch size.

        Args:
            entry: Entry object containing simulation data
            batch_size: size of mini-batch for memory efficiency
        Returns:
            df_all: DataFrame of predicted chemical shifts for all atom types
        """
        # Timing
#        torch.cuda.synchronize()
#        initial_time = time.time()
        assert (
            entry.species.dim() == 1
        ), "Species should be only for a single frame even it's a trajectory."

        num_frames = entry.coordinates.shape[0]
        # compute AEVs and run models for all batches
        cs_all_batches = self._compute_aevs_and_run_models(
            entry, batch_size, num_frames
        )

        # Combine results from all batches and all atom types into a single DataFrame
        df_all = self._combine_all_frames(cs_all_batches, num_frames, entry)
        return df_all

    def _compute_aevs_and_run_models(self, entry, batch_size, num_frames):
        cs_all_batches = []
        # We need to do mini-batch processing because of memory limitation
        num_batches = math.ceil(num_frames / batch_size)

        for i in range(num_batches):
            # Compute AEVs for the current batch
            aevs_batch = self.compute_batch_aev(entry, i, batch_size)

            # Run the models to predict the chemical shifts
            chemical_shifts = self.run_models(aevs_batch, entry.res_idx, entry.indices)

            cs_all_batches.append(chemical_shifts)

        return cs_all_batches

    def _combine_all_frames(self, cs_all_batches, num_frames, entry):
        # cs_all_batches is a list of dictionary, each dictionary is for one atom type
        # we need to merge each atom type's dictionary into a single dictionary that have all frames
        # then we can convert it to a dataframe
        all_data = []
        all_data_type = []
        for atype in self.interested_atypes:
            cs_all_frames_atype = torch.cat(
                [batch[atype] for batch in cs_all_batches], dim=0
            )  # [num_frames, num_atoms]
            cs_std_all_frames_atype = torch.cat(
                [batch[atype+'_std'] for batch in cs_all_batches], dim=0
            )  # [num_frames, num_atoms]
            res_idx_atype = entry.res_idx.index_select(0, entry.indices[atype])

            # Adjust for a single frame
            if num_frames == 1:
                cs_all_frames_atype = cs_all_frames_atype.flatten()
                cs_std_all_frames_atype = cs_std_all_frames_atype.flatten()
            else:
                cs_all_frames_atype = cs_all_frames_atype.transpose(
                    0, 1
                )  # [num_atoms, num_frames]
                cs_std_all_frames_atype = cs_std_all_frames_atype.transpose(
                    0, 1
                )

            # Create DataFrame for current atom type and append to df_all
            data_type = {
                "ATOM_TYPE": [atype] * len(res_idx_atype),
                "RESIDUE_ID": res_idx_atype.flatten().tolist()
            }

            data = {
                "CHEMICAL_SHIFT": cs_all_frames_atype,
                "CHEMICAL_SHIFT_STD": cs_std_all_frames_atype
            }
            all_data.append(data)
            all_data_type.append(pd.DataFrame(data_type))

        combined_data = {
            key: torch.cat([batch[key] for batch in all_data], dim=0)
            for key in all_data[0]
        }
        return pd.concat(all_data_type, ignore_index=True), combined_data

    def compute_batch_aev(self, entry, i, batch_size):
        """
        Calculate a batch of AEVs, cuaev could only process single molecule at a time
        """
        # Define batch boundaries
        start = i * batch_size
        num_frames = entry.coordinates.shape[0]
        end = min((i + 1) * batch_size, num_frames)

        # Create batch of AEVs
        aevs_all = []
        for j in range(start, end):
            aevs = self.aev_computer(
                entry.species.unsqueeze(0), entry.coordinates[j].unsqueeze(0)
            )
            aevs_all.append(aevs)
        aevs_all = torch.cat(aevs_all, 0)
        return aevs_all

    def run_models(self, aevs, res_idx, indices):
        cs_all = {}
        num_frames = aevs.shape[0]

        for atype in self.interested_atypes:
            #print(f"Predicting {atype} chemical shifts")
            # todo register indices as buffer
            aevs_atype = aevs.index_select(1, indices[atype])
            res_idx_atype = res_idx.index_select(0, indices[atype])
            res_idx_atype_expanded = res_idx_atype.unsqueeze(-1).expand(
                num_frames, -1, -1
            )  # [num_frames, num_atoms, 1]

            inputs = torch.cat(
                [aevs_atype, res_idx_atype_expanded], -1
            )  # [num_frames, num_atoms, num_features]
            inputs = inputs.flatten(0, 1)  # [num_frames * num_atoms, num_features]
            print("if model input require grad:",inputs.requires_grad)
            models_outputs = []
            for model in self.models[atype]:
                model = model
                model.eval()
               # with torch.no_grad():
                outputs = model(inputs)  # [num_frames * num_atoms]
                models_outputs.append(outputs)
  
            models_avg = torch.stack(models_outputs).mean(0)  # [num_frames * num_atoms]
            print("if model output average requires grad?", models_avg.requires_grad)
            models_std = self.denorm_chemical_shifts(
                torch.stack(models_outputs), atype, res_idx_atype_expanded.flatten()
            ).std(0) #[num_frames * num_atoms]

            denormed_cs = self.denorm_chemical_shifts(
                models_avg, atype, res_idx_atype_expanded.flatten()
            )
            denormed_cs = denormed_cs.view(num_frames, -1)  # [num_frames, num_atoms]
            models_std = models_std.view(num_frames, -1)  # [num_frames, num_atoms]
            cs_all[atype] = denormed_cs
            cs_all[atype+'_std'] = models_std
        return cs_all

    def denorm_chemical_shifts(self, chemical_shift, atype, res_idx_atype):
        ens_stdevs_atype = self.ens_stdevs[atype].to(chemical_shift.device)
        ens_means_atype = self.ens_means[atype].to(chemical_shift.device)
        denormed_cs = (
            chemical_shift * ens_stdevs_atype[res_idx_atype]
            + ens_means_atype[res_idx_atype]
        )
        return denormed_cs

