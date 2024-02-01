from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import torch
from transformers.data.data_collator import DataCollatorMixin
from transformers.tokenization_utils import PreTrainedTokenizerBase

from .utils import TokenType
from .config import MolTConfig


def torch_mask_atoms_bonds(
    input_ids, token_ids, atom_mask, bond_mask, atom_bond_mask_probability
):
    probability_matrix = torch.full_like(
        input_ids, atom_bond_mask_probability, dtype=torch.float
    )
    probability_matrix.masked_fill_(~atom_mask, value=0.0)
    masked_atom_tokens = torch.bernoulli(probability_matrix).bool()

    # Not sure how to vectorize the loop
    for batch_idx in range(masked_atom_tokens.shape[0]):
        batch_token_ids = token_ids[batch_idx]
        batch_masked_atom_tokens = masked_atom_tokens[batch_idx]
        batch_atom_mask = atom_mask[batch_idx]

        # Do a comparison and mark duplicates
        initial_masked_atom_token_idsx = batch_token_ids[batch_masked_atom_tokens]
        same_atom_idx_mask = (
            initial_masked_atom_token_idsx.unsqueeze(-1) == batch_token_ids
        ).any(dim=0)

        # above tensor may be marking bonds as duplicates; enforce atom mask to remove them
        same_atom_idx_mask = same_atom_idx_mask & batch_atom_mask
        masked_atom_tokens[batch_idx] = same_atom_idx_mask

    # mask all outgoing bonds from masked atoms; since every atom token is followed up with a
    # bond token that is attached to the atom token, we can use torch.roll to mask the attached bond
    masked_tokens = masked_atom_tokens | masked_atom_tokens.roll(1, dims=[-1])
    # some special tokens might have been masked; enforce atom and bond masks to remove them
    masked_tokens = masked_tokens & (atom_mask | bond_mask)

    return masked_tokens


def torch_mask_mol_features(input_ids, mol_feature_mask, mol_feature_mask_probability):
    probability_matrix = torch.full_like(
        input_ids, mol_feature_mask_probability, dtype=torch.float
    )
    probability_matrix.masked_fill_(~mol_feature_mask, value=0.0)
    masked_feature_tokens = torch.bernoulli(probability_matrix).bool()
    return masked_feature_tokens


def torch_mask_target(input_ids, target_mask, target_mask_probability):
    probability_matrix = torch.full_like(
        input_ids, target_mask_probability, dtype=torch.float
    )
    probability_matrix.masked_fill_(~target_mask, value=0.0)
    masked_target_tokens = torch.bernoulli(probability_matrix).bool()
    return masked_target_tokens


@dataclass
class DataCollatorForMaskedMolecularModeling(DataCollatorMixin):
    tokenizer: PreTrainedTokenizerBase
    config: MolTConfig
    pad_to_multiple_of: Optional[int] = None
    tf_experimental_compile: bool = False
    return_tensors: str = "pt"

    def __post_init__(self):
        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. "
                "You should pass `mlm=False` to train on causal language modeling instead."
            )

    def torch_mask_tokens(self, batch):
        token_ids = batch["token_ids"]
        input_ids = batch["input_ids"]
        token_type_ids = batch["token_type_ids"]
        labels = input_ids.clone()

        # mask atom and bonds
        atom_mask = token_type_ids == TokenType.ATOM
        bond_mask = token_type_ids == TokenType.BOND
        masked_atom_bond_tokens = torch_mask_atoms_bonds(
            input_ids,
            token_ids,
            atom_mask,
            bond_mask,
            self.config.atom_bond_mask_probability,
        )

        # mask mol descriptor tokens
        mol_feature_mask = token_type_ids == TokenType.FEAT
        masked_mol_feature_tokens = torch_mask_mol_features(
            input_ids, mol_feature_mask, self.config.molecule_feature_mask_probability
        )

        # mask target tokens
        target_mask = token_type_ids == TokenType.TGT
        masked_target_tokens = torch_mask_target(
            input_ids, target_mask, self.config.target_mask_probability
        )

        # We will use the typical mlm type masking/randomization for input_ids of atoms and bonds
        # Masking of mol descriptor tokens as well as atom/bond property tensors should be handled differently
        # For the sake of simplicity, the tokenizer will return a mask containing all the masked token positions
        # The actual masking will be handled inside the specialized class for masked molecule modelling
        labels[~masked_atom_bond_tokens] = -100  # We only compute loss on masked tokens

        # Skipping randomization and masking all the tokens for now. When randomizing, an atom
        # might get replaced with a bond. This is probably simple for the model to figure out
        # Furthermore, there are only a handful of atoms in the vocab. Randomization is used to
        # prevent overfitting but with such a small number of atoms, this might not be a problem
        input_ids[masked_atom_bond_tokens] = self.tokenizer.convert_tokens_to_ids(  # type: ignore
            self.tokenizer.mask_token
        )

        # Skipping 80% masking
        # # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        # indices_replaced = (
        #     torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_atom_bond_tokens
        # )

        # input_ids[indices_replaced] = self.tokenizer.convert_tokens_to_ids(
        #     self.tokenizer.mask_token
        # )

        # Skipping randomization for now; not sure how to randomize atom/bond props
        # # 10% of the time, we replace masked input tokens with random word
        # indices_random = (
        #     torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
        #     & masked_atom_bond_tokens
        #     & ~indices_replaced
        # )
        # random_words = torch.randint(
        #     len(self.tokenizer), labels.shape, dtype=torch.long
        # )
        # input_ids[indices_random] = random_words[indices_random]

        batch["input_ids"] = input_ids
        batch["labels"] = labels
        batch["mm_mask"] = (
            masked_atom_bond_tokens | masked_mol_feature_tokens | masked_target_tokens
        )
        return batch
    
    def scatter_sparse(self, mask, values):
        B = mask.shape[0]
        L = mask.shape[1]
        D = values.shape[-1]
        # Do below only if ndims of mask != values
        # TODO: Handle for non batched entries
        mask = mask.unsqueeze(-1).expand(-1, -1, D)
        I = torch.argwhere(mask).T
        V = values[~torch.isnan(values)].flatten()
        sparse_rep = torch.sparse_coo_tensor(
            I, V, size=(B, L, D)
        ).coalesce()
        return sparse_rep

    
    def convert_to_sparse(self, batch):
        token_type_ids = batch['token_type_ids']

        atom_props = batch['atom_props']
        atom_mask = token_type_ids == TokenType.ATOM
        batch['atom_props'] =  self.scatter_sparse(atom_mask, atom_props)
        
        bond_props = batch['bond_props']
        bond_mask = token_type_ids == TokenType.BOND
        batch['bond_props'] =  self.scatter_sparse(bond_mask, bond_props)

        batch['pos_embed_ids'] = self.scatter_sparse(atom_mask | bond_mask, batch['pos_embed_ids'])

        return batch

    def torch_call(
        self, examples: List[Union[List[int], Any, Dict[str, Any]]]
    ) -> Dict[str, Any]:
        # Handle dict or lists with proper padding and conversion to tensor.
        batch = self.tokenizer.pad(
            examples,  # type: ignore
            return_tensors="pt",
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

        batch = self.torch_mask_tokens(batch)
        batch = self.convert_to_sparse(batch)
        return batch
