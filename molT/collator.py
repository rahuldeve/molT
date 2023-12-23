from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import torch
from transformers.data.data_collator import DataCollatorMixin
from transformers.tokenization_utils import PreTrainedTokenizerBase

from molT.utils import TokenType


def torch_mask_atoms_bonds(
    input_ids, token_idxs, atom_mask, bond_mask, mlm_probability
):
    probability_matrix = torch.full_like(input_ids, mlm_probability, dtype=torch.float)
    probability_matrix.masked_fill_(~atom_mask, value=0.0)
    masked_atom_tokens = torch.bernoulli(probability_matrix).bool()

    # Not sure how to vectorize the loop
    for batch_idx in range(masked_atom_tokens.shape[0]):
        batch_token_idxs = token_idxs[batch_idx]
        batch_masked_atom_tokens = masked_atom_tokens[batch_idx]
        batch_atom_mask = atom_mask[batch_idx]

        # Do a comparison and mark duplicates
        initial_masked_atom_token_idsx = batch_token_idxs[batch_masked_atom_tokens]
        same_atom_idx_mask = (
            initial_masked_atom_token_idsx.unsqueeze(-1) == batch_token_idxs
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


def torch_mask_mol_descriptors(input_ids, mol_descriptor_mask, mlm_probability):
    probability_matrix = torch.full_like(input_ids, mlm_probability, dtype=torch.float)
    probability_matrix.masked_fill_(~mol_descriptor_mask, value=0.0)
    masked_descriptor_tokens = torch.bernoulli(probability_matrix).bool()
    return masked_descriptor_tokens


def torch_mask_target(input_ids, target_mask, mlm_probability):
    probability_matrix = torch.full_like(input_ids, mlm_probability, dtype=torch.float)
    probability_matrix.masked_fill_(~target_mask, value=0.0)
    masked_descriptor_tokens = torch.bernoulli(probability_matrix).bool()
    return masked_descriptor_tokens


@dataclass
class DataCollatorForMaskedMolecularModeling(DataCollatorMixin):
    tokenizer: PreTrainedTokenizerBase
    mlm_probability: float = 0.15
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
        token_idxs = batch["token_idxs"]
        input_ids = batch["input_ids"]
        token_type_ids = batch["token_type_ids"]
        labels = input_ids.clone()

        # mask atom and bonds
        atom_mask = token_type_ids == TokenType.ATOM
        bond_mask = token_type_ids == TokenType.BOND
        masked_atom_bond_tokens = torch_mask_atoms_bonds(
            input_ids, token_idxs, atom_mask, bond_mask, self.mlm_probability
        )

        # mask mol descriptor tokens
        mol_descriptor_mask = token_type_ids == TokenType.DESC
        masked_mol_descriptor_tokens = torch_mask_mol_descriptors(
            input_ids, mol_descriptor_mask, self.mlm_probability
        )

        # mask target tokens
        target_mask = token_type_ids == TokenType.TGT
        masked_target_tokens = torch_mask_target(
            input_ids, target_mask, self.mlm_probability
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
            masked_atom_bond_tokens
            | masked_mol_descriptor_tokens
            | masked_target_tokens
        )
        return batch

    def torch_call(
        self, examples: List[Union[List[int], Any, Dict[str, Any]]]
    ) -> Dict[str, Any]:
        # Handle dict or lists with proper padding and conversion to tensor.
        batch = self.tokenizer.pad(
            examples,
            return_tensors="pt",
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

        batch = self.torch_mask_tokens(batch)
        return batch


# # handle masking of duplicate atom tokens (suplicate here meaining having same token idxs)
# mask atoms
# atom_mask = token_type_ids == TokenType.ATOM
# bond_mask = token_type_ids == TokenType.BOND
# ignore_mask = ~atom_mask
# probability_matrix = torch.full_like(
#     input_ids, self.mlm_probability, dtype=torch.float
# )

# probability_matrix.masked_fill_(ignore_mask, value=0.0)
# masked_atom_tokens = torch.bernoulli(probability_matrix).bool()
#         masked_tokens = masked_atom_tokens.clone()
#         # Not sure how to vectorize the loop
#         for batch_idx in range(masked_tokens.shape[0]):
#             batch_token_idxs = token_idxs[batch_idx]
#             batch_masked_atom_tokens = masked_atom_tokens[batch_idx]
#             batch_token_types = token_type_ids[batch_idx]

#             batch_masked_atom_tokens = (
#                 batch_token_idxs[batch_masked_atom_tokens].unsqueeze(-1)
#                 == batch_token_idxs
#             ).any(dim=0) & (batch_token_types == TokenType.ATOM)

#             # mask all outgoing bonds from masked atoms; since every atom token is followed up with a
#             # bond token that is attached to the atom token, we can use torch.roll to mask the attached bond
#             masked_tokens[
#                 batch_idx
#             ] = batch_masked_atom_tokens | batch_masked_atom_tokens.roll(1) & ~(
#                 batch_token_idxs == TokenType.SPECIAL
#             )
