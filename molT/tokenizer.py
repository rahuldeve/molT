from collections import defaultdict
from itertools import chain
from typing import Dict, List, Optional, Union

import numpy as np
from rdkit import Chem
from torch import TensorType
from transformers import BatchEncoding, PreTrainedTokenizerBase
from transformers.tokenization_utils_base import (
    EncodedInput,
    EncodedInputPair,
    PaddingStrategy,
    PreTokenizedInput,
    PreTokenizedInputPair,
    TextInput,
    TextInputPair,
    TruncationStrategy,
)

from .config import MolTConfig
from .utils import TokenType, pack_atom_properties, pack_bond_properties


def get_lp_embeddings(adj_mat, k, flip_signs=True):
    A = adj_mat
    D = np.diag(A.sum(axis=-1))
    N = np.linalg.pinv(D) ** 0.5
    ID = np.eye(A.shape[0])
    L = ID - N @ A @ N

    eig_val, eig_vec = np.linalg.eig(L)
    # eig_val = np.sort(np.abs(np.real(eig_val)))

    if eig_vec.shape[-1] < k:
        eig_vec = np.pad(
            eig_vec, ((0, 0), (0, k - eig_vec.shape[-1])), constant_values=(0.0,)
        )
        eig_vec = np.real(eig_vec)
    else:
        eig_vec = np.real(eig_vec[:, :k])

    if flip_signs:
        rand_sign = 2 * (np.random.rand(k) > 0.5) - 1
        eig_vec = rand_sign * eig_vec

    return eig_vec


def get_atom_properties(mol):
    atoms = []
    in_ring = []
    charge = []
    hybridization = []
    chirality = []
    idxs = []
    for atom in mol.GetAtoms():
        idxs.append(atom.GetIdx())
        atoms.append(atom.GetSymbol())
        in_ring.append(atom.IsInRing())
        charge.append(atom.GetFormalCharge())
        hybridization.append(atom.GetHybridization().real)
        chirality.append(atom.GetChiralTag())

    properties = {
        "prop_atom_in_ring": np.array(in_ring, dtype=np.uint) + 1,
        "prop_atom_charge": np.array(charge, dtype=int) + 2,
        "prop_atom_hybridization": np.array(hybridization, dtype=int) + 1,
        "prop_atom_chirality": np.array(chirality, dtype=int) + 1,
    }

    return atoms, properties


def get_bond_properties(mol):
    bonds = []
    aromatic = []
    conjugated = []
    stereo = []
    edge_list = np.zeros((mol.GetNumBonds(), 2), dtype=np.uint)
    for bond in mol.GetBonds():
        idx = bond.GetIdx()
        bonds.append(bond.GetBondType().name)
        aromatic.append(bond.GetIsAromatic())
        conjugated.append(bond.GetIsConjugated())
        stereo.append(bond.GetStereo().real)

        edge_list[idx, 0] = bond.GetBeginAtomIdx()
        edge_list[idx, 1] = bond.GetEndAtomIdx()

    # adding 1 to make space for null embeds
    properties = {
        "prop_bond_aromatic": np.array(aromatic, dtype=np.uint) + 1,
        "prop_bond_conjugated": np.array(conjugated, dtype=np.uint) + 1,
        "prop_bond_stereo": np.array(stereo, dtype=int) + 1,
    }

    return bonds, edge_list, properties


# Test this function
def generate_tokens_atom_bond_mask_pos_embed_ids(edge_list):
    tokens = []
    atom_mask = []
    pos_embed_ids = []
    atoms_seen = set()
    for bond_idx in range(edge_list.shape[0]):
        st_atom_idx = edge_list[bond_idx, 0]
        en_atom_idx = edge_list[bond_idx, 1]

        if st_atom_idx not in atoms_seen:
            tokens.append(st_atom_idx)
            atom_mask.append(True)
            pos_embed_ids.append([st_atom_idx, st_atom_idx])
            atoms_seen.add(st_atom_idx)

        tokens.append(bond_idx)
        atom_mask.append(False)
        pos_embed_ids.append([st_atom_idx, en_atom_idx])

        if en_atom_idx not in atoms_seen:
            tokens.append(en_atom_idx)
            atom_mask.append(True)
            atoms_seen.add(en_atom_idx)
            pos_embed_ids.append([en_atom_idx, en_atom_idx])

    tokens = np.array(tokens, dtype=np.uint)
    atom_mask = np.array(atom_mask, dtype=bool)
    bond_mask = ~atom_mask
    pos_embed_ids = np.stack(pos_embed_ids, axis=0)
    return tokens, atom_mask, bond_mask, pos_embed_ids


class MolTTokenizer(PreTrainedTokenizerBase):
    model_input_names: list[str] = [
        "input_ids",
        "attention_mask",
        "token_type_ids",
        "pos_embed_ids",
        "lp_embeds",
        "labels",
        "token_ids",
        "mol_features",
        "atom_props",
        "bond_props",
        "target_values",
    ]

    def __init__(
        self,
        config: MolTConfig,
        errors="replace",
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        add_prefix_space=False,
        **kwargs,
    ):
        super().__init__(
            errors=errors,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            add_prefix_space=add_prefix_space,
            **kwargs,
        )
        self.config = config
        atoms = config.atoms
        bonds = config.bonds
        mol_features = config.mol_descriptors
        self.target_token = "<target>"

        # make target token a special token
        special_tokens = [
            self.bos_token,
            self.eos_token,
            self.pad_token,
            self.sep_token,
            self.cls_token,
            self.unk_token,
            self.mask_token,
        ]

        token_encoder = dict()
        for val in special_tokens:
            idx = len(token_encoder)
            if val in token_encoder:
                token_encoder[val] = min(idx, token_encoder[val])
            else:
                token_encoder[val] = idx

        token_encoder[self.target_token] = len(token_encoder)

        for k in bonds + atoms:
            token_encoder[k] = len(token_encoder)

        for k in mol_features:
            token_encoder[k] = len(token_encoder)

        self.encoder = token_encoder
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.laplace_embedding_dim = config.laplace_embeds_size
        self.is_fast = False

    def __len__(self) -> int:
        return len(self.encoder)

    @property
    def vocab_size(self):
        return len(self.encoder)

    def get_vocab(self) -> Dict[str, int]:
        return self.encoder

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.decoder.get(index)

    def tokenize(
        self,
        text: str,
        pair: str | None = None,
        add_special_tokens: bool = False,
        **kwargs,
    ) -> List[str]:
        raise NotImplementedError()

    def _decode(
        self,
        token_ids: int | List[int],
        skip_special_tokens: Optional[bool] = False,
        clean_up_tokenization_spaces: Optional[bool] = None,
        **kwargs,
    ) -> str:
        return super()._decode(
            token_ids,
            skip_special_tokens,  # type: ignore
            clean_up_tokenization_spaces,  # type: ignore
            **kwargs,  # type: ignore
        )

    def _encode_plus(
        self,
        text: Union[TextInput, PreTokenizedInput, EncodedInput],
        text_pair: Optional[Union[TextInput, PreTokenizedInput, EncodedInput]] = None,
        add_special_tokens: bool = True,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_attention_mask: Optional[bool] = None,
        return_special_tokens_mask: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs,
    ) -> BatchEncoding:
        mol = Chem.MolFromSmiles(text)  # type: ignore
        # mol = Chem.AddHs(mol)  # type: ignore

        adj_mat = Chem.rdmolops.GetAdjacencyMatrix(mol)  # type: ignore
        lp_embeds = get_lp_embeddings(adj_mat, self.laplace_embedding_dim)

        atoms, atom_props = get_atom_properties(mol)
        bonds, edge_list, bond_props = get_bond_properties(mol)

        (
            token_ids,
            atom_mask,
            bond_mask,
            pos_embed_ids,
        ) = generate_tokens_atom_bond_mask_pos_embed_ids(edge_list)

        assert set(chain.from_iterable(edge_list.tolist())) == set(
            token_ids[atom_mask].tolist()
        )

        # convert the tokens idxs to match token_mapping dictionary
        input_ids = np.zeros_like(token_ids, dtype=np.uint)
        input_ids[atom_mask] = np.array(
            [self.encoder[atoms[x]] for x in token_ids[atom_mask]]
        )
        input_ids[bond_mask] = np.array(
            [self.encoder[bonds[x]] for x in token_ids[bond_mask]]
        )

        # padding lp_embeds to match shape of input_ids
        # This is needed for _pad later on
        npad = input_ids.shape[0] - lp_embeds.shape[0]
        lp_embeds = np.pad(lp_embeds, ((0, npad), (0, 0)), constant_values=0.0)

        def adjust_properties(props, token_ids, mask):
            adjusted_props = dict()
            for k, v in props.items():
                adjusted_v = np.zeros(shape=token_ids.shape, dtype=v.dtype)
                adjusted_v[mask] = np.array([v[x] for x in token_ids[mask]])
                adjusted_props[k] = adjusted_v

            return adjusted_props

        atom_props = adjust_properties(atom_props, token_ids, atom_mask)
        bond_props = adjust_properties(bond_props, token_ids, bond_mask)

        atom_props = pack_atom_properties(**atom_props)
        bond_props = pack_bond_properties(**bond_props)

        token_type_ids = np.zeros_like(input_ids, dtype=np.uint)
        token_type_ids[atom_mask] = TokenType.ATOM.value
        token_type_ids[bond_mask] = TokenType.BOND.value

        # append mol_prop tokens
        if self.config.use_mol_descriptor_tokens:
            mol_features = np.array(
                [kwargs[feat] for feat in self.config.feature_names]
            )
            n_mol_feats = mol_features.shape[0]

            mol_feature_input_ids = [self.encoder[x] for x in self.config.feature_names]
            input_ids = np.concatenate([input_ids, mol_feature_input_ids], axis=-1)

            mol_feat_pad_len = input_ids.shape[0] - mol_features.shape[0]
            mol_features = np.pad(
                mol_features, (mol_feat_pad_len, 0), constant_values=0.0
            )

            token_type_ids = np.pad(
                token_type_ids, (0, n_mol_feats), constant_values=TokenType.FEAT.value
            )

            token_ids = np.pad(token_ids, (0, n_mol_feats), constant_values=0)
            pos_embed_ids = np.pad(
                pos_embed_ids, ((0, n_mol_feats), (0, 0)), constant_values=0
            )  # type: ignore
            lp_embeds = np.pad(lp_embeds, ((0, n_mol_feats), (0, 0)), constant_values=0)
            atom_props = np.pad(
                atom_props, ((0, 0), (0, n_mol_feats)), constant_values=0
            )  # type: ignore
            bond_props = np.pad(
                bond_props, ((0, 0), (0, n_mol_feats)), constant_values=0
            )  # type: ignore

        # Add target value tokens
        if self.config.use_target_token:
            input_ids = np.pad(
                input_ids, (0, 1), constant_values=self.encoder[self.target_token]
            )

            target_values = np.zeros_like(input_ids, dtype=float)
            target_values[-1] = kwargs[self.config.target_col_name]

            token_type_ids = np.pad(
                token_type_ids, (0, 1), constant_values=TokenType.TGT.value
            )

            token_ids = np.pad(token_ids, (0, 1), constant_values=0)
            pos_embed_ids = np.pad(pos_embed_ids, ((0, 1), (0, 0)), constant_values=0)  # type: ignore
            lp_embeds = np.pad(lp_embeds, ((0, 1), (0, 0)), constant_values=0)  # type: ignore
            atom_props = np.pad(atom_props, ((0, 0), (0, 1)), constant_values=0)  # type: ignore
            bond_props = np.pad(bond_props, ((0, 0), (0, 1)), constant_values=0)  # type: ignore
            mol_features = np.pad(mol_features, (0, 1), constant_values=0)  # type: ignore
        else:
            target_values = kwargs[self.config.target_col_name]

        # convert to lists for easier processing later
        # flattening pos_embeds for compatiability with BatchEncoding and DataCollator
        # maybe do list conversion after special_tokens?

        encoded_inputs = {
            "token_ids": token_ids.astype(int).tolist(),
            "input_ids": input_ids.astype(int).tolist(),
            "pos_embed_ids": pos_embed_ids.flatten().tolist(),
            "lp_embeds": lp_embeds.flatten().tolist(),
            "token_type_ids": token_type_ids.astype(int).tolist(),
            "atom_props": atom_props.tolist(),
            "bond_props": bond_props.tolist(),
            "target_values": target_values,
        }

        if self.config.use_mol_descriptor_tokens:
            encoded_inputs["mol_features"] = mol_features.tolist()

        if self.config.use_target_token:
            encoded_inputs["target_values"] = target_values.tolist()
        else:
            encoded_inputs["target_values"] = target_values

        if truncation_strategy != TruncationStrategy.DO_NOT_TRUNCATE:
            raise ValueError("Truncation for molecules does not make sense")

        if add_special_tokens:
            # add only start and end tokens for now
            encoded_inputs["input_ids"].insert(0, self.cls_token_id)
            encoded_inputs["input_ids"].append(self.eos_token_id)

            encoded_inputs["token_ids"].insert(0, 0)
            encoded_inputs["token_ids"].append(0)

            encoded_inputs["token_type_ids"].insert(0, TokenType.SPECIAL)
            encoded_inputs["token_type_ids"].append(TokenType.SPECIAL)

            for entry in encoded_inputs["atom_props"]:
                entry.insert(0, 0)
                entry.append(0)

            for entry in encoded_inputs["bond_props"]:
                entry.insert(0, 0)
                entry.append(0)

            if self.config.use_mol_descriptor_tokens:
                encoded_inputs["mol_features"].insert(0, 0.0)
                encoded_inputs["mol_features"].append(0.0)

            if self.config.use_target_token:
                encoded_inputs["target_values"].insert(0, 0.0)
                encoded_inputs["target_values"].append(0.0)
            

            encoded_inputs["pos_embed_ids"] = (
                [0.0] * 2 + encoded_inputs["pos_embed_ids"] + [0.0] * 2
            )

            # We are not padding from the left. If we did, there would be a mismatch
            # between indexes of lp_embeds and indexes inside pos_embed_ids; the token
            # with index 0 would have the lp_embeds assigned as 0.0
            encoded_inputs["lp_embeds"] = (
                encoded_inputs["lp_embeds"]
                + [0.0] * self.config.laplace_embeds_size
                + [0.0] * self.config.laplace_embeds_size
            )

        # Padding
        if padding_strategy != PaddingStrategy.DO_NOT_PAD or return_attention_mask:
            encoded_inputs = self.pad(
                encoded_inputs,
                max_length=max_length,
                padding=padding_strategy.value,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask,
            )

        if return_special_tokens_mask:
            encoded_inputs["special_tokens_mask"] = [
                1 if (t == TokenType.SPECIAL) else 0
                for t in encoded_inputs["token_type_ids"]  # type: ignore
            ]

        batch_outputs = BatchEncoding(
            encoded_inputs,
            tensor_type=return_tensors,
            prepend_batch_axis=True,  # type: ignore
        )

        return batch_outputs

    def _batch_encode_plus(
        self,
        batch_text_or_text_pairs: List[TextInput]
        | List[TextInputPair]
        | List[PreTokenizedInput]
        | List[PreTokenizedInputPair]
        | List[EncodedInput]
        | List[EncodedInputPair],
        add_special_tokens: bool = True,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
        max_length: int | None = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: int | None = None,
        return_tensors: str | TensorType | None = None,
        return_token_type_ids: bool | None = None,
        return_attention_mask: bool | None = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs,
    ) -> BatchEncoding:
        batched_encoded_entries = defaultdict(list)
        for entry in batch_text_or_text_pairs:
            encoded_entry = self._encode_plus(
                text=entry,  # type: ignore
                text_pair=None,
                add_special_tokens=add_special_tokens,
                padding_strategy=padding_strategy,
                truncation_strategy=truncation_strategy,
                max_length=max_length,
                stride=stride,
                is_split_into_words=is_split_into_words,
                pad_to_multiple_of=pad_to_multiple_of,
                return_tensors=return_tensors,
                return_token_type_ids=return_token_type_ids,
                return_attention_mask=return_attention_mask,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask,
                return_offsets_mapping=return_offsets_mapping,
                return_length=return_length,
                verbose=verbose,
                **kwargs,
            ).data

            for k, v in encoded_entry.items():
                batched_encoded_entries[k].append(v)

        return BatchEncoding(
            batched_encoded_entries,
            tensor_type=return_tensors,
            prepend_batch_axis=True,  # type: ignore
        )

    def convert_tokens_to_ids(
        self, tokens: Union[str, List[str]]
    ) -> Union[int, List[int]]:
        if isinstance(tokens, str):
            return self.encoder[tokens]

        return [self.encoder[t] for t in tokens]

    def _pad(
        self,
        encoded_inputs: Dict[str, EncodedInput] | BatchEncoding,
        max_length: int | None = None,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        pad_to_multiple_of: int | None = None,
        return_attention_mask: bool | None = None,
    ) -> Dict[str, EncodedInput] | BatchEncoding:
        # Load from model defaults
        if return_attention_mask is None:
            return_attention_mask = "attention_mask" in self.model_input_names

        if self.padding_side == "left":
            raise ValueError("Padding is supported only on the right side")

        input_length = len(encoded_inputs[self.model_input_names[0]])  # type: ignore

        if padding_strategy == PaddingStrategy.LONGEST:
            max_length = input_length

        if (
            max_length is not None
            and pad_to_multiple_of is not None
            and (max_length % pad_to_multiple_of != 0)
        ):
            max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of

        needs_to_be_padded = (
            padding_strategy != PaddingStrategy.DO_NOT_PAD
            and input_length != max_length
        )

        # Initialize attention mask if not present.
        if return_attention_mask and "attention_mask" not in encoded_inputs:
            encoded_inputs["attention_mask"] = [1] * input_length

        if needs_to_be_padded and max_length is not None:
            difference = max_length - input_length

            padding_funcs = {
                "input_ids": lambda x: x + [self.pad_token_id] * difference,
                "attention_mask": lambda x: x + [0] * difference,
                "token_ids": lambda x: x + [0] * difference,
                "token_type_ids": lambda x: x + [TokenType.SPECIAL] * difference,
                "pos_embed_ids": lambda x: x + [0.0] * (2 * difference),
                "lp_embeds": lambda x: x
                + [0.0] * (self.laplace_embedding_dim * difference),
                "atom_props": lambda x: [entry + [0] * difference for entry in x],
                "bond_props": lambda x: [entry + [0] * difference for entry in x],
                "mol_features": lambda x: x + [0] * difference,
                "target_values": lambda x: x + [0] * difference,
            }

            if self.config.use_target_token:
                padding_funcs["target_values"] = lambda x: x + [0] * difference
            else:
                padding_funcs["target_values"] = lambda x: x

            for k in encoded_inputs.keys():
                encoded_inputs[k] = padding_funcs[k](encoded_inputs[k])

        return encoded_inputs
