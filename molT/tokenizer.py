from collections import defaultdict
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
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.Descriptors import _descList
from .utils import TokenType, pack_atom_properties, pack_bond_properties
from .config import MolTConfig

# def get_lp_embeddings(adj_mat, k):
#     A = adj_mat
#     D = np.diag(A.sum(axis=-1))
#     N = np.linalg.pinv(D) ** 0.5
#     I = np.eye(A.shape[0])
#     L = I - N @ A @ N

#     eig_val, eig_vec = np.linalg.eig(L)
#     if L.shape[0] <= k:
#         # handle cases where num nodes <= k
#         difference = k + 1 - L.shape[0]
#         eig_vec = np.pad(eig_vec, ((0, 0), (0, difference)), constant_values=(0.0) )
#         return eig_vec

#     kpartition_indices = np.argpartition(eig_val, k + 1)[: k + 1]
#     topk_eig_vals = eig_val[kpartition_indices]
#     topk_indices = kpartition_indices[topk_eig_vals.argsort()][1:]
#     topk_eig_vec = np.real(eig_vec[:, topk_indices])
#     return topk_eig_vec


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


def get_mol_desc(mol, descriptor_names):
    relevant_descriptors = filter(lambda x: x[0] in descriptor_names, _descList)
    descriptor_funcs = (x[1] for x in relevant_descriptors)
    return np.array([f(mol) for f in descriptor_funcs])


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


def generate_pos_embeddings(tokens, lp_embeds, edge_list, atom_mask, bond_mask):
    pos_embeds = np.zeros((tokens.shape[0], 2 * lp_embeds.shape[-1]))
    pos_embeds[bond_mask, :] = lp_embeds[edge_list].reshape((edge_list.shape[0], -1))
    pos_embeds[atom_mask, :] = lp_embeds[
        np.repeat(tokens[atom_mask][:, None], 2, axis=-1)
    ].reshape((edge_list.shape[0], -1))
    return pos_embeds


def generate_tokens_atom_mask_bond_mask(edge_list):
    tokens = np.zeros(2 * edge_list.shape[0], dtype=np.uint)
    atom_mask = np.zeros_like(tokens, dtype=bool)
    atom_mask[0::2] = True
    bond_mask = ~atom_mask

    tokens[atom_mask] = edge_list[:, 0]
    tokens[bond_mask] = np.arange(edge_list.shape[0])
    return tokens, atom_mask, bond_mask


class MolTTokenizer(PreTrainedTokenizerBase):
    model_input_names: list[str] = [
        "input_ids",
        "attention_mask",
        "token_type_ids",
        "pos_embeds",
        "pos_embeds_shape",
        "labels",
        "token_idxs",
        "mol_desc",
        "morgan_fps",
        "atom_props",
        "bond_props",
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
        mol_descriptors = config.mol_descriptors

        special_tokens = [
            self.bos_token,
            self.eos_token,
            self.pad_token,
            self.sep_token,
            self.cls_token,
            self.unk_token,
            self.mask_token,
        ]

        # assert len(special_tokens) == config.num_special_tokens

        # special_tokens_encoder_map = dict()
        # for val in special_tokens:
        #     idx = len(special_tokens_encoder_map)
        #     if val in special_tokens_encoder_map:
        #         special_tokens_encoder_map[val] = min(
        #             idx, special_tokens_encoder_map[val]
        #         )
        #     else:
        #         special_tokens_encoder_map[val] = idx

        # atom_bond_token_encoder_map = {
        #     k: idx + len(special_tokens_encoder_map)
        #     for idx, k in enumerate(bonds + atoms)
        # }

        # self.encoder = special_tokens_encoder_map | atom_bond_token_encoder_map

        token_encoder = dict()
        for val in special_tokens:
            idx = len(token_encoder)
            if val in token_encoder:
                token_encoder[val] = min(idx, token_encoder[val])
            else:
                token_encoder[val] = idx

        for k in bonds + atoms:
            token_encoder[k] = len(token_encoder)

        for k in mol_descriptors:
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
        # mol = Chem.MolFromSmiles(text)  # type: ignore
        # # mol = Chem.AddHs(mol)  # type: ignore

        # atoms, _ = get_atom_properties(mol)
        # bonds, edge_list, _ = get_bond_properties(mol)
        # tokens, atom_mask, bond_mask = generate_tokens_atom_mask_bond_mask(edge_list)

        # # convert the tokens idxs to match token_mapping dictionary
        # input_ids = np.zeros_like(tokens, dtype=np.uint)
        # input_ids[atom_mask] = np.array(
        #     [self.encoder[atoms[x]] for x in tokens[atom_mask]]
        # )
        # input_ids[bond_mask] = np.array(
        #     [self.encoder[bonds[x]] for x in tokens[bond_mask]]
        # )

        # input_ids = input_ids.tolist()
        # if add_special_tokens:
        #     input_ids = [self.cls_token_id] + input_ids

        # return [self.decoder[x] for x in input_ids]

    def _decode(
        self,
        token_ids: int | List[int],
        skip_special_tokens: Optional[bool] = False,
        clean_up_tokenization_spaces: Optional[bool] = None,
        **kwargs,
    ) -> str:
        return super()._decode(
            token_ids, skip_special_tokens, clean_up_tokenization_spaces, **kwargs  # type: ignore
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
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
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

        tokens, atom_mask, bond_mask = generate_tokens_atom_mask_bond_mask(edge_list)
        pos_embeds = generate_pos_embeddings(
            tokens, lp_embeds, edge_list, atom_mask, bond_mask
        )

        # convert the tokens idxs to match token_mapping dictionary
        input_ids = np.zeros_like(tokens, dtype=np.uint)
        input_ids[atom_mask] = np.array(
            [self.encoder[atoms[x]] for x in tokens[atom_mask]]
        )
        input_ids[bond_mask] = np.array(
            [self.encoder[bonds[x]] for x in tokens[bond_mask]]
        )

        def adjust_properties(props, tokens, mask):
            adjusted_props = dict()
            for k, v in props.items():
                adjusted_v = np.zeros(shape=tokens.shape, dtype=v.dtype)
                adjusted_v[mask] = np.array([v[x] for x in tokens[mask]])
                adjusted_props[k] = adjusted_v

            return adjusted_props

        atom_props = adjust_properties(atom_props, tokens, atom_mask)
        bond_props = adjust_properties(bond_props, tokens, bond_mask)

        atom_props = pack_atom_properties(**atom_props)
        bond_props = pack_bond_properties(**bond_props)

        token_type_ids = np.zeros_like(input_ids, dtype=np.uint)
        token_type_ids[atom_mask] = TokenType.ATOM
        token_type_ids[bond_mask] = TokenType.BOND

        # append mol_prop tokens
        mol_desc = get_mol_desc(mol, self.config.mol_descriptors)
        n_mol_desc = len(self.config.mol_descriptors)

        mol_descriptor_ids = [self.encoder[x] for x in self.config.mol_descriptors]
        input_ids = np.concatenate([input_ids, mol_descriptor_ids], axis=-1)

        mol_desc_pad_len = input_ids.shape[0] - mol_desc.shape[0]
        mol_desc = np.pad(mol_desc, (mol_desc_pad_len, 0), constant_values=0.0)

        token_type_ids = np.pad(
            token_type_ids, (0, n_mol_desc), constant_values=TokenType.DESC.value
        )

        tokens = np.pad(tokens, (0, n_mol_desc), constant_values=0)
        pos_embeds = np.pad(pos_embeds, ((0, n_mol_desc), (0, 0)), constant_values=0)  # type: ignore
        atom_props = np.pad(atom_props, ((0, 0), (0, n_mol_desc)), constant_values=0)  # type: ignore
        bond_props = np.pad(bond_props, ((0, 0), (0, n_mol_desc)), constant_values=0)  # type: ignore

        # convert to lists for easier processing later
        # flattening pos_embeds for compatiability with BatchEncoding and DataCollator
        # maybe do list conversion after special_tokens?
        token_idxs = tokens.astype(int).tolist()
        input_ids = input_ids.astype(int).tolist()
        pos_embeds_shape = list(pos_embeds.shape)
        pos_embeds = pos_embeds.flatten().tolist()
        token_type_ids = token_type_ids.astype(int).tolist()
        atom_props = atom_props.tolist()
        bond_props = bond_props.tolist()
        mol_desc = mol_desc.tolist()

        encoded_inputs = {
            "token_idxs": token_idxs,
            "input_ids": input_ids,
            "pos_embeds": pos_embeds,
            "pos_embeds_shape": pos_embeds_shape,
            "token_type_ids": token_type_ids,
            "atom_props": atom_props,
            "bond_props": bond_props,
            "mol_desc": mol_desc,
        }

        if truncation_strategy != TruncationStrategy.DO_NOT_TRUNCATE:
            raise ValueError("Truncation for molecules does not make sense")

        if add_special_tokens:
            # add only start and end tokens for now
            encoded_inputs["input_ids"].insert(0, self.cls_token_id)
            encoded_inputs["input_ids"].append(self.eos_token_id)

            encoded_inputs["token_idxs"].insert(0, 0)
            encoded_inputs["token_idxs"].append(0)

            encoded_inputs["token_type_ids"].insert(0, TokenType.SPECIAL)
            encoded_inputs["token_type_ids"].append(TokenType.SPECIAL)

            for entry in encoded_inputs["atom_props"]:
                entry.insert(0, 0)
                entry.append(0)

            for entry in encoded_inputs["bond_props"]:
                entry.insert(0, 0)
                entry.append(0)

            encoded_inputs["mol_desc"].insert(0, 0.0)
            encoded_inputs["mol_desc"].append(0.0)

            # encoded_inputs['pos_embeds'].insert(0, [0.0] * (2 * self.laplace_embedding_dim))
            # encoded_inputs['pos_embeds'].append([0.0] * (2 * self.laplace_embedding_dim))

            # encoded_inputs["pos_embeds_shape"] = [
            #     encoded_inputs["pos_embeds_shape"][0] + 2,
            #     encoded_inputs["pos_embeds_shape"][1],
            # ]

            encoded_inputs["pos_embeds"] = (
                [0.0] * (2 * self.laplace_embedding_dim)
                + encoded_inputs["pos_embeds"]
                + [0.0] * (2 * self.laplace_embedding_dim)
            )

            encoded_inputs["pos_embeds_shape"] = [
                encoded_inputs["pos_embeds_shape"][0] + 2,
                encoded_inputs["pos_embeds_shape"][1],
            ]

        # # TODO: rewrite the below stuff using append and insert for clarity ?
        # if add_special_tokens:
        #     # add only start and end tokens for now
        #     encoded_inputs["input_ids"] = (
        #         [self.cls_token_id] + encoded_inputs["input_ids"] + [self.eos_token_id]
        #     )
        #     encoded_inputs["token_idxs"] = [0] + encoded_inputs["token_idxs"] + [0]
        #     # using origin as position embedding for [CLS] token
        #     encoded_inputs["pos_embeds"] = (
        #         [0.0] * (2 * self.laplace_embedding_dim)
        #         + encoded_inputs["pos_embeds"]
        #         + [0.0] * (2 * self.laplace_embedding_dim)
        #     )

        #     encoded_inputs["pos_embeds_shape"] = [
        #         encoded_inputs["pos_embeds_shape"][0] + 2,
        #         encoded_inputs["pos_embeds_shape"][1],
        #     ]

        #     encoded_inputs["token_type_ids"] = (
        #         [TokenType.SPECIAL]
        #         + encoded_inputs["token_type_ids"]
        #         + [TokenType.SPECIAL]
        #     )

        #     for entry in encoded_inputs["atom_props"]:
        #         entry.insert(0, 0)
        #         entry.append(0)

        #     for entry in encoded_inputs["bond_props"]:
        #         entry.insert(0, 0)
        #         entry.append(0)

        #     mol_desc.insert(0, 0)
        #     mol_desc.append(0)

        #     # for k, v in encoded_inputs.items():
        #     #     if k.startswith("prop"):
        #     #         encoded_inputs[k] = [0] + v + [0]

        # Padding
        if padding_strategy != PaddingStrategy.DO_NOT_PAD or return_attention_mask:
            encoded_inputs = self.pad(
                encoded_inputs,
                max_length=max_length,
                padding=padding_strategy.value,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask,
            )

        # Fix this; should probably not be including padding
        # if return_length:
        #     encoded_inputs["length"] = len(encoded_inputs["input_ids"])

        if return_special_tokens_mask:
            encoded_inputs["special_tokens_mask"] = [
                1 if (t == TokenType.SPECIAL) else 0
                for t in encoded_inputs["token_type_ids"]  # type: ignore
            ]

        batch_outputs = BatchEncoding(
            encoded_inputs, tensor_type=return_tensors, prepend_batch_axis=True  # type: ignore
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
            batched_encoded_entries, tensor_type=return_tensors, prepend_batch_axis=True  # type: ignore
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
            if self.padding_side == "left":
                encoded_inputs["input_ids"] = [
                    self.pad_token_id
                ] * difference + encoded_inputs[
                    "input_ids"
                ]  # type: ignore

                if return_attention_mask:
                    encoded_inputs["attention_mask"] = [
                        0
                    ] * difference + encoded_inputs[
                        "attention_mask"
                    ]  # type: ignore

                encoded_inputs["token_idxs"] = [0] * difference + encoded_inputs[
                    "token_idxs"
                ]  # type: ignore

                encoded_inputs["token_type_ids"] = [
                    TokenType.SPECIAL
                ] * difference + encoded_inputs[
                    "token_type_ids"
                ]  # type: ignore

                encoded_inputs["pos_embeds"] = [0.0] * (
                    2 * self.laplace_embedding_dim * difference
                ) + encoded_inputs[
                    "pos_embeds"
                ]  # type: ignore

                encoded_inputs["pos_embeds_shape"] = [
                    encoded_inputs["pos_embeds_shape"][0] + difference,  # type: ignore
                    encoded_inputs["pos_embeds_shape"][1],  # type: ignore
                ]

                # n_atom_props = len(encoded_inputs["atom_props"])
                encoded_inputs["atom_props"] = [
                    [0] * difference + entry for entry in encoded_inputs["atom_props"]  # type: ignore
                ]

                encoded_inputs["bond_props"] = [
                    [0] * difference + entry for entry in encoded_inputs["bond_props"]  # type: ignore
                ]

                encoded_inputs["mol_desc"] = [0] * difference + encoded_inputs[
                    "mol_desc"
                ]  # type: ignore

            elif self.padding_side == "right":
                encoded_inputs["input_ids"] = (
                    encoded_inputs["input_ids"] + [self.pad_token_id] * difference  # type: ignore
                )

                if return_attention_mask:
                    encoded_inputs["attention_mask"] = (
                        encoded_inputs["attention_mask"] + [0] * difference  # type: ignore
                    )

                encoded_inputs["token_idxs"] = (
                    encoded_inputs["token_idxs"] + [0] * difference  # type: ignore
                )

                encoded_inputs["token_type_ids"] = (
                    encoded_inputs["token_type_ids"] + [TokenType.SPECIAL] * difference  # type: ignore
                )

                encoded_inputs["pos_embeds"] = encoded_inputs["pos_embeds"] + [0.0] * (
                    2 * self.laplace_embedding_dim * difference
                )  # type: ignore

                encoded_inputs["pos_embeds_shape"] = [
                    encoded_inputs["pos_embeds_shape"][0] + difference,  # type: ignore
                    encoded_inputs["pos_embeds_shape"][1],  # type: ignore
                ]

                encoded_inputs["atom_props"] = [
                    entry + [0] * difference for entry in encoded_inputs["atom_props"]  # type: ignore
                ]

                encoded_inputs["bond_props"] = [
                    entry + [0] * difference for entry in encoded_inputs["bond_props"]  # type: ignore
                ]

                encoded_inputs["mol_desc"] = (
                    encoded_inputs["mol_desc"] + [0] * difference  # type: ignore
                )

            else:
                raise ValueError("Invalid padding strategy:" + str(self.padding_side))

        return encoded_inputs
