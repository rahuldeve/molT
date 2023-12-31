import torch
import torch.nn as nn
from rdkit.Chem.rdchem import ChiralType, HybridizationType, StereoType

from .config import MolTConfig
from .utils import TokenType, unpack_atom_properties, unpack_bond_properties


class AtomPropertyEmbedder(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.per_prop_embedding_size = config.embedding_size // 4
        assert self.per_prop_embedding_size * 4 == config.embedding_size

        self.in_ring_embedding = nn.Embedding(
            3, self.per_prop_embedding_size, padding_idx=0
        )
        self.charge_embedding = nn.Embedding(
            4, self.per_prop_embedding_size, padding_idx=0
        )
        self.hybridization_embedding = nn.Embedding(
            len(HybridizationType.values) + 1,
            self.per_prop_embedding_size,
            padding_idx=0,
        )
        self.chirality_embedding = nn.Embedding(
            len(ChiralType.values) + 1, self.per_prop_embedding_size, padding_idx=0
        )

    def forward(
        self,
        prop_atom_in_ring,
        prop_atom_charge,
        prop_atom_hybridization,
        prop_atom_chirality,
    ):
        in_ring_embeds = self.in_ring_embedding(prop_atom_in_ring.long())
        charge_embedding = self.charge_embedding(prop_atom_charge.long())
        hybridization_embedding = self.hybridization_embedding(
            prop_atom_hybridization.long()
        )
        chirality_embedding = self.chirality_embedding(prop_atom_chirality.long())

        prop_embedding = torch.cat(
            [
                in_ring_embeds,
                charge_embedding,
                hybridization_embedding,
                chirality_embedding,
            ],
            dim=-1,
        )

        return prop_embedding


class BondPropertyEmbedder(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.per_prop_embedding_size = config.embedding_size // 3
        assert self.per_prop_embedding_size * 3 == config.embedding_size

        self.aromatic_embedding = nn.Embedding(
            3, self.per_prop_embedding_size, padding_idx=0
        )

        self.conjugated_embedding = nn.Embedding(
            3, self.per_prop_embedding_size, padding_idx=0
        )

        self.stereo_embedding = nn.Embedding(
            len(StereoType.values) + 1, self.per_prop_embedding_size, padding_idx=0
        )

    def forward(
        self,
        prop_bond_aromatic,
        prop_bond_conjugated,
        prop_bond_stereo,
    ):
        aromatic_embeds = self.aromatic_embedding(prop_bond_aromatic.long())
        conjugated_embeds = self.conjugated_embedding(prop_bond_conjugated.long())
        stereo_embeds = self.stereo_embedding(prop_bond_stereo.long())

        prop_embedding = torch.cat(
            [
                aromatic_embeds,
                conjugated_embeds,
                stereo_embeds,
            ],
            dim=-1,
        )
        return prop_embedding


class MolDescriptorEmbedder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_embeddings, mol_desc_mask, mol_descriptors):
        mol_desc_embeddings = torch.where(
            mol_desc_mask.unsqueeze(-1),
            input_embeddings * mol_descriptors.unsqueeze(-1),
            0.0,
        )
        return mol_desc_embeddings


class RegressionTargetEmbedder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_embeddings, target_mask, target_values):
        target_encoded_embeddings = torch.where(
            target_mask.unsqueeze(-1),
            input_embeddings * target_values.unsqueeze(-1),
            0.0,
        )
        return target_encoded_embeddings


class MolTEmbeddings(nn.Module):
    def __init__(self, config: MolTConfig):
        super().__init__()
        self.embeddings = nn.Embedding(
            config.vocab_size, config.embedding_size, padding_idx=config.pad_token_id
        )
        self.atom_prop_embeddings = AtomPropertyEmbedder(config)
        self.bond_prop_embeddings = BondPropertyEmbedder(config)
        self.mol_descriptor_embeddings = MolDescriptorEmbedder()
        self.target_embedding = RegressionTargetEmbedder()

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.padding_idx = config.pad_token_id

        self.type_embeddings = nn.Embedding(len(TokenType), config.embedding_size)

    def forward(
        self,
        input_ids,
        token_type_ids,
        pos_embeds,
        pos_embeds_shape,
        atom_props,
        bond_props,
        mol_desc,
        target_values,
        **kwargs,
    ):
        pos_embeds_shape = (pos_embeds.shape[0], *(pos_embeds_shape[0].tolist()))
        pos_embeds = pos_embeds.reshape(pos_embeds_shape)

        token_type_embeds = self.type_embeddings(token_type_ids)

        input_embeddings = self.embeddings(input_ids)

        mol_desc_mask = token_type_ids == TokenType.DESC
        input_embeddings += self.mol_descriptor_embeddings(
            input_embeddings, mol_desc_mask, mol_desc
        )

        target_mask = token_type_ids == TokenType.TGT
        input_embeddings += self.target_embedding(
            input_embeddings, target_mask, target_values
        )

        atom_props = unpack_atom_properties(atom_props)
        bond_props = unpack_bond_properties(bond_props)
        atom_prop_embeddings = self.atom_prop_embeddings(**atom_props)
        bond_prop_embeddings = self.bond_prop_embeddings(**bond_props)

        prop_embeddings = atom_prop_embeddings + bond_prop_embeddings

        embeddings = torch.cat(
            [input_embeddings, pos_embeds, token_type_embeds, prop_embeddings], dim=-1
        )
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
