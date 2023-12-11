from enum import IntEnum, auto

class DescriptorType(IntEnum):
    CONTINIOUS = auto()
    ORDINAL = auto()
    CATEGORICAL = auto()


mol_descriptor_type_mapping = {
    'qed': DescriptorType.CONTINIOUS
}