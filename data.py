import numpy as np
from sklearn.preprocessing import StandardScaler
import rdkit.Chem as Chem


def standardize(smiles):
    mol = Chem.MolFromSmiles(smiles)  # type: ignore
    return Chem.MolToSmiles(mol)  # type: ignore


def generate_rdkit_descriptors(entry, descriptor_names):
    from rdkit.Chem.Descriptors import _descList
    import rdkit.Chem as Chem

    smiles = entry["smiles"]
    # std_smiles = standardize(smiles)
    mol = Chem.MolFromSmiles(smiles)  # type: ignore
    mol = Chem.AddHs(mol)  # type: ignore
    relevant_descriptors = filter(lambda x: x[0] in descriptor_names, _descList)
    descriptor_func_mapping = {x[0]: x[1] for x in relevant_descriptors}
    descriptors = {
        desc: desc_func(mol) for desc, desc_func in descriptor_func_mapping.items()
    }
    return descriptors


def scale_batch(batch, scaler):
    batch = batch.pa_table.to_pandas()
    batch.loc[:, scaler.feature_names_in_] = scaler.transform(
        batch.loc[:, scaler.feature_names_in_]
    )
    return batch


def generate_and_scale_mol_descriptors(
    ds, descriptor_names, num_samples=10_000, num_proc=None
):
    ds = ds.map(
        generate_rdkit_descriptors,
        fn_kwargs={"descriptor_names": descriptor_names},
        num_proc=num_proc,
    )

    np.random.seed(42)
    rand_ids = np.random.randint(0, len(ds["train"]), num_samples)
    random_samples = ds["train"].select(rand_ids).select_columns(descriptor_names)
    random_samples = random_samples.to_pandas()

    scaler = StandardScaler().set_output(transform="pandas")
    scaler = scaler.fit(random_samples)  # type: ignore

    ds = ds.map(
        scale_batch,
        fn_kwargs={"scaler": scaler},
        batched=True,
        num_proc=num_proc
    )

    return ds, scaler
