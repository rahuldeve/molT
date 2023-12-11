from datasets import load_dataset
from molT import (
    MolTConfig,
    MolTTokenizer,
    MolTForMaskedMM,
    DataCollatorForMaskedMolecularModeling,
)
from functools import partial
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import SchedulerType
from transformers import TrainerCallback

from accelerate import Accelerator


def tokenize(entry, tokenizer):
    return tokenizer(
        entry["smiles"],
        truncation=False,
        return_attention_mask=True,
        return_special_tokens_mask=True,
    )


def load_data():
    ds = load_dataset("sagawa/ZINC-canonicalized")["validation"].select(range(5000))
    return ds.train_test_split(seed=42)

def preprocess(ds, tokenizer):
    tok_func = partial(tokenize, tokenizer=tokenizer)
    ds = ds.map(tok_func, num_proc=2)
    return ds


def fn_metrics(eval_pred):
    mm_loss, atom_prop_loss, bond_prop_loss, mol_desc_loss = eval_pred.predictions
    return {
        "mm_loss": mm_loss.mean(),
        "atom_prop_loss": atom_prop_loss.mean(),
        "bond_prop_loss": bond_prop_loss.mean(),
        "mol_desc_loss": mol_desc_loss.mean(),
    }


def train_func(model, ds, data_collator):
    training_args = TrainingArguments(
        output_dir="molT_runs",
        evaluation_strategy="steps",
        learning_rate=2e-4,
        num_train_epochs=20,
        weight_decay=0.01,
        push_to_hub=False,
        logging_steps=2,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        eval_steps=4,
        gradient_accumulation_steps=64,
        warmup_ratio=0.1,
        # report_to="tensorboard",
        # dataloader_num_workers=4,
        # lr_scheduler_type=SchedulerType.COSINE_WITH_RESTARTS,
        data_seed=42,
        # label_names = ['reg'],
        # load_best_model_at_end = True,
        # metric_for_best_model = "eval_loss",
        max_grad_norm=0.5
        # greater_is_better = False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        data_collator=data_collator,
        compute_metrics=fn_metrics,
    )

    trainer.train()


if __name__ == "__main__":
    # accelerator = Accelerator()

    model_config = MolTConfig()
    tokenizer = MolTTokenizer(model_config)
    model = MolTForMaskedMM(model_config)

    # try to use datasets caching below
    ds = load_data()
    ds = preprocess(ds, tokenizer)
    # accelerator.wait_for_everyone()

    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForMaskedMolecularModeling(
        tokenizer=tokenizer, mlm_probability=0.15
    )

    train_func(model, ds, data_collator)
