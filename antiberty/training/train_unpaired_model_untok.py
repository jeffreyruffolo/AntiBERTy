import os
import argparse
from datetime import datetime
import datasets
import transformers
import wandb

import deeph3
from deeph3.models.AntiBERTy.antiberty_config import *
from deeph3.models.AntiBERTy.AntiBERTy import AntiBERTy
from deeph3.models.AntiBERTy.DataCollatorForSpeciesChain import DataCollatorForSpeciesChain
from deeph3.util.util import exists, count_params

os.environ["TOKENIZERS_PARALLELISM"] = "true"


def cli():
    project_path = os.path.abspath(os.path.join(deeph3.__file__, "../.."))

    desc = 'Creates Huggingface dataset from OAS unpaired csv files'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('dataset',
                        type=str,
                        help='The CSV dataset for training')
    now = str(datetime.now().strftime('%y-%m-%d %H:%M:%S'))
    default_model_path = os.path.join(
        project_path, 'trained_models/BERTunpaired_{}/'.format(now))
    parser.add_argument('--output_dir', type=str, default=default_model_path)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--wandb_group', type=str, default="oasX")

    parser.add_argument('--model_size', type=str, default="SM")
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

    return parser.parse_args()


if __name__ == '__main__':
    args = cli()

    dataset = args.dataset
    output_dir = args.output_dir
    checkpoint = args.checkpoint
    group_name = args.wandb_group
    model_size = args.model_size
    batch_size = args.batch_size
    epochs = args.epochs
    local_rank = args.local_rank
    gradient_accumulation_steps = args.gradient_accumulation_steps

    tokenizer = transformers.BertTokenizerFast(
        vocab_file="deeph3/models/AntiBERTy/vocab.txt", do_lower_case=False)
    data_collator = DataCollatorForSpeciesChain(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15,
    )

    if model_size == "SM":
        config = transformers.BertConfig(vocab_size=len(tokenizer.vocab),
                                         **antiberty_sm_config)
    elif model_size == "MD":
        config = transformers.BertConfig(vocab_size=len(tokenizer.vocab),
                                         **antiberty_md_config)
    elif model_size == "LG":
        config = transformers.BertConfig(vocab_size=len(tokenizer.vocab),
                                         **antiberty_lg_config)
    elif model_size == "XL":
        config = transformers.BertConfig(vocab_size=len(tokenizer.vocab),
                                         **antiberty_xl_config)
    else:
        exit("Invalid model size: {}".format(model_size))

    if exists(checkpoint):
        model = AntiBERTy.from_pretrained(checkpoint)
    else:
        model = AntiBERTy(config=config)

    dataset = datasets.load_from_disk(dataset, keep_in_memory=False)

    # Split train, eval
    dataset_dict = dataset.train_test_split(test_size=int(1e6), seed=0)
    train_dataset, eval_dataset = dataset_dict["train"], dataset_dict["test"]

    run_name = os.path.split(output_dir)[1]
    wandb.login()

    if local_rank == 0:
        wandb.init(project="AntiBERTy",
                   name=run_name,
                   group=group_name,
                   resume=exists(checkpoint))
        wandb.config.update({"parameters": count_params(model)})

    training_args = transformers.TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        report_to="wandb",
        run_name=run_name,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        remove_unused_columns=False,
        dataloader_num_workers=4,
        save_steps=10_000,
        save_total_limit=2,
        prediction_loss_only=True,
        evaluation_strategy="steps",
        eval_steps=10_000,
        fp16=True,
        gradient_accumulation_steps=gradient_accumulation_steps,
        local_rank=local_rank,
    )
    trainer = transformers.Trainer(model=model,
                                   args=training_args,
                                   data_collator=data_collator,
                                   train_dataset=train_dataset,
                                   eval_dataset=eval_dataset)

    if checkpoint != None:
        trainer.train(checkpoint)
    else:
        trainer.train()

    trainer.save_model(output_dir)
    wandb.finish()

# deepspeed --num_gpus=4 deeph3/models/AntiBERTy/train_unpaired_model.py /home/jruffol1/scratch16-jgray21/jruffol1/OAS_unpaired/HF_OAS_unpaired/ --output_dir trained_models/BERTunpaired_e5 --batch_size 1024 --epochs 5
