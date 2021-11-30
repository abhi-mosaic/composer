import logging
import multiprocessing as mp
import tempfile
from dataclasses import dataclass
from itertools import chain
from os.path import join
from typing import List

import datasets
import yahp as hp
from transformers.testing_utils import CaptureLogger

from composer.core.types import Batch
from composer.datasets.hparams import DataloaderSpec, DatasetHparams

log = logging.getLogger(__name__)


def _split_dict_fn(batch: Batch, n_microbatches: int) -> List[Batch]:
    if isinstance(batch, dict):
        chunked = {k: v.chunk(n_microbatches) for k, v in batch.items()}
        num_chunks = len(list(chunked.values())[0])
        return [{k: v[idx] for k, v in chunked.items()} for idx in range(num_chunks)]
    else:
        raise ValueError(f'Expect batch from dataloader to be of type Dict[str, Tensor], but got {type(batch)}')


@dataclass
class StreamingLMDatasetHparams(DatasetHparams):
    """
    Defines a generic dataset class for autoregressive and masked language models.
    """

    dataset_name: str = hp.required("Name of the dataset to load.")
    split: str = hp.required("Whether to use 'train', 'validation' or 'test' split.")
    tokenizer_name: str = hp.required("The name of the tokenizer to preprocess text with.")
    use_masked_lm: bool = hp.required("Whether the dataset shoud be encoded with masked language modeling or not.")
    num_tokens: int = hp.optional(doc='If desired, the number of tokens to truncate the dataset to.', default=0)
    mlm_probability: float = hp.optional("If using masked language modeling, the probability to mask tokens with.",
                                         default=0.15)
    dataset_specific_name: str = hp.optional(
        "If required, the specific subset of the dataset that you would like to use.", default=None)
    seed: int = hp.optional("Which seed to use to generate train and validation splits.", default=5)
    subsample_ratio: float = hp.optional(default=1.0, doc='If desired, the percentage of the dataset to use.')
    train_sequence_length: int = hp.optional(
        default=1024, doc='Optionally, the ability to set a custom sequence length for the training dataset.')
    val_sequence_length: int = hp.optional(
        default=1024, doc='Optionally, the ability to set a custom sequence length for the validation dataset.')
    shuffle: bool = hp.optional("Whether to shuffle the dataset for each epoch.", default=True)
    drop_last: bool = hp.optional("Whether to drop the last samples for the last batch.", default=False)

    def validate(self):
        # TODO (Moin): this re-loads a large dataset into memory three times -- can the REng team permit
        # returning a dataloader for a particular split?
        if self.split not in ['train', 'validation', 'test']:
            raise ValueError("The dataset split must be one of 'train', 'validation', or 'test'.")

        if self.use_masked_lm:
            if self.mlm_probability <= 0.0:
                raise ValueError(
                    "If using Masked Language Modeling, you must replace tokens with a non-zero probability.")

        if self.num_tokens > 0 and self.subsample_ratio < 1.0:
            raise Exception("Must specify one of num_tokens OR subsample_ratio, cannot specify both.")

    def initialize_object(self) -> DataloaderSpec:
        try:
            import datasets
            import transformers
        except ImportError:
            raise ImportError('huggingface transformers and datasets are not installed. '
                              'Please install with `pip install mosaicml-composer[nlp]`')
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.tokenizer_name)  #type: ignore (thirdparty)
        self.config = transformers.AutoConfig.from_pretrained(self.tokenizer_name)  #type: ignore (thirdparty)
        lm_datasets = datasets.load_dataset(self.dataset_name,
                                            self.dataset_specific_name,
                                            split=self.split,
                                            streaming=True)

        # TODO (Moin): this re-loads a large dataset into memory three times -- can the REng team permit
        # returning a dataloader for a particular split?
        if self.split not in ['train', 'validation', 'test']:
            raise ValueError("The dataset split must be one of 'train', 'validation', or 'test'.")

        log.info(f"LM datasets: {lm_datasets}")
        log.info(f"Total number of samples: {lm_datasets.info.splits[self.split].num_examples:e}")
        # we're going to hack the len() property so this can be an instance of collections.abc.Sized
        self.dataset = lm_datasets
        cpu_count = mp.cpu_count()

        # column_names = self.dataset.column_names
        text_column_name = "text"

        def tokenize_function(examples):
            return self.tokenizer(examples[text_column_name])

        batch_size = 1000
        self.dataset = self.dataset.map(
            tokenize_function,
            batched=True,
            batch_size=batch_size,
        )

        block_size = 1024
        # if block_size > self.tokenizer.model_max_length:
        # log.warning(f"The block_size passed ({block_size}) is larger than the maximum length for the model"
        # f"({self.tokenizer.model_max_length}). Using block_size={self.tokenizer.model_max_length}.")
        block_size = min(block_size, self.tokenizer.model_max_length)

        # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
        batch_size = 100
        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {}
            for k in examples.keys():
                concatenated_examples[k] = list(chain(*examples[k]))
            # concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
            # customize this part to your needs.
            if total_length >= block_size:
                total_length = (total_length // block_size) * block_size
            # Split by chunks of max_len.
            result = {
                k: [t[i:i + block_size] for i in range(0, total_length, block_size)
                   ] for k, t in concatenated_examples.items()
            }
            result["labels"] = result["input_ids"].copy()
            return result

        # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
        # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
        # to preprocess.
        #
        # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
        # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

        self.dataset = self.dataset.map(group_texts, batched=True, batch_size=batch_size)

        self.dataset = SizedIterableDataset(self.dataset)
        # self.data_collator = transformers.DataCollatorForLanguageModeling(tokenizer=self.tokenizer,
        # mlm=self.use_masked_lm,
        # mlm_probability=self.mlm_probability,
        # pad_to_multiple_of=8)

        self.data_collator = transformers.data.data_collator.default_data_collator

        return DataloaderSpec(
            dataset=self.dataset,  #type: ignore (thirdparty)
            collate_fn=self.data_collator,
            shuffle=self.shuffle,
            drop_last=self.drop_last,
            split_fn=_split_dict_fn)


class SizedIterableDataset(datasets.IterableDataset):

    def __init__(self, dataset):

        self.dataset = iter(dataset)

    def __len__(self):
        return int(1e3 * 512)

    def __iter__(self):
        return iter(self.dataset)

    def __getitem__(self, idx):
        item = next(self.dataset)
        return item
