import random
import typing as tp

import torch
from datasets import Dataset
from dataclasses import dataclass
from transformers import DataCollatorWithPadding

from datasets.arrow_dataset import (
    DatasetTransformationNotAllowedError,
)

import pandas as pd
import numpy as np


class LLMtgDataset(Dataset):
    """GReaT Dataset

    The LLMtgDataset overwrites the _getitem function of the HuggingFace Dataset Class to include the permutation step.

    Attributes:
        tokenizer (AutoTokenizer): Tokenizer from HuggingFace
    """

    def set_shuffler(self, shuffle=True):
        if shuffle:
            self.shuffle = True
        else:
            self.shuffle = False

    def set_serializer(self, serializer="great"):
        self.serializer = serializer
        if serializer == "list":
            self.others_token = {"prefix": None, "key_val_sep": ":", "text_sep": "\n"}
        elif serializer == "text":
            self.others_token = {"prefix": "The", "key_val_sep": "is", "text_sep": "."}
        elif serializer == "apval":
            self.others_token = {"prefix": None, "key_val_sep": ":", "text_sep": ","}
        else:
            self.others_token = {"prefix": None, "key_val_sep": "is", "text_sep": ","}

    def set_tokenizer(self, tokenizer):
        """Set the Tokenizer

        Args:
            tokenizer: Tokenizer from HuggingFace
        """
        self.tokenizer = tokenizer

        try:
            name_or_path = self.tokenizer.init_kwargs["name_or_path"].lower()
        except:
            name_or_path = "gpt2"

        # this was necessary because of llama2 which by default prepends space to input.
        # couldn't figure out a way to disable this feature
        self.tokenizer.add_tokens(self.others_token["text_sep"])

        # don't forget to
        # model.resize_token_embeddings(len(tokenizer))

        if "llama" in name_or_path:
            self.keys_token_id = {
                j: self.tokenizer.encode(f"{j}", add_special_tokens=False)
                for j in self.column_names
            }
            self.values_token_id = {
                j: self.tokenizer.encode(f"{j}", add_special_tokens=False)
                for j in list(map(str, set(self.to_pandas().values.flatten())))
            }
            self.others_token_id = {
                k: self.tokenizer.encode(f"{j}", add_special_tokens=False)
                for k, j in self.others_token.items()
            }
        else:
            self.keys_token_id = {
                j: self.tokenizer.encode(f" {j}", add_special_tokens=False)
                for j in self.column_names
            }
            self.values_token_id = {
                j: self.tokenizer.encode(f" {j}", add_special_tokens=False)
                for j in list(map(str, set(self.to_pandas().values.flatten())))
            }
            self.others_token_id = {}
            for k, v in self.others_token.items():
                if k == "text_sep":
                    self.others_token_id[k] = self.tokenizer.encode(
                        f"{v}", add_special_tokens=False
                    )
                else:
                    self.others_token_id[k] = self.tokenizer.encode(
                        f" {v}", add_special_tokens=False
                    )

    def _getitem(
        self, key: tp.Union[int, slice, str], decoded: bool = True, **kwargs
    ) -> tp.Union[tp.Dict, tp.List]:
        """Get Item from Tabular Data

        Get one instance of the tabular data, permuted, converted to text and tokenized.
        """
        # If int, what else?
        row = self._data.fast_slice(key, 1)

        self.shuffle_idx = list(range(len(self.column_names)))
        if self.shuffle:
            random.shuffle(self.shuffle_idx)

        if self.serializer == "apval":
            column_values = [str(j) for i in row.columns for j in i.to_pylist()]
            column_values = [column_values[i] for i in self.shuffle_idx]
            cat_keys = f"{self.others_token['text_sep']} ".join(
                [self.column_names[i] for i in self.shuffle_idx]
            )
            shuffled_text = " %s %s %s" % (
                cat_keys,
                self.others_token["key_val_sep"],
                f"{self.others_token['text_sep']} ".join(column_values),
            )
        else:
            # shuffle_idx = list(range(row.num_columns))
            # random.shuffle(shuffle_idx)

            if self.others_token["prefix"] is not None:
                shuffled_text = self.others_token["text_sep"].join(
                    [
                        " %s %s %s %s"
                        % (
                            self.others_token["prefix"],
                            row.column_names[i],
                            self.others_token["key_val_sep"],
                            str(row.columns[i].to_pylist()[0]).strip(),
                        )
                        for i in self.shuffle_idx
                    ]
                )
            else:
                shuffled_text = self.others_token["text_sep"].join(
                    [
                        " %s %s %s"
                        % (
                            row.column_names[i],
                            self.others_token["key_val_sep"],
                            str(row.columns[i].to_pylist()[0]).strip(),
                        )
                        for i in self.shuffle_idx
                    ]
                )
        tokenized_text = self.tokenizer(shuffled_text)
        return tokenized_text

    def __getitems__(self, keys: tp.Union[int, slice, str, list]):
        if isinstance(keys, list):
            return [self._getitem(key) for key in keys]
        else:
            return self._getitem(keys)

    def _select_contiguous(
        self,
        start: int,
        length: int,
        new_fingerprint=None,
    ):
        """
        Creates a new dataset with rows from a contiguous slice of data.

        Args:
            start (int): Start index of the slice.
            length (int): Length of the slice to select.

        Returns:
            LLMtgDataset: A new dataset instance from the specified slice.

        Raises:
            DatasetTransformationNotAllowedError: If the dataset has attached indexes.
            ValueError: If the start index or the length is out of range.
        """
        self._validate_selection(start, length)

        indices_table = None
        if self._indices is not None:
            indices_table = self._indices.slice(start, length)

        selected_table = (
            self.to_pandas().iloc[start : start + length]
            if indices_table is None
            else self.to_pandas()
        )
        new_dataset = LLMtgDataset.from_pandas(selected_table)
        new_dataset.set_serializer(self.serializer)
        new_dataset.set_tokenizer(self.tokenizer)
        new_dataset.set_shuffler(self.shuffle)

        return new_dataset

    def _validate_selection(self, start, length):
        """Validates the start index and length for dataset selection."""
        if len(self.list_indexes()) > 0:
            raise DatasetTransformationNotAllowedError(
                "Using `.select` on a dataset with attached indexes is not allowed. Run `.drop_index()` to remove your index, then re-add it."
            )
        if len(self) == 0 or length == 0:
            return
        if start < 0 or start >= len(self) or start + length > len(self):
            raise ValueError(
                "Start index and length are out of range for dataset selection."
            )


@dataclass
class TGCollator(DataCollatorWithPadding):
    def __call__(self, features):
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch["labels"] = batch["input_ids"].clone()
        if "position_ids" not in batch:
            input_ids = batch["input_ids"]
            batch["position_ids"] = torch.arange(
                input_ids.shape[1], dtype=torch.long, device=input_ids.device
            ).repeat(input_ids.shape[0], 1)
        return batch


class Deserializer:
    def __init__(self, serialization_type, column_names, dataset_object=None):
        self.serialization_type = serialization_type
        self.columns = column_names
        self.dataset_object = dataset_object

    def get_deserializer(self):
        if self.serialization_type == "great":
            return self.convert_great
        elif self.serialization_type == "text":
            return self.convert_text
        elif self.serialization_type == "apval":
            return self.convert_apval
        elif self.serialization_type == "list":
            return self.convert_list
        else:
            raise NotImplementedError

    def deserialize(self, text):
        generated = self.get_deserializer()(text, self.columns)
        df_gen = pd.DataFrame(generated)
        # print(df_gen)
        # df_gen.replace("None", None, inplace=True)
        return df_gen

    def convert_great(self, text, columns) -> pd.DataFrame:
        generated = []

        if self.dataset_object is not None:
            others_token = self.dataset_object.others_token
        else:
            others_token = {"prefix": None, "key_val_sep": "is", "text_sep": ","}

        # Convert text to tabular data
        for t in text:
            features = t.split(others_token["text_sep"])
            td = dict.fromkeys(columns, "placeholder")

            # Transform all features back to tabular data
            for f in features:
                values = f.strip().split(f" {others_token['key_val_sep']} ")
                # if values[0] in columns and td[values[0]] == "placeholder":
                if values[0] in columns:  # overrites previous values.
                    try:
                        td[values[0]] = values[1]
                    except IndexError:
                        # print("An Index Error occurred - if this happends a lot, consider fine-tuning your model further.")
                        pass
            generated.append(td)

        return generated

    def convert_text(self, text, columns) -> pd.DataFrame:
        generated = []

        if self.dataset_object is not None:
            others_token = self.dataset_object.others_token
        else:
            others_token = {"prefix": "The", "key_val_sep": "is", "text_sep": "."}

        # Convert text to tabular data
        for t in text:
            features = t.split(others_token["text_sep"])
            td = dict.fromkeys(columns, "placeholder")

            # Transform all features back to tabular data
            for f in features:
                values = (
                    f.strip()
                    .lstrip(others_token["prefix"])
                    .strip()
                    .split(f" {others_token['key_val_sep']} ")
                )
                # if values[0] in columns and td[values[0]] == "placeholder":
                if values[0] in columns:  # overrites previous values.
                    try:
                        td[values[0]] = values[1]
                    except IndexError:
                        # print("An Index Error occurred - if this happends a lot, consider fine-tuning your model further.")
                        pass
            generated.append(td)
        return generated

    def convert_list(self, text, columns) -> pd.DataFrame:
        generated = []

        if self.dataset_object is not None:
            others_token = self.dataset_object.others_token
        else:
            others_token = {"prefix": None, "key_val_sep": ":", "text_sep": "\n"}

        # Convert text to tabular data
        for t in text:
            features = t.split(others_token["text_sep"])
            td = dict.fromkeys(columns, "placeholder")
            # import pdb;pdb.set_trace()

            # Transform all features back to tabular data
            for f in features:
                values = f.strip().split(f" {others_token['key_val_sep']} ")
                # if values[0] in columns and td[values[0]] == "placeholder":
                if values[0] in columns:  # overrites previous values.
                    try:
                        td[values[0]] = values[1]
                    except IndexError:
                        # print("An Index Error occurred - if this happends a lot, consider fine-tuning your model further.")
                        pass
            generated.append(td)

        return generated

    def convert_apval(self, text, columns) -> pd.DataFrame:
        generated = []

        if self.dataset_object is not None:
            others_token = self.dataset_object.others_token
        else:
            others_token = {"prefix": None, "key_val_sep": ":", "text_sep": ","}

        # Convert text to tabular data
        for t in text:
            features = t.split(f" {others_token['key_val_sep']} ")
            td = dict.fromkeys(columns, "placeholder")

            # Transform all features back to tabular data

            keys = features[0].strip().split(others_token["text_sep"])
            if len(features) > 1:
                values = features[1].strip().split(others_token["text_sep"])
            else:
                values = None

            for index, k in enumerate(keys):
                k = k.strip()
                try:
                    v = values[index].strip()
                except:
                    v = None
                # if k in columns and td[k] == "placeholder":
                if k in columns:  # overrites previous values.
                    try:
                        td[k] = v
                    except IndexError:
                        # print("An Index Error occurred - if this happends a lot, consider fine-tuning your model further.")
                        pass
            generated.append(td)

        return generated


class Deserializer2:
    def __init__(self, serialization_type, column_names, dataset_object=None):
        self.serialization_type = serialization_type
        self.columns = column_names
        self.dataset_object = dataset_object

    def get_deserializer(self):
        if self.serialization_type == "great":
            return self.convert_great
        elif self.serialization_type == "text":
            return self.convert_text
        elif self.serialization_type == "apval":
            return self.convert_apval
        elif self.serialization_type == "list":
            return self.convert_list
        else:
            raise NotImplementedError

    def deserialize(self, text):
        generated = self.get_deserializer()(text, self.columns)
        df_gen = pd.DataFrame(generated)
        # print(df_gen)
        # df_gen.replace("None", None, inplace=True)
        return df_gen

    def convert_great(self, text, columns) -> pd.DataFrame:
        generated = []

        if self.dataset_object is not None:
            others_token = self.dataset_object.others_token
        else:
            others_token = {"prefix": None, "key_val_sep": "is", "text_sep": ","}

        # Convert text to tabular data
        for t in text:
            features = t.split(others_token["text_sep"])
            td = dict.fromkeys(columns, "placeholder")

            # Transform all features back to tabular data
            for f in features:
                values = f.strip().split(f" {others_token['key_val_sep']} ")
                if values[0] in columns and td[values[0]] == "placeholder":
                    # if values[0] in columns: # overrites previous values.
                    try:
                        td[values[0]] = values[1]
                    except IndexError:
                        # print("An Index Error occurred - if this happends a lot, consider fine-tuning your model further.")
                        pass
            generated.append(td)

        return generated

    def convert_text(self, text, columns) -> pd.DataFrame:
        generated = []

        if self.dataset_object is not None:
            others_token = self.dataset_object.others_token
        else:
            others_token = {"prefix": "The", "key_val_sep": "is", "text_sep": "."}

        # Convert text to tabular data
        for t in text:
            features = t.split(others_token["text_sep"])
            td = dict.fromkeys(columns, "placeholder")

            # Transform all features back to tabular data
            for f in features:
                values = (
                    f.strip()
                    .lstrip(others_token["prefix"])
                    .strip()
                    .split(f" {others_token['key_val_sep']} ")
                )
                # if values[0] in columns and td[values[0]] == "placeholder":
                if values[0] in columns:  # overrites previous values.
                    try:
                        td[values[0]] = values[1]
                    except IndexError:
                        # print("An Index Error occurred - if this happends a lot, consider fine-tuning your model further.")
                        pass
            generated.append(td)
        return generated

    def convert_list(self, text, columns) -> pd.DataFrame:
        generated = []

        if self.dataset_object is not None:
            others_token = self.dataset_object.others_token
        else:
            others_token = {"prefix": None, "key_val_sep": ":", "text_sep": "\n"}

        # Convert text to tabular data
        for t in text:
            features = t.split(others_token["text_sep"])
            td = dict.fromkeys(columns, "placeholder")
            # import pdb;pdb.set_trace()

            # Transform all features back to tabular data
            for f in features:
                values = f.strip().split(f" {others_token['key_val_sep']} ")
                if values[0] in columns and td[values[0]] == "placeholder":
                    # if values[0] in columns: # overrites previous values.
                    try:
                        td[values[0]] = values[1]
                    except IndexError:
                        # print("An Index Error occurred - if this happends a lot, consider fine-tuning your model further.")
                        pass
            generated.append(td)

        return generated

    def convert_apval(self, text, columns) -> pd.DataFrame:
        generated = []

        if self.dataset_object is not None:
            others_token = self.dataset_object.others_token
        else:
            others_token = {"prefix": None, "key_val_sep": ":", "text_sep": ","}

        # Convert text to tabular data
        for t in text:
            features = t.split(f" {others_token['key_val_sep']} ")
            td = dict.fromkeys(columns, "placeholder")

            # Transform all features back to tabular data

            keys = features[0].strip().split(others_token["text_sep"])
            if len(features) > 1:
                values = features[1].strip().split(others_token["text_sep"])
            else:
                values = None

            for index, k in enumerate(keys):
                k = k.strip()
                try:
                    v = values[index].strip()
                except:
                    v = None
                if k in columns and td[k] == "placeholder":
                    # if k in columns: # overrites previous values.
                    try:
                        td[k] = v
                    except IndexError:
                        # print("An Index Error occurred - if this happends a lot, consider fine-tuning your model further.")
                        pass
            generated.append(td)

        return generated


class GenerateStartTokens:
    TEMPLATES = {
        "key": {"great": "{} is", "list": "{} :", "text": "The {} is", "apval": "{} :"},
        "key_value": {
            "great": "{} is {},",
            "list": "{} : {}\n",
            "text": "The {} is {}.",
            "apval": "{} : {},",
        },
    }

    def __init__(
        self, n_samples, dataset, prompt_template=None, instruction=None, nshot=0
    ):
        self.n_samples = n_samples
        self.dataset = dataset
        self.all_columns = dataset.column_names
        self.prompt_template = prompt_template
        self.instruction = instruction
        self.nshot = nshot

    def get_template(self, tag):
        if self.prompt_template is None:
            prompt_template = self.TEMPLATES[tag].get(self.dataset.serializer, "{}")
        else:
            prompt_template = self.prompt_template
        return prompt_template

    def _pad(self, x, length, pad_value=50256):
        return [pad_value] * (length - len(x)) + x

    def _pad_tokens(self, tokens):
        max_length = len(max(tokens, key=len))
        tokens = [
            self._pad(t, max_length, self.dataset.tokenizer.pad_token_id)
            for t in tokens
        ]
        return tokens

    def get_start_tokens(self, start_prompt="random", start_col=None):
        start_text = self.get_start_text(start_prompt, start_col)

        start_tokens = self._pad_tokens(self.dataset.tokenizer(start_text)["input_ids"])
        return start_tokens

    def get_start_text(self, start_prompt="random", start_col=None):
        if start_prompt == "default":
            start_text = self.start(start_col)

        elif start_prompt == "random":
            start_text = self.random_start()

        elif start_prompt == "categorical":
            start_text = self.categorical_start(start_col)

        elif start_prompt == "continuous":
            start_text = self.continuous_start(start_col)

        elif start_prompt == "categorical_and_random":
            start_text = self.categorical_and_random_start(start_col)

        elif start_prompt == "continuous_and_random":
            start_text = self.continuous_and_random_start(start_col)
        elif start_prompt == "partial":
            start_text = self.partial_start(start_col)

        else:
            if type(start_prompt) == str:
                start_prompt = [start_prompt]

            start_prompt = [i.replace("\\n", "\n") for i in start_prompt]
            start_text = start_prompt

        start_text = self.expand_prompt(start_text)
        return start_text

    def start(self, start_col):
        n_samples = self.n_samples
        if self.dataset.serializer == "apval":
            start_words = []
            columns = [start_col] + [
                i for i in self.all_columns if i.lower() != start_col.lower()
            ]
            for i in range(n_samples):
                start_word = ", ".join(columns)
                start_words.append(start_word)
        else:
            start_words = [start_col for _ in range(n_samples)]

        prompt_template = self.get_template("key")
        start_text = [prompt_template.format(s) for s in start_words]
        return start_text

    def random_start(self):
        n_samples = self.n_samples

        if self.dataset.serializer == "apval":
            shuffle_idx = list(range(len(self.all_columns)))
            start_words = []
            for i in range(n_samples):
                random.shuffle(shuffle_idx)
                start_word = ", ".join([self.all_columns[j] for j in shuffle_idx])
                start_words.append(start_word)
        else:
            start_words = random.choices(self.all_columns, k=n_samples)

        prompt_template = self.get_template("key")
        start_text = [prompt_template.format(s) for s in start_words]
        return start_text

    def categorical_start(self, start_col):
        n_samples = self.n_samples

        metadata = get_metadata(self.dataset.to_pandas())
        assert metadata[start_col]["dtype"] == "object"
        population = metadata[start_col]["categories"]["unique"]
        weights = metadata[start_col]["categories"]["weights"]
        start_words = random.choices(population, weights, k=n_samples)

        if self.dataset.serializer == "apval":
            columns = [start_col] + [
                i for i in self.all_columns if i.lower() != start_col.lower()
            ]
            start_col = ", ".join(columns)

        prompt_template = self.get_template("key_value")
        start_text = [prompt_template.format(start_col, s) for s in start_words]
        return start_text

    def continuous_start(self, start_col, noise=0.01, decimal_places=5):
        n_samples = self.n_samples

        metadata = get_metadata(self.dataset.to_pandas())
        assert metadata[start_col]["dtype"].startswith("i")

        values = metadata[start_col]["stats"]["list"](start_col)
        start_words = random.choices(values, k=n_samples)

        if self.dataset.serializer == "apval":
            columns = [start_col] + [
                i for i in self.all_columns if i.lower() != start_col.lower()
            ]
            start_col = ", ".join(columns)

        prompt_template = self.get_template("key_value")
        start_text = [prompt_template.format(start_col, s) for s in start_words]

        return start_text

    def categorical_and_random_start(self, start_col):
        start_text1 = self.categorical_start(start_col)

        if self.dataset.serializer == "apval":
            return start_text1

        start_text2 = self.random_start()
        start_text = [i + " " + j for i, j in zip(start_text1, start_text2)]

        return start_text

    def continuous_and_random_start(self, start_col):
        start_text1 = self.continuous_start(start_col)

        if self.dataset.serializer == "apval":
            return start_text1

        start_text2 = self.random_start()
        start_text = [i + " " + j for i, j in zip(start_text1, start_text2)]

        return start_text

    def partial_start(self, df):
        conditions = list(df.apply(self._encode_row_partial, axis=1))
        if self.dataset.serializer == "apval":
            return conditions
        else:
            start_cols = list(df.apply(self._get_random_missing, axis=1))
            self.n_samples = 1
            start_texts = [
                self.get_start_text("default", start_col)[0] for start_col in start_cols
            ]
            prompt_text = [i + " " + j for i, j in zip(conditions, start_texts)]
            return prompt_text

    def _get_random_missing(self, row):
        """Return a random missing column or None if all columns are filled."""
        nans = list(row[pd.isna(row)].index)
        return np.random.choice(nans) if len(nans) > 0 else None

    def _encode_row_partial(self, row, shuffle=True):
        prompt_template = self.get_template("key_value")

        num_cols = len(row.index)
        if not shuffle:
            idx_list = np.arange(num_cols)
        else:
            idx_list = np.random.permutation(num_cols)

        if self.dataset.serializer == "apval":
            keys_values = list(
                zip(
                    *[
                        (row.index[j], str(row[row.index[j]]))
                        for j in idx_list
                        if not pd.isna(row[row.index[j]])
                    ]
                )
            )
            nans = tuple(row[pd.isna(row)].index)
            keys = ", ".join(keys_values[0] + nans)
            values = ", ".join(keys_values[1])
            lists = prompt_template.format(keys, values)

        else:
            lists = " ".join(
                sum(
                    [
                        [prompt_template.format(row.index[i], row[row.index[i]])]
                        if not pd.isna(row[row.index[i]])
                        else []
                        for i in idx_list
                    ],
                    [],
                )
            )
        return lists

    def expand_prompt(self, start_text):
        if self.instruction is not None and self.nshot > 0:
            examples = [
                self.dataset.tokenizer.decode(i["input_ids"])
                for i in self.dataset.__getitems__(list(range(self.nshot)))
            ]
            examples = "\n".join(examples)

            start_text = [
                self.instruction.format(examples) + text for text in start_text
            ]

        elif self.instruction is not None:
            start_text = [self.instruction + text for text in start_text]

        elif self.nshot > 0:
            examples = [
                self.dataset.tokenizer.decode(i["input_ids"])
                for i in self.dataset.__getitems__(list(range(self.nshot)))
            ]
            examples = "\n".join(examples)

            start_text = [examples + "\n" + text for text in start_text]

        return start_text


def postprocess_data(table, metadata=None, dropna=False):
    """
    Post-processes a pandas DataFrame based on provided metadata.

    Parameters:
    table (pandas.DataFrame): The DataFrame to be processed.
    metadata (dict, optional): Metadata containing information about data types and categories for DataFrame columns.

    Returns:
    pandas.DataFrame: The processed DataFrame.
    """
    df = table.copy()

    def _numeric_check(x):
        try:
            return float(x)
        except:
            return np.nan

    if metadata:
        column_names = [col.lower() for col in df.columns]
        assert sorted(column_names) == sorted(
            metadata.keys()
        ), "DataFrame columns and metadata keys must match."
        df.columns = column_names

        for col in column_names:
            col_metadata = metadata[col]
            if col_metadata["dtype"] == "object":
                categories = col_metadata["categories"]["unique"]
                case_function = col_metadata["categories"]["case"]
                df[col] = df[col].apply(
                    lambda x: case_function.get(str(x).lower(), str.lower)(x)
                )
                condition = df[col].isin(categories)
                df[col] = df[col].where(condition)
            else:
                df[col] = df[col].apply(_numeric_check)
                try:
                    df[col] = df[col].astype(col_metadata["dtype"])
                except:
                    continue

            if dropna:
                df = df.dropna()
                df[col] = df[col].astype(col_metadata["dtype"])

        # Update column names to match case specified in metadata
        df.columns = [metadata[col]["case"](col) for col in column_names]

    # Replace "None" with NaN
    df = df.replace("None", np.nan)

    if dropna:
        df = df.dropna()

    return df.reset_index(drop=True)


def get_word_case(word):
    """
    Determines the case of a given word and returns an appropriate string method.

    :param word: A string whose case is to be determined.
    :return: A string method corresponding to the detected case.
    """
    word = str(word)
    if word.isupper():
        return str.upper
    elif word.islower():
        return str.lower
    elif word.istitle():
        return str.title
    elif word[0].isupper() and not word.isalpha():
        return str.capitalize
    else:
        return str.lower


def get_metadata(data):
    """
    Generates metadata for each column in a pandas DataFrame.

    :param data (pandas.DataFrame): A pandas DataFrame.
    :return: A dictionary containing metadata for each column.
            For each column, the metadata includes data type ('dtype'),
            the case of the column name ('case'), and for columns with
            object data types, unique categories ('categories') and their cases.
    """

    metadata = {}
    for col in data.columns:
        col = col.lower()
        metadata[col] = {
            "dtype": str(data[col].dtypes),
            "case": get_word_case(col),
        }

        if data[col].dtypes == object:
            # unique_categories = list(data[col].unique())
            value_counts = data[col].value_counts(normalize=True)
            unique_categories = list(value_counts.index)
            weights = list(value_counts.values)
            stats = data[[col]].describe(include=["O"]).to_dict()[col]
            stats.pop("count")
            metadata[col]["stats"] = stats
            metadata[col]["categories"] = {
                "unique": unique_categories,
                "weights": weights,
                "case": {cat.lower(): get_word_case(cat) for cat in unique_categories},
            }
        else:
            metadata[col]["stats"] = {
                "min": data[col].min(),
                "max": data[col].max(),
                "mean": data[col].mean(),
                "std": data[col].std(),
                "list": lambda x: data[x].to_list(),
            }

    return metadata


def convert_tokens_to_text(tokens, tokenizer):
    """Decodes the tokens back to strings

    Args:
        tokens: List of tokens to decode
        tokenizer: Tokenizer used for decoding

    Returns:
        List of decoded strings
    """
    # Convert tokens to text
    text_data = [tokenizer.decode(t, skip_special_tokens=True) for t in tokens]

    # import pdb; pdb.set_trace()
    # print(text_data)
    # Clean text
    # text_data = [d.replace("<|endoftext|>", "") for d in text_data]
    # text_data = [d.replace("\n", " ") for d in text_data]
    text_data = [d.lstrip("\n") for d in text_data]
    # text_data = [d.lstrip("\\n") for d in text_data]
    text_data = [d.lstrip("\r") for d in text_data]
    # text_data = [d.replace("\r", "") for d in text_data]

    return text_data


def test_dataset():
    from transformers import AutoTokenizer
    import pandas as pd
    import random

    random.seed(42)

    llm = "gpt2"
    # llm = "meta-llama/Llama-2-7b-hf"
    tokenizer = AutoTokenizer.from_pretrained(llm)
    tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.add_prefix_space = False
    # print(tokenizer.add_prefix_space)

    df = pd.read_csv("./data/adult/train.csv")
    great_ds = LLMtgDataset.from_pandas(df)

    great_ds.set_serializer()
    great_ds.set_tokenizer(tokenizer)
    great_ds.set_shuffler(shuffle=True)

    print(great_ds.column_names)

    tokens = [great_ds[i]["input_ids"] for i in range(1)]
    print(tokens)

    decoded_tokens = [
        great_ds.tokenizer.decode(j, skip_special_tokens=True) for j in tokens
    ]
    print(decoded_tokens)

    value = (
        decoded_tokens[0]
        .split(great_ds.others_token["text_sep"])[0]
        .split(great_ds.others_token["key_val_sep"])[1]
        .strip()
    )
    ###
    # This block validates that the tokenized key, values, others
    # matches how it's tokenized as part of a sentence.
    for k, v in great_ds.keys_token_id.items():
        assert v[0] in tokens[0]

    great_ds.values_token_id[value] in tokens[0]
    great_ds.others_token_id["key_val_sep"] in tokens[0]

    ####

    print(great_ds.keys_token_id)
    print(great_ds.tokenizer.encode("isis"))
    print(great_ds.tokenizer.encode("is is"))
    print(great_ds.tokenizer.encode("isis terrorist"))
    print(great_ds.tokenizer.encode("ageage"))
    print(great_ds.tokenizer.encode("age age"))
    print(
        great_ds.tokenizer.decode(
            tokenizer.encode("age is 32, age is 32"), skip_special_tokens=True
        )
    )
    # to ensure that tokens are the same, prepend with a space.
    tokens = great_ds.tokenizer.encode(" sex is Male, sex is Male. sex is ?, sex is NA")
    print(tokens)
    print(
        tokenizer.decode(
            [
                1714,
                318,
                12674,
                11,
                1714,
                318,
                12674,
                13,
                1714,
                318,
                5633,
                11,
                1714,
                318,
                11746,
            ]
        )
    )
    # "sex is Male, sex is Male. sex is?, sex is NA"


def test_data_collator():
    from transformers import AutoTokenizer
    import pandas as pd
    from torch.utils.data import DataLoader

    llm = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(llm)
    tokenizer.pad_token = tokenizer.eos_token
    df = pd.read_csv("./data/adult/train.csv")
    great_ds = LLMtgDataset.from_pandas(df)
    great_ds.set_serializer()
    great_ds.set_tokenizer(tokenizer)
    great_ds.set_shuffler(shuffle=True)
    dataloader = DataLoader(
        great_ds,
        shuffle=True,
        collate_fn=DataCollator(tokenizer),
        batch_size=5,
    )
    for i in dataloader:
        print(i["input_ids"])
        print(len(i["input_ids"]))
        break


def test_apval_serialize():
    from transformers import AutoTokenizer
    import pandas as pd
    from torch.utils.data import DataLoader

    llm = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(llm)
    tokenizer.pad_token = tokenizer.eos_token
    df = pd.read_csv("./data/adult/train.csv")
    ds = LLMtgDataset.from_pandas(df)
    ds.set_serializer("apval")
    # ds.set_serializer("list")
    ds.set_tokenizer(tokenizer)
    ds.set_shuffler(shuffle=False)

    print(ds.others_token)
    print(ds.others_token_id)
    out = ds.__getitems__([0, 1])
    print(out)
    for i in out:
        print(ds.tokenizer.decode(i["input_ids"]))
    # print(ds.tokenizer.decode([2479, 11, 670, 4871, 11, 277, 21283, 86, 13655, 11, 3707, 11, 3707, 12, 22510, 11, 29555, 12, 13376, 11, 13755, 11, 2776, 11, 3234, 11, 1714, 11, 3139, 12, 48544, 11, 3139, 12, 22462, 11, 2250, 12, 525, 12, 10464, 11, 6868, 12, 19315, 11, 3739, 1058, 5014, 11, 1812, 12, 9567, 11, 767, 2425, 1433, 11, 347, 9636, 669, 11, 1511, 11, 7236, 12, 30526, 11, 1215, 76, 12, 22902, 605, 11, 1892, 12, 259, 12, 17989, 11, 2635, 11, 12674, 11, 362, 22985, 11, 657, 11, 2319, 11, 1578, 12, 27219, 11, 19841, 1120, 74]))


def test_tabular_data_conversion():
    from transformers import AutoTokenizer
    import pandas as pd
    from torch.utils.data import DataLoader
    import pandas as pd

    # from misc import get_metadata

    llm = "gpt2"
    serialization_type = "apval"
    tokenizer = AutoTokenizer.from_pretrained(llm)
    tokenizer.pad_token = tokenizer.eos_token
    df = pd.read_csv("./data/adult/train.csv")
    ds = LLMtgDataset.from_pandas(df)
    ds.set_serializer(serialization_type)

    ds.set_tokenizer(tokenizer)
    ds.set_shuffler(shuffle=False)

    out = ds.__getitems__([0, 1])
    output_text = [ds.tokenizer.decode(i["input_ids"]) for i in out]
    print(output_text[0])
    # output_text = ["income is related to income, education, occupation, marital status, education-level, occupation, capital-gain, education-num, marital-status, occupation, capital-loss, capital-gain, education-state, capital-loss, marital-status, capital-loss, capital-gain, education-num, capital-loss, capital-gain, capital-loss, capital-loss, capital-gain, capital-loss, capital-loss, capital-loss, capital-loss, capital-gain, capital-loss, capital-loss, capital-loss, capital-gain, capital-loss, capital-loss, capital-loss, capital-gain, capital-loss, capital-loss, capital-loss, capital-loss, capital-loss, capital-loss, capital-gain, capital-loss, capital-loss, capital-loss, capital-loss, capital-loss, capital-loss, capital-loss, capital-loss, capital-loss, capital-loss, capital-loss, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0"]
    # output_text = ["education, occupation, marital-status, education-level, occupation, capital-gain, education-num, marital-status, occupation, capital-loss, capital-gain, education-state, capital-loss, marital-status, capital-loss, capital-gain, education-num, capital-loss, capital-gain, capital-loss, capital-loss, capital-gain, capital-loss, capital-loss, capital-loss, capital-loss, capital-gain, capital-loss, capital-loss, capital-loss, capital-gain, capital-loss, capital-loss, capital-loss, capital-gain, capital-loss, capital-loss, capital-loss, capital-loss, capital-loss, capital-loss, capital-gain, capital-loss, capital-loss, capital-loss, capital-loss, capital-loss, capital-loss, capital-loss, capital-loss, capital-loss, capital-loss, capital-loss, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0"]
    # output_text = ['education, occupation, marital-status, education-num : bachelors, adm-clerical, Never-Married, 13']
    print(output_text)

    deserializer = Deserializer(serialization_type, ds.column_names, ds)
    output_table = deserializer.deserialize(output_text)

    metadata = get_metadata(ds.to_pandas())
    if metadata is not None:
        output_table = postprocess_data(output_table, metadata)

    print(output_table)
    print(ds.to_pandas().loc[[0, 1], :])
    assert output_table.equals(ds.to_pandas().loc[[0, 1], :])


# if __name__ == "__main__":
#     # test_dataset()
#     # test_data_collator()
#     test_append_serialize()
# test_tabular_data_conversion()
