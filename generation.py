import os
import time
import pandas as pd
import torch
from tqdm import tqdm
import random
import numpy as np

from utils.misc import mkdir
from utils.dataset import (
    Deserializer,
    postprocess_data,
    GenerateStartTokens,
    convert_tokens_to_text,
    get_metadata,
)
from transformers import set_seed


def generate(
    n_samples,
    model,
    dataset,
    start_prompt="random",
    start_col=None,
    temperature=0.7,
    top_p=1.0,
    k=100,
    max_length=100,
    drop_nan=False,
    do_impute=False,
    prompt_template=None,
    device="cuda",
    max_retries=5,
    max_allowed_time=300,
    save_folder="./synth_data",
    save_name="0",
    seed=1000,
):
    # print(max_length)
    mkdir(save_folder)
    mkdir(save_folder + "/raw_texts")

    if not os.path.exists(f"{save_folder}/elapsed_time.txt"):
        with open(f"{save_folder}/elapsed_time.txt", "w") as f:
            f.write("save_name,secs,mins,hrs\n")

    # Define the file path where the seeds were saved
    save_seed_file = f"{save_folder}/random_seeds_{save_name}.pt"

    save_raw_text = f"{save_folder}/raw_texts/{save_name}.txt"
    save_prompt = f"{save_folder}/prompt_{save_name}.txt"
    total_sampled_so_far = 0
    total_sampled = 0
    total_processed = 0
    total_expected = n_samples

    if not os.path.exists(save_seed_file):
        set_seed(seed)
        with open(save_raw_text, "w") as f:
            # clear existing text
            pass

        with open(save_prompt, "w") as f:
            # clear existing text
            pass
    else:
        load_random_seeds(save_seed_file)
        with open(save_raw_text, "r") as f:
            total_sampled_so_far = len(f.readlines())

        if os.path.exists(f"{save_folder}/raw_tables/{save_name}.csv"):
            existing_tables = pd.read_csv(f"{save_folder}/raw_tables/{save_name}.csv")
            total_sampled = len(existing_tables)

        if os.path.exists(f"{save_folder}/processed_tables/{save_name}.csv"):
            existing_tables = pd.read_csv(
                f"{save_folder}/processed_tables/{save_name}.csv"
            )
            total_processed = len(existing_tables)
            n_samples = n_samples - total_processed

    if n_samples == 0:
        raise Exception("Breaking -- All done champ!")

    if max_allowed_time is not None:
        max_allowed_time = int(max_allowed_time)

    if not os.path.exists(f"{save_folder}/stats.txt"):
        with open(f"{save_folder}/stats.txt", "w") as f:
            f.write("name,elapsed_time,total_sampled,total_processed,total_expected")
            f.write("\n")

    output_texts = []
    raw_tables = []
    processed_tables = []

    metadata = get_metadata(dataset.to_pandas())
    start_time = time.time()

    # total_na = 0

    # import pdb; pdb.set_trace()

    with tqdm(total=n_samples) as pbar:
        already_generated = 0
        _cnt = 0

        # if k > n_samples:
        #     k=n_samples

        # sample_batch = min(k, (n_samples-already_generated))

        try:
            while already_generated < n_samples:
                start_tokens = GenerateStartTokens(
                    min(k, (n_samples - already_generated)),
                    dataset,
                    prompt_template=prompt_template,
                )

                input_batch = start_tokens.get_start_tokens(
                    start_prompt=start_prompt,
                    start_col=start_col,
                )
                # import pdb;
                # pdb.set_trace()

                # print(input_batch.shape)

                raw_table, output_text = sample(
                    model=model,
                    starting_prompts=input_batch,
                    dataset=dataset,
                    temperature=temperature,
                    max_length=max_length,
                    device=device,
                    save_path=save_raw_text,
                    save_prompt=save_prompt,
                    top_p=top_p,
                )

                raw_tables.append(raw_table)
                output_texts.extend(output_text)

                processed_table = postprocess_data(
                    raw_table, metadata=metadata, dropna=False
                )
                df_miss = processed_table.loc[
                    processed_table.isna().any(axis=1).values
                ].reset_index(drop=True)

                # total_na += len(df_miss)
                total_sampled += len(output_text)

                if do_impute:
                    if len(df_miss) != 0:
                        df_notna = processed_table.loc[
                            processed_table.notna().all(axis=1).values
                        ].reset_index(drop=True)

                        df_impute, total_retries, df_imputed_track = impute(
                            model,
                            df_miss,
                            dataset,
                            temperature=temperature,
                            max_length=max_length,
                            max_retries=max_retries,
                            device=device,
                            save_path=save_raw_text,
                            save_prompt=save_prompt,
                            start_time=start_time,
                            max_allowed_time=max_allowed_time,
                            top_p=top_p,
                        )

                        processed_table = pd.concat(
                            [df_notna, df_impute], axis=0
                        ).reset_index(drop=True)

                        raw_tables.append(df_imputed_track)

                        already_generated += len(processed_table)
                        total_sampled += total_retries

                        _cnt += total_retries

                    else:
                        already_generated += len(processed_table)

                elif drop_nan:
                    processed_table = processed_table.dropna()
                    already_generated += len(processed_table)
                    _cnt += 1

                else:
                    already_generated += len(raw_table)
                    processed_table = []
                    _cnt += 1

                if len(processed_table) != 0:
                    total_processed += len(processed_table)
                    processed_tables.append(processed_table)

                # _cnt += 1
                save_data(
                    output_texts,
                    raw_table,
                    processed_table,
                    save_folder,
                    save_name,
                    seed,
                    time.time(),
                    total_sampled_so_far + total_sampled,
                    total_expected,
                )

                if max_allowed_time is not None:
                    time_so_far = time.time() - start_time
                    print(time_so_far, max_allowed_time)
                    if time_so_far >= max_allowed_time:
                        raise Exception("Breaking the generation loop!")

                else:
                    if _cnt > max_retries and already_generated == 0:
                        raise Exception("Breaking the generation loop!")

                if drop_nan or do_impute:
                    pbar.update(len(processed_table))

                else:
                    pbar.update(len(raw_table))

        except Exception as e:
            print(e)

    time_elapsed = time.time() - start_time

    with open(f"{save_folder}/elapsed_time.txt", "a") as f:
        f.write(f"{save_name},{time_elapsed},{time_elapsed/60},{time_elapsed/3600}\n")

    if len(raw_tables) != 0:
        raw_tables = pd.concat(raw_tables).reset_index(drop=True)
        incomplete_generation = raw_tables.loc[
            (raw_tables == "placeholder").any(axis=1).values
        ]
    else:
        incomplete_generation = pd.DataFrame([])

    if len(processed_tables) != 0:
        processed_tables = pd.concat(processed_tables).reset_index(drop=True)

    # with open(f"{save_folder}/stats.txt", "a") as f:
    #     f.write(
    #         f"{save_name},{time_elapsed},{total_sampled},{total_processed},{len(incomplete_generation)}"
    #     )
    #     f.write("\n")

    if drop_nan or do_impute:
        return processed_tables
    else:
        return raw_tables


def sample(
    model,
    starting_prompts,
    dataset,
    temperature=0.7,
    top_p=1.0,
    max_length=100,
    device="cuda",
    save_path=None,
    save_prompt=None,
):
    if save_prompt is not None:
        with open(save_prompt, "a", encoding="utf-8") as f:
            for i in starting_prompts:
                decode = dataset.tokenizer.decode(i)
                f.write(decode + "\n")

    input_ids = torch.Tensor(starting_prompts).long().to(device)
    model.to(device)

    if input_ids.dim() < 2:
        input_ids = torch.unsqueeze(input_ids, 0)

    output_token = model.generate(
        input_ids=input_ids,
        pad_token_id=dataset.tokenizer.pad_token_id,
        max_new_tokens=max_length,
        # max_length=max_length,
        temperature=temperature,
        do_sample=True,
        top_p=top_p,
        # repetition_penalty=1.0,
        # top_k=1
    )
    output_token = list(output_token.detach().cpu().numpy())
    output_text = convert_tokens_to_text(output_token, dataset.tokenizer)

    if save_path is not None:
        with open(save_path, "a", encoding="utf-8") as f:
            for i in output_text:
                f.write(i + "\n")

    deserializer = Deserializer(
        dataset.serializer,
        dataset.column_names,
        dataset,
    )

    raw_table = deserializer.deserialize(output_text)

    return raw_table, output_text


def impute(
    model,
    df_miss,
    dataset,
    temperature=0.7,
    top_p=1.0,
    max_length=100,
    max_retries=5,
    device="cuda",
    save_path=None,
    save_prompt=None,
    start_time=None,
    max_allowed_time=None,
):
    if start_time is not None and max_allowed_time is not None:
        time_so_far = time.time() - start_time
        print(time_so_far, max_allowed_time)
        if time_so_far >= max_allowed_time:
            raise Exception("Breaking the generation loop!")

    column_names = dataset.column_names
    if set(df_miss.columns) != set(column_names):
        raise ValueError(
            "The column nam es in the DataFrame passed to impute do not match that used to train the model."
        )

    model.to(device)
    metadata = get_metadata(dataset.to_pandas())
    index = 0
    df_list = []
    df_imputed_track = []
    total_retries = 0
    with tqdm(total=len(df_miss)) as pbar:
        while index < len(df_miss):
            is_complete = False
            retries = 0
            # import pdb; pdb.set_trace()
            df_curr = df_miss.loc[[index]]
            org_index = df_curr.index
            while not is_complete:
                num_attrs_missing = pd.isna(df_curr).sum().sum()
                starting_prompts = GenerateStartTokens(1, dataset).get_start_tokens(
                    "partial", df_curr
                )

                df_curr, _ = sample(
                    model,
                    starting_prompts,
                    dataset,
                    temperature,
                    top_p,
                    max_length,
                    device,
                    save_path=save_path,
                    save_prompt=save_prompt,
                )
                df_imputed_track.append(df_curr)
                total_retries += 1

                df_curr = postprocess_data(df_curr, metadata=metadata, dropna=False)

                # Check for missing values
                if not df_curr.isna().any().any():
                    is_complete = True
                    df_list.append(df_curr.set_index(org_index))
                else:
                    retries += 1

                if retries == max_retries:
                    print("Max retries reached.")
                    break

            index += 1
            pbar.update(1)
    return (
        pd.concat(df_list, axis=0),
        total_retries,
        pd.concat(df_imputed_track, axis=0),
    )


def save_data(
    output_texts,
    raw_tables,
    processed_tables,
    save_folder,
    save_name,
    seed,
    time_elapsed,
    total_generated,
    total_expected,
):
    # raw_text_folder = f"{save_folder}/raw_texts"
    processed_table_folder = f"{save_folder}/processed_tables"
    raw_table_folder = f"{save_folder}/raw_tables"

    mkdir(processed_table_folder)
    mkdir(raw_table_folder)

    # with open(raw_text_folder + f"/{save_name}.txt", "a") as f:
    #     for i in output_texts:
    #         f.write(i)
    #         f.write("\n")

    if len(raw_tables) != 0:
        # raw_tables = pd.concat(raw_tables)
        raw_table_file = raw_table_folder + f"/{save_name}.csv"
        if os.path.exists(raw_table_file):
            existing_raw_tables = pd.read_csv(raw_table_file)
            raw_tables = pd.concat([existing_raw_tables, raw_tables], ignore_index=True)
        raw_tables.to_csv(raw_table_folder + f"/{save_name}.csv", index=None)

    if len(processed_tables) != 0:
        # processed_tables = pd.concat(processed_tables)
        processed_table_file = processed_table_folder + f"/{save_name}.csv"
        if os.path.exists(processed_table_file):
            existing_processed_tables = pd.read_csv(processed_table_file)
            processed_tables = pd.concat(
                [existing_processed_tables, processed_tables], ignore_index=True
            )
        processed_tables.to_csv(
            processed_table_folder + f"/{save_name}.csv", index=None
        )

    save_current_random_seeds(seed, save_folder, save_name)

    with open(f"{save_folder}/stats.txt", "a") as f:
        f.write(
            f"{save_name},{time_elapsed},{total_generated},{len(processed_tables)},{total_expected}"
        )
        f.write("\n")


def save_current_random_seeds(seed, save_folder, save_name):
    """Save random seeds as a .pt file, including manual seed, Python, NumPy, and PyTorch states."""

    # Get current random seed states
    python_seed = random.getstate()
    numpy_seed = np.random.get_state()
    torch_seed = torch.get_rng_state()
    cuda_seed = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    # Prepare the seed data
    seed_data = {
        "manual_seed": seed,  # Save the manually set seed
        "python_seed": python_seed,  # Python seed state
        "numpy_seed": numpy_seed,  # NumPy seed state
        "torch_seed": torch_seed,  # PyTorch CPU seed state
        "cuda_seed": cuda_seed,  # PyTorch CUDA seed states (if any)
    }

    # Save the seed data to a .pt file using torch.save()
    save_seed_file = os.path.join(save_folder, f"random_seeds_{save_name}.pt")
    torch.save(seed_data, save_seed_file)
    print(f"Random seeds saved at {save_seed_file}")


def load_random_seeds(save_seed_file):
    """Load random seeds from a .pt file and restore the manual seed and random states for Python, NumPy, and PyTorch."""

    # Define the file path
    # save_seed_file = os.path.join(save_folder, f"random_seeds_{save_name}.pt")

    if not os.path.exists(save_seed_file):
        raise FileNotFoundError(f"Seed file not found: {save_seed_file}")

    # Load the seeds from the .pt file using torch.load()
    seed_data = torch.load(save_seed_file)

    # Restore the manual seed
    manual_seed = seed_data["manual_seed"]
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(manual_seed)
    print(f"Manual seed restored: {manual_seed}")

    # Restore Python's random seed
    python_seed = seed_data["python_seed"]
    random.setstate(python_seed)
    print("Python random seed restored.")

    # Restore NumPy's random seed
    numpy_seed = seed_data["numpy_seed"]
    np.random.set_state(numpy_seed)
    print("NumPy random seed restored.")

    # Restore PyTorch CPU seed
    torch_seed = seed_data["torch_seed"]
    torch.set_rng_state(torch_seed)
    print("PyTorch CPU random seed restored.")

    # Restore PyTorch CUDA seed (if available)
    cuda_seed = seed_data["cuda_seed"]
    if cuda_seed is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_seed)
        print("PyTorch CUDA random seed restored.")

    print("Random seed states successfully loaded and restored.")


def test_get_start_token():
    from transformers import AutoTokenizer
    import pandas as pd
    from torch.utils.data import DataLoader
    import pandas as pd

    # from misc import get_metadata
    from dataset_v2 import LLMtgDataset, get_metadata

    llm = "gpt2"
    # llm = "meta-llama/Llama-2-7b-hf"
    serialization_type = "great"
    # serialization_type="apval"
    tokenizer = AutoTokenizer.from_pretrained(llm)
    tokenizer.pad_token = tokenizer.eos_token
    df = pd.read_csv("./data/adult/train.csv")
    ds = LLMtgDataset.from_pandas(df)
    ds.set_serializer(serialization_type)

    print(tokenizer.bos_token_id)
    ds.set_tokenizer(tokenizer)
    ds.set_shuffler(shuffle=False)
    # start_tokens=GenerateStartTokens(ds, instruction="You are an expert tabular data generator. Generate tables with these columns name: ", nshot=2)
    start_tokens = GenerateStartTokens(
        5,
        ds,
        # prompt_template="The {} is {},",
        # instruction="You are an expert tabular data generator. Here are some examples {}. Generate a new example following the column names: ",
        # nshot=1,
    )

    # print(start_tokens.start('age'))
    # print(start_tokens.random_start())
    # print(start_tokens.categorical_start("income"))
    # print(start_tokens.continuous_start("age"))
    # print(start_tokens.categorical_and_random_start("income"))

    print(start_tokens.get_start_text("income is"))
    print(start_tokens.get_start_tokens("income"))

    # print(start_tokens.get_start_text("random"))
    # print(start_tokens.get_start_text("default", "age"))
    # print(start_tokens.get_start_text("categorical", "income"))
    # print(start_tokens.get_start_text("continuous", "age"))
    # print(start_tokens.get_start_text("categorical_and_random", "income"))
    # print(start_tokens.get_start_text("continuous_and_random", "age"))

    # print(start_tokens.get_start_tokens("random", None))
    # print(start_tokens.get_start_tokens("categorical", "income"))


def test_partial():
    from transformers import AutoTokenizer
    import pandas as pd
    from torch.utils.data import DataLoader
    import pandas as pd

    # from misc import get_metadata
    from dataset_v2 import LLMtgDataset, get_metadata
    import numpy as np

    llm = "gpt2"
    # llm = "meta-llama/Llama-2-7b-hf"
    serialization_type = "apval"
    # serialization_type="apval"
    tokenizer = AutoTokenizer.from_pretrained(llm)
    tokenizer.pad_token = tokenizer.eos_token
    df = pd.read_csv("./data/adult/train.csv")
    ds = LLMtgDataset.from_pandas(df)
    ds.set_serializer(serialization_type)

    print(tokenizer.bos_token_id)
    ds.set_tokenizer(tokenizer)
    ds.set_shuffler(shuffle=False)

    df_curr = ds.to_pandas().head(2)
    df_curr.loc[0, "age"] = np.nan
    df_curr.loc[0, "income"] = np.nan
    df_curr.loc[1, "income"] = np.nan
    df_curr.loc[1, "sex"] = np.nan
    print(df_curr)
    # print(df_curr.isna())

    # print(df_curr[~df_curr.isna()])

    # for i in df_curr.iterrows():
    #     print(i[1].index)
    #     print("**")

    # out=df_curr.apply(_encode_row_partial , axis=1)
    # print(out)

    start_tokens = GenerateStartTokens(
        1,
        ds,
        # prompt_template="The {} is {},",
        # instruction="You are an expert tabular data generator. Here are some examples {}. Generate a new example following the column names: ",
        # nshot=1,
    )

    print(start_tokens.partial_start(df_curr))
    print(start_tokens.get_start_tokens("partial", df_curr))
    print(
        tokenizer.decode(
            [
                50256,
                50256,
                50256,
                50256,
                50256,
                40796,
                11,
                2250,
                12,
                525,
                12,
                10464,
                11,
                3234,
                11,
                2776,
                11,
                3707,
                12,
                22510,
                11,
                6868,
                12,
                19315,
                11,
                670,
                4871,
                11,
                277,
                21283,
                86,
                13655,
                11,
                3139,
                12,
                48544,
                11,
                1714,
                11,
                29555,
                12,
                13376,
                11,
                13755,
                11,
                3139,
                12,
                22462,
                11,
                2479,
                11,
                3739,
                1058,
                347,
                9636,
                669,
                11,
                2319,
                11,
                2635,
                11,
                1892,
                12,
                259,
                12,
                17989,
                11,
                1511,
                11,
                1578,
                12,
                27219,
                11,
                1812,
                12,
                9567,
                11,
                767,
                2425,
                1433,
                11,
                362,
                22985,
                11,
                12674,
                11,
                7236,
                12,
                30526,
                11,
                1215,
                76,
                12,
                22902,
                605,
                11,
                657,
                11,
            ]
        )
    )

    # num_attrs_missing = pd.isna(df_curr).sum().sum()
    # print(num_attrs_missing)

    # print(df_curr.apply(_get_random_missing, axis=1))

    # starting_prompts = _partial_df_to_promts(df_curr)
    # print(starting_prompts)


def test_sample():
    from transformers import AutoTokenizer
    import pandas as pd
    from torch.utils.data import DataLoader
    import pandas as pd

    from utils.dataset import LLMtgDataset
    from ft_opacus import get_model

    # sys.path.append("/home/teju/Documents/Projects/llm-tg/utils")

    llm = "gpt2"
    # llm = "meta-llama/Llama-2-7b-hf"
    # serialization_type = "apval"
    serialization_type = "great"
    tokenizer = AutoTokenizer.from_pretrained(llm)
    tokenizer.pad_token = tokenizer.eos_token
    df = pd.read_csv("./data/adult/train.csv")
    ds = LLMtgDataset.from_pandas(df)
    ds.set_serializer(serialization_type)

    print(tokenizer.bos_token_id)
    ds.set_tokenizer(tokenizer)
    ds.set_shuffler(shuffle=False)

    checkpoint_path = "/p/home/jusers/afonja1/juwels/teju/llm-tg/llm_tg/runs/15.07.24/adult/NonDP/GPT2/entire/ts30932-bs32-epoch20/model.safetensors"
    model = get_model(
        llm,
        finetune_type="entire",
        model_type=llm,
        checkpoint_path=checkpoint_path,
        tokenizer=ds.tokenizer,
    )
    # print(model)

    # start_tokens = GenerateStartTokens(
    #     10,
    #     ds,
    #     # prompt_template="The {} is {},",
    #     # instruction="You are an expert tabular data generator. Here are some examples {}. Generate a new example following the column names: ",
    #     # nshot=1,
    # )
    # starting_prompts = start_tokens.get_start_tokens("random")
    # out_table = sample(model, starting_prompts, ds, temperature=0.7, max_length=100, device="cuda")
    # print(out_table[0])

    # metadata=get_metadata(ds.to_pandas())
    # processed_table = postprocess_data(out_table[0], metadata=metadata, dropna=False)
    # print(processed_table)

    # # print(processed_table.isna().any(axis=1).values)
    # df_miss = processed_table.loc[processed_table.isna().any(axis=1).values].reset_index(drop=True)
    # print(df_miss)
    # df_impute = impute(model, df_miss, ds, temperature=0.7, max_length=100, max_retries=15, device="cuda")
    # print(df_impute)

    sample = generate(
        n_samples=500,
        model=model,
        dataset=ds,
        start_prompt="random",
        start_col=None,
        temperature=0.7,
        k=100,
        max_length=100,
        drop_nan=False,
        do_impute=True,
        prompt_template=None,
        device="cuda",
        max_retries=15,
        max_allowed_time=5,
        save_folder="./testing_synth",
        save_name="2",
    )


def main():
    from ft_opacus import parse_args, set_seed, get_dataset, get_model, get_tokenizer

    args = parse_args()

    print(args)
    # set_seed()

    tokenizer_name = (
        args.tokenizer_name
        if args.tokenizer_name is not None
        else args.model_name_or_path
    )

    tokenizer = get_tokenizer(
        tokenizer_name,
        cache_dir=args.cache_dir,
        use_slow_tokenizer=args.use_slow_tokenizer,
        trust_remote_code=args.trust_remote_code,
    )

    train_dataset, validation_dataset = get_dataset(
        args,
        tokenizer=tokenizer,
    )

    n_samples = (
        args.n_synth_samples if args.n_synth_samples is not None else len(train_dataset)
    )

    model = get_model(
        args.model_name_or_path,
        finetune_type="entire",
        model_type=args.model_type,
        checkpoint_path=args.checkpoint_path,
        tokenizer=train_dataset.tokenizer,
    )

    sample = generate(
        n_samples=n_samples,
        model=model,
        dataset=train_dataset,
        start_prompt=args.start_prompt,
        start_col=args.start_col,
        temperature=args.temperature,
        top_p=args.top_p,
        k=args.sample_batch,
        max_length=args.max_new_tokens,
        drop_nan=args.rejection_sample,
        do_impute=args.do_impute,
        prompt_template=args.prompt_template,
        device=args.device,
        max_retries=args.sampling_max_retries,
        max_allowed_time=args.sampling_max_allowed_time,
        save_folder=args.output_dir,
        save_name=args.synth_save_as,
    )


if __name__ == "__main__":
    # test_get_start_token()
    # test_sample()
    main()
