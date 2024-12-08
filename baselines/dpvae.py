import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import argparse
import random
import pandas as pd
from opacus import PrivacyEngine
from opacus.utils.uniform_sampler import UniformWithReplacementSampler
import torch.nn as nn
from snsynth.transform.table import TableTransformer

import time

from utils.utils import dumpy_config_to_json
from baselines.baseline_utils.transformer import DataTransformer

from utils.RDP_accountant import get_noise_multiplier
from torch.nn.functional import cross_entropy

from utils.dataset import postprocess_data, get_metadata


def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        time_elapsed = end_time - start_time
        with open(f"{OUTPUT_DIR}/elapsed_time.txt", "a") as f:
            f.write(
                f"Time elapsed: {time_elapsed} secs / {time_elapsed/60} mins / {time_elapsed/3600} hrs\n"
            )
        return result

    return wrapper


class VAE(nn.Module):
    def __init__(self, x_dim, z_dim, factor=1):
        super(VAE, self).__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.factor = factor

        # Encoder part
        self.fc_feat_x = nn.Sequential(
            nn.Linear(x_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(128, z_dim)
        self.fc_logvar = nn.Linear(128, z_dim)

        self.dec = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, x_dim),
        )

        self.rec_crit = nn.MSELoss()

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return mu, logvar, self.decode(z)

    def encode(self, x):
        feat = self.fc_feat_x(x)
        return self.fc_mu(feat), self.fc_logvar(feat)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.dec(z)

    def compute_loss(self, x, output_info):
        mu, logvar, recon_x = self.forward(x)
        st = 0
        loss = []
        for column_info in output_info:
            for span_info in column_info:
                if span_info.activation_fn != "softmax":
                    ed = st + span_info.dim
                    eq = x[:, st] - torch.tanh(recon_x[:, st])
                    loss.append((eq**2).sum())
                    st = ed
                else:
                    ed = st + span_info.dim
                    loss.append(
                        cross_entropy(
                            recon_x[:, st:ed],
                            torch.argmax(x[:, st:ed], dim=-1),
                            reduction="sum",
                        )
                    )
                    st = ed

        assert st == recon_x.size()[1]
        KLD = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp())
        return sum(loss) * self.factor / x.size()[0], KLD / x.size()[0]

    def sample(self, samples, batch_size, transformer, device, seed=1000):
        self.eval()
        set_seed(seed)

        steps = samples // batch_size + 1
        data = []
        for _ in range(steps):
            mean = torch.zeros(batch_size, self.z_dim)
            std = mean + 1
            noise = torch.normal(mean=mean, std=std).to(device)
            fake = self.decode(noise)
            fake = torch.tanh(fake)
            data.append(fake.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:samples]
        return transformer.inverse_transform(data)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="adult",
        help="The name of the dataset to use.",
        choices=["adult", "airline"],
    )
    parser.add_argument(
        "--device_type", "-d", type=str, default="cuda", help="type of device"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="./data/adult/k1000/train.csv",
        help="The path to the csv dataset file",
    )
    parser.add_argument("--l2scale", type=float, default=1e-5, help="learning rate")
    parser.add_argument(
        "--factor", type=float, default=2, help="weight for reconstruction loss"
    )
    parser.add_argument("--z_dim", "-z_dim", type=int, default=128, help="z_dim")
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--output_dir", default="./runs/", type=str)
    parser.add_argument("--epochs", default=2, type=int)
    parser.add_argument("--do_train", default=False, action="store_true")
    parser.add_argument("--do_generate", default=False, action="store_true")
    parser.add_argument("--n_synth_set", default=1, type=int)
    parser.add_argument("--n_synth_samples", default=10, type=int)
    parser.add_argument("--seed", default=1000, type=int)
    parser.add_argument(
        "--numerical_preprocess",
        default="standard",
        help="Preprocessing for numerical column",
    )
    parser.add_argument(
        "--enable_privacy",
        default=False,
        action="store_true",
        help="Enable private data generation",
    )
    parser.add_argument(
        "--target_epsilon", type=float, default=1, help="Epsilon DP parameter"
    )
    parser.add_argument(
        "--target_delta", type=float, default=1e-5, help="Delta DP parameter"
    )
    parser.add_argument(
        "--sigma", type=float, default=None, help="Gaussian noise variance multiplier."
    )
    parser.add_argument(
        "--max_norm",
        type=float,
        default=1,
        help="The coefficient to clip the gradients before adding noise for private SGD training",
    )
    return parser.parse_args()


@timeit
def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device_type if torch.cuda.is_available() else "cpu")

    global OUTPUT_DIR
    OUTPUT_DIR = args.output_dir

    if args.enable_privacy:
        synthesizer = "dpvae"
        save_path = f"{args.output_dir}/{args.dataset_name}/{synthesizer}/bs{args.batch_size}-eps{args.target_epsilon}-clip{args.max_norm}"
    else:
        synthesizer = "vae"
        save_path = (
            f"{args.output_dir}/{args.dataset_name}/{synthesizer}/bs{args.batch_size}"
        )

    OUTPUT_DIR = save_path

    os.makedirs(save_path, exist_ok=True)
    dumpy_config_to_json(f"{save_path}/train_config.json", vars(args))

    train_data_orig = pd.read_csv(args.dataset_path)
    nan_column = []
    for i in train_data_orig:
        if train_data_orig[i].isna().any():
            nan_column.append(i)
            train_data_orig[i] = train_data_orig[i].fillna(-1)  # fill na

    discrete_columns = train_data_orig.select_dtypes(include=["object"]).columns

    if not args.enable_privacy:
        transformer = DataTransformer(numerical_preprocess=args.numerical_preprocess)
        transformer.fit(train_data_orig, discrete_columns)
        train_data = transformer.transform(train_data_orig)
        data_dim = transformer.output_dimensions

    else:
        columns = train_data_orig.columns.to_list()
        transformer = TableTransformer.create(
            train_data_orig,
            style="gan",
            categorical_columns=discrete_columns,
            continuous_columns=list(set(columns) - set(discrete_columns)),
            # ordinal_columns=list(set(columns) - set(discrete_columns)),
            nullable=False,
            constraints=None,
        )
        transformer.fit(train_data_orig, epsilon=0.1)
        eps_spent, _ = transformer.odometer.spent
        args.target_epsilon -= eps_spent
        print(
            f"Spent {eps_spent} epsilon on preprocessor, leaving {args.target_epsilon} for training"
        )

        train_data = transformer.transform(train_data_orig)
        train_data = np.array(
            [[float(x) if x is not None else 0.0 for x in row] for row in train_data]
        )
        data_dim = transformer.output_width

    if args.enable_privacy:
        transformer_doppler = DataTransformer(numerical_preprocess="none")
        transformer_doppler.fit(train_data_orig, discrete_columns)
    else:
        transformer_doppler = transformer

    dset = TensorDataset(torch.from_numpy(train_data.astype("float32")).to(device))

    if args.enable_privacy:
        sample_rate = args.batch_size / len(dset)
        generator = None
        kwargs = {"num_workers": 0, "pin_memory": True}
        loader = DataLoader(
            dset,
            generator=generator,
            batch_sampler=UniformWithReplacementSampler(
                num_samples=len(dset),
                sample_rate=sample_rate,
                generator=generator,
            ),
        )
    else:
        loader = DataLoader(dset, batch_size=args.batch_size, shuffle=True)

    model = VAE(x_dim=data_dim, z_dim=args.z_dim, factor=args.factor).to(device)

    opt = optim.Adam(model.parameters(), lr=args.l2scale)

    if args.enable_privacy:
        sample_rate = args.batch_size / len(dset)
        num_iter = int(np.ceil((len(dset) / args.batch_size))) * args.epochs
        noise_multiplier = get_noise_multiplier(
            target_epsilon=args.target_epsilon,
            target_delta=args.target_delta,
            sample_rate=sample_rate,
            steps=num_iter,
        )
        sigma = noise_multiplier if args.sigma is None else args.sigma
        print(
            f"epsilon: {args.target_epsilon}, delta: {args.target_delta}, sigma: {sigma}, total_iter:{num_iter}, sampling_prob:{sample_rate}"
        )
        with open(f"{save_path}/privacy_info.txt", "w") as f:
            f.write(
                f"Epsilon: {args.target_epsilon}\nNoise: {sigma}\nDelta: {args.target_delta}\nTotal iters: {num_iter}\nSampling Probability: {sample_rate}"
            )

        privacy_engine = PrivacyEngine(accountant="rdp")
        model, opt, loader = privacy_engine.make_private(
            module=model,
            optimizer=opt,
            data_loader=loader,
            max_grad_norm=args.max_norm,
            noise_multiplier=sigma,
        )

    if args.do_train:
        for epoch in range(1, args.epochs + 1):
            for data_x in loader:
                data_x = data_x[0].to(device)
                model.train()
                opt.zero_grad()

                if args.enable_privacy:
                    loss_1, loss_2 = model._module.compute_loss(
                        data_x, output_info=transformer_doppler.output_info_list
                    )
                else:
                    loss_1, loss_2 = model.compute_loss(
                        data_x, output_info=transformer.output_info_list
                    )
                loss = loss_1 + loss_2

                loss.backward()
                opt.step()

            if epoch == 1:
                with open(f"{save_path}/train_loss.txt", "w") as f:
                    f.write("epoch,loss,recons_loss,kl_loss\n")
                    f.write(f"{epoch},{loss},{loss_1},{loss_2}\n")
            else:
                with open(f"{save_path}/train_loss.txt", "a") as f:
                    f.write(f"{epoch},{loss},{loss_1},{loss_2}\n")

            if args.enable_privacy:
                epsilon, best_alpha = privacy_engine.accountant.get_privacy_spent(
                    delta=args.target_delta
                )
                print(
                    f"Privacy spent: (ε = {epsilon:.2f}, δ = {args.target_delta}) with best alpha: {best_alpha}"
                )

            if epoch % 10 == 0:
                print(epoch, loss.item())

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": (
                        model._module.state_dict()
                        if args.enable_privacy
                        else model.state_dict()
                    ),
                    "opt_state_dict": opt.state_dict(),
                },
                os.path.join(save_path, "model.pkl"),
            )

    if args.do_generate:
        if not args.do_train:
            model_dir = os.path.join(save_path, "model.pkl")
            checkpth = torch.load(model_dir)

            model.load_state_dict(checkpth["model_state_dict"])
            opt.load_state_dict(checkpth["opt_state_dict"])
        elif args.enable_privacy:
            model = model._module

        metadata = get_metadata(train_data_orig)
        synth_folder = f"{save_path}/synth_data"
        synth_folder = f"{save_path}/synth_data"
        raw_folder = f"{synth_folder}/raw_tables"
        processed_folder = f"{synth_folder}/processed_tables"
        os.makedirs(synth_folder, exist_ok=True)
        os.makedirs(raw_folder, exist_ok=True)
        os.makedirs(processed_folder, exist_ok=True)

        args.transformer = transformer
        args.device = device
        for i in range(args.n_synth_set):
            args.i = i
            processed_tables, raw_tables = sample(
                args, model, metadata=metadata, nan_column=nan_column
            )
            print(i)
            raw_tables.to_csv(f"{raw_folder}/synth_{i}.csv", index=False)
            processed_tables.to_csv(f"{processed_folder}/synth_{i}.csv", index=False)


def sample(args, model, metadata=None, nan_column=[]):
    raw_tables = []
    processed_tables = []

    remaining = 0

    while remaining < args.n_synth_samples:
        raw_table = model.sample(
            args.n_synth_samples,
            args.batch_size,
            args.transformer,
            args.device,
            seed=args.seed + args.i,
        )
        raw_table = pd.DataFrame(raw_table)
        raw_tables.append(raw_table)

        if len(nan_column) != 0:
            for i in nan_column:
                raw_table[i] = raw_table[i].apply(lambda x: None if x == -1 else x)

        processed_table = postprocess_data(raw_table, metadata=metadata, dropna=True)

        processed_tables.append(processed_table)
        remaining += len(processed_table)

    raw_tables = pd.concat(raw_tables).reset_index(drop=True)
    processed_tables = pd.concat(processed_tables).reset_index(drop=True)
    processed_tables = processed_tables.head(args.n_synth_samples)

    return processed_tables, raw_tables


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


if __name__ == "__main__":
    main()
