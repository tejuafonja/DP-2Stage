import torch.nn as nn
import torch.nn.functional as F


class AugmentedBlock(nn.Module):
    def __init__(self, augmented_type, block):
        super(AugmentedBlock, self).__init__()
        self.augmented_type = augmented_type
        self.block = block

        for p in self.block.parameters():
            p.requires_grad = False

        self.allowed_aug_type = ["append", "resb4"]

        assert (
            self.augmented_type in self.allowed_aug_type
        ), f"Unkwnown augmented block type: {self.augmented_type}."

        self.append_fc, self.res_fc, self.zero_fc1, self.zero_fc2 = (
            None,
            None,
            None,
            None,
        )
        if "append" in self.augmented_type:
            in_features = self.block.in_features
            self.append_fc = nn.Linear(in_features, in_features)  # size?
            self.layer_norm = nn.LayerNorm(in_features, eps=1e-5)

        if "resb4" in self.augmented_type:
            in_features = self.block.in_features
            self.res_fc = nn.Linear(in_features, in_features)  # size?
            self.zero_out1 = nn.Linear(in_features, in_features)
            self.zero_out1.apply(self._zero_init)
            self.layer_norm = nn.LayerNorm(in_features, eps=1e-5)
            self.zero_out2 = nn.Linear(in_features, in_features)
            self.zero_out2.apply(self._zero_init)

    def _zero_init(self, m):
        if isinstance(m, nn.Linear):
            m.weight.data.fill_(0)
            m.bias.data.fill_(0)

    def forward(self, x):
        if self.augmented_type == "append":
            out = self.append_fc(x)
            out = F.relu(out)
            out = self.layer_norm(out)

        elif self.augmented_type == "resb4":
            out = self.zero_out1(x)
            out = self.res_fc(out)
            out = F.relu(out)
            out = self.layer_norm(out)
            out = self.zero_out2(out)
            out = self.block(x + out)
        else:
            raise NotImplementedError()

        return out
