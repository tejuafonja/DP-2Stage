import argparse, os
import json, yaml
from collections import defaultdict

choice_dict = {}
default_dict = {}
allowed_overwritten_list = ["seed"]


class Parser(dict):
    def __init__(self, *args):
        super(Parser, self).__init__()
        for i_d, d in enumerate(args):
            if isinstance(d, argparse.Namespace):
                d = vars(d)
            elif d is None:
                print(f"Settings {i_d} is empty")
                continue

            for k, v in d.items():
                # ensure the input arguments have been in the dict;
                # otherwise report the error except for those allowed ones
                if (
                    k in allowed_overwritten_list
                    and k in self.keys()
                    and self[k] != None
                ):
                    print(f"{k} is found and overwritten in the arg parser.")
                    continue
                assert (
                    k not in self.keys()
                ), f"duplicated arguments {k}, please check the configuration file."

                k = k.replace("-", "_")
                # check whether arguments match the limited choices
                if k in choice_dict.keys() and v not in choice_dict[k]:
                    raise ValueError(
                        f"Illegal argument '{k}' for choices {choice_dict[k]}"
                    )
                # convert string None to Nonetype, which is a side effect of using yaml
                self[k] = None if v == "None" else v

        # check whether the default options has been in args; otherswise, add it.
        for k in default_dict.keys():
            if k not in self.keys():
                self[k] = default_dict[k]

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"{name}")

    def __setattr__(self, key, val):
        self[key] = val


def dumpy_config_to_json(out_path, *args):
    out_dict = {}
    for d in args:
        for k, v in d.items():
            assert not k in out_dict.keys(), f"Found an existing key {k}."
            out_dict[k] = v
            # print(k, type(v))

    _, ext = os.path.splitext(out_path)
    # assert not os.path.exists(out_path), f'File {out_path} has existed.'

    if ext == ".json":
        with open(out_path, "w") as outfile:
            json.dump(
                out_dict, outfile, sort_keys=True, indent=4, separators=(",", ": ")
            )
    elif ext == ".yaml":
        with open(out_path, "w") as outfile:
            yaml.dump(out_dict, outfile, default_flow_style=False)
    else:
        raise NotImplementedError(f"Unknown file extension {ext}")
