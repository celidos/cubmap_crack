import json


def read_args_from_config(filename: str, fold: int, device: str):
    with open(filename) as f:
        args = json.load(f)
    args['device'] = device
    args['swa_device'] = device
    args['current_fold'] = fold
    
    return args


def dump_args(filename: str, args):
    with open(filename, "w") as outfile:
        json.dump(args, outfile, indent=4)
