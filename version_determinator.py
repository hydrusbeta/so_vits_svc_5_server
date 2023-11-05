import sys

import torch


def main(checkpoint_path):
    # enc_p.pre.weight has shape:
    # 	torch.Size([192, 1024, 5]) <== v1 models
    # 	torch.Size([192, 1280, 5]) <== v2 models
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
    version = '1' if checkpoint_dict['model_g']['enc_p.pre.weight'].shape[1] == 1024 else '2'
    print(version, end='')  # Don't print a newline at the end, to make output easier to parse.


if __name__ == '__main__':
    main(sys.argv[1])
