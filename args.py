import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_dir', default='Bone_Fracture_Binary_Classification')
    parser.add_argument('-output_dir', default='output')
    parser.add_argument('-backbone', default='resnet18', choices=['resnet18', 'resnet34', 'resnet50'])
    parser.add_argument('-epochs', type=int, default=20)
    parser.add_argument('-bs', type=int, default=16)
    parser.add_argument('-lr', type=float, default=0.0001)
    parser.add_argument('-wd', type=float, default=0.00001)
    parser.add_argument('--horizontal-flip', action='store_true',
                        help='Enable horizontal flipping only if appropriate for your X-rays.')
    return parser.parse_args()
