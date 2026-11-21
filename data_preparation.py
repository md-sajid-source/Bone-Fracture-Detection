
import os
import hashlib
from collections import defaultdict
import pandas as pd
from args import get_args

IMAGE_TYPES = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')


def find_label(folder_names):
    names = {n.lower().replace('_', '').replace('-', '').replace(' ', '')
             for n in folder_names}
    if names & {'notfractured', 'nonfractured', 'normal', 'nofracture', 'unfractured'}:
        return 0
    if names & {'fractured', 'fracture'}:
        return 1
    return None


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def make_csvs(data_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    choices = {
        'training': ('train', 'training'),
        'validation': ('val', 'valid', 'validation'),
        'test': ('test', 'testing'),
    }
    records = defaultdict(list)

    for split, names in choices.items():
        root = next((os.path.join(data_dir, n) for n in names
                     if os.path.isdir(os.path.join(data_dir, n))), None)
        if root is None:
            raise FileNotFoundError(f'Missing {split} folder in {data_dir}')

        for folder, subfolders, filenames in os.walk(root):
            subfolders[:] = sorted(n for n in subfolders
                                   if n.lower() not in ('unlabeled', 'unlabelled'))
            label = find_label(os.path.relpath(folder, root).split(os.sep))
            if label is None:
                continue
            for filename in sorted(filenames):
                if filename.lower().endswith(IMAGE_TYPES):
                    path = os.path.abspath(os.path.join(folder, filename))
                    records[file_hash(path)].append((split, path, label))

    conflicts = [group for group in records.values()
                 if len({entry[2] for entry in group}) > 1]
    if conflicts:
        raise ValueError(f'{len(conflicts)} duplicate groups have conflicting labels: '
                         f'{conflicts[0]}')

    priority = {'test': 0, 'validation': 1, 'training': 2}
    kept = {split: [] for split in choices}
    removed = {split: 0 for split in choices}
    report = []

    for image_hash, group in records.items():
        group.sort(key=lambda e: (priority[e[0]], e[1]))
        split, path, label = group[0]
        kept[split].append((path, label))

        for duplicate_split, duplicate_path, _ in group[1:]:
            removed[duplicate_split] += 1
            report.append((image_hash, split, path,
                           duplicate_split, duplicate_path, label))

    frames = {}
    for split, rows in kept.items():
        df = pd.DataFrame(rows, columns=['image_path', 'label'])
        if df.empty or set(df['label']) != {0, 1}:
            raise ValueError(f'{split} has insufficient class coverage.')
        frames[split] = df

    for split, df in frames.items():
        df.to_csv(os.path.join(output_dir, f'{split}.csv'), index=False)
        print(f'{split}: {len(df)} images; not fractured = {(df.label == 0).sum()}, '
              f'fractured = {(df.label == 1).sum()}; duplicates removed = {removed[split]}')

    pd.DataFrame(report, columns=[
        'image_hash', 'kept_split', 'kept_image',
        'removed_split', 'removed_image', 'label'
    ]).to_csv(os.path.join(output_dir, 'duplicate_report.csv'), index=False)

    print(f'Total duplicates excluded: {sum(removed.values())}')
    print('Original images unchanged. Patient overlap not verified.')


if __name__ == '__main__':
    args = get_args()
    make_csvs(args.data_dir, args.output_dir)
