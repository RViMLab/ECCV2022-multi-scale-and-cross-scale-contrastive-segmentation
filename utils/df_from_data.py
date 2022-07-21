import pathlib
import pandas as pd

import argparse

# Set path to data, e.g. python df_from_data.py --path <path/to/segmentation>
parser = argparse.ArgumentParser()
path = "C:\\Users\\Theodoros Pissas\\Documents\\tresorit\\CaDIS\\segmentation"
parser.add_argument('-p', '--path', type=str, default=path,
                    help='Set path to data, e.g. python df_from_data.py --path <path/to/segmentation>')
args = parser.parse_args()

record_list = []
data_path = pathlib.Path(args.path)
subfolders = [[f, f.name] for f in data_path.iterdir() if f.is_dir()]
for folder_path, folder_name in subfolders:
    for image in (folder_path / 'Images').iterdir():
        record_list.append([
            int(folder_name[-2:]),                       # Video number: 'Video01' --> 1
            str(pathlib.PurePosixPath(pathlib.Path(folder_name) / 'Images' / image.name)),  # Relative path to the image
            str(pathlib.PurePosixPath(pathlib.Path(folder_name) / 'Labels' / image.name)),  # Relative path ot the label
        ])
df = pd.DataFrame(data=record_list, columns=['vid_num', 'img_path', 'lbl_path'])
df = df.sort_values(by=['vid_num', 'img_path']).reset_index(drop=True)
df.to_pickle('../data/data.pkl')
