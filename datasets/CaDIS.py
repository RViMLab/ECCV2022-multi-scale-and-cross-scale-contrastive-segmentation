import pandas as pd
from utils import DATASETS_INFO, printlog
import pathlib


def get_cadis_dataframes(config: dict):
    # Make dataframes for the training and the validation set
    assert 'data' in config
    dataset = config['data']['dataset']
    assert dataset == 'CADIS', f'dataset must be CADIS instead got {dataset}'
    df = pd.read_csv('data/data.csv') # todo this should be moved to data dir

    if 'random_split' in config['data']:
        print("***Legacy mode: random split of all data used, instead of split of videos!***")
        train = df.sample(frac=config['data']['random_split'][0]).copy()
        valid = df.drop(train.index).copy()
        split_of_rest = config['data']['random_split'][1] / (1 - config['data']['random_split'][0])
        valid = valid.sample(frac=split_of_rest)
    else:
        splits = DATASETS_INFO[dataset].DATA_SPLITS[int(config['data']['split'])]
        if len(splits) == 3:
            printlog("using train-val-test split")
            train_videos, valid_videos, test_videos = splits
            if config['mode'] == 'infer':
                printlog(f"CADIS with mode {config['mode']}")
                printlog(f"going to use test_videos as vadilation set")
                valid_videos = test_videos
        elif len(splits) == 2:
            printlog("using train-merged[valtest] split")
            train_videos, valid_videos = splits
        else:
            raise ValueError('splits must be a list of length 2 or 3')
        train = df.loc[df['vid_num'].isin(train_videos)].copy()
        valid = df.loc[(df['vid_num'].isin(valid_videos)) & (df['propagated'] == 0)].copy()  # No prop lbl in valid
    info_string = "Dataframes created. Number of records training / validation: {:06d} / {:06d}\n" \
                  "                    Actual data split training / validation: {:.3f}  / {:.3f}" \
        .format(len(train.index), len(valid.index), len(train.index) / len(df), len(valid.index) / len(df))

    # Replace incorrectly annotated frames if flag set
    if config['data']['use_relabeled']:
        train_idx_list = train[train['relabeled'] == 1].index
        for idx in train_idx_list:
            train.loc[idx, 'blacklisted'] = 0  # So the frames don't get removed after
            lbl_path = pathlib.Path(train.loc[idx, 'lbl_path']).name
            train.loc[idx, 'lbl_path'] = 'relabeled/' + str(lbl_path)
        valid_idx_list = valid[valid['relabeled'] == 1].index
        for idx in valid_idx_list:
            valid.loc[idx, 'blacklisted'] = 0  # So the frames don't get removed after
            lbl_path = pathlib.Path(valid.loc[idx, 'lbl_path']).name
            valid.loc[idx, 'lbl_path'] = 'relabeled/' + str(lbl_path)
        info_string += "\n                                       Relabeled train recs: {}\n" \
                       "                                       Relabeled valid recs: {}" \
            .format(len(train_idx_list), len(valid_idx_list))

    # Remove incorrectly annotated frames if flag set
    if config['data']['blacklist']:
        train = train.drop(train[train['blacklisted'] == 1].index)
        valid = valid.drop(valid[valid['blacklisted'] == 1].index)
        t_len, v_len = len(train.index), len(valid.index)
        info_string += "\n        After blacklisting: Number of records train / valid: {:06d} / {:06d}\n" \
                       "                          Relative data split train / valid: {:.3f}  / {:.3f}" \
            .format(t_len, v_len, t_len / (t_len + v_len), v_len / (t_len + v_len))
    train = train.reset_index()
    valid = valid.reset_index()

    printlog(f" dataset {dataset}")
    printlog(info_string)
    return train, valid
