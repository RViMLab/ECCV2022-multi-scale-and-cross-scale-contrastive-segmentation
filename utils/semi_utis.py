import pathlib
from collections import OrderedDict
from torch.utils.data import Dataset


class BalancedConcatDataset(Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets
        dataset_lengths = [len(d) for d in self.datasets]
        self.max_len = max(dataset_lengths)
        self.min_len = min(dataset_lengths)

    def __getitem__(self, i):
        # each item is a a tuple of 1 unlabelled sample and 1 labelled sample
        v = [d[i % len(d)] for d in self.datasets]
        b = tuple(v)
        # b : b[0] = list containing dataset_1 img of shape (C,H,W), mask of shape (H,W), pseudo info of shape (,)
        # b : b[1] = list containing dataset_2 img of shape (C,H,W), mask of shape (H,W), pseudo info of shape (,)
        return b

    def __len__(self):
        # stop when the longest dataset runs out of samples
        return self.max_len


def get_video_files_from_split(ids, debug=False):
    """ gets list of video ids (i.e a split's train videos) and returns a list
     the names of the corresponding mp4 files"""
    dicts = dict()
    dicts['train_1'] = [1, 2, 3, 4, 5, 6, 7, 8] if not debug else [1, 3, 6]
    dicts['train_2'] = [9, 10, 11, 12, 13, 14, 15, 16]
    dicts['train_3'] = [17, 18, 19, 20, 21, 22, 23, 24]
    dicts['train_4'] = [25]
    files = []
    for i in ids:
        # s = "{0:0=1d}".format(i)
        s = "%02d" % i
        if i in dicts['train_1']:
            files.append(pathlib.Path('train_1') / pathlib.Path('train{}.mp4'.format(s)))
        elif i in dicts['train_2'] and not debug:
            files.append(pathlib.Path('train_2') / pathlib.Path('train{}.mp4'.format(s)))
        elif i in dicts['train_3'] and not debug:
            files.append(pathlib.Path('train_3') / pathlib.Path('train{}.mp4'.format(s)))
        elif i in dicts['train_4'] and not debug:
            files.append(pathlib.Path('train_4') / pathlib.Path('train{}.mp4'.format(s)))
    return files


def get_excluded_frames_from_df(df, train_videos):
    train = df.loc[df['vid_num'].isin(train_videos)]
    train.reset_index()
    train = train.reset_index()
    train = train.drop(train[train['blacklisted'] == 1].index)
    train = train.reset_index()
    img_vid_frames = train['img_path']
    img_vid_frames = img_vid_frames.tolist()
    video_to_excluded_frames_dict = OrderedDict()
    for f in img_vid_frames:
        frame_id = int(f.split('.')[-2][-6:])
        video_id = f.split('Video')[-1][0:2] if '_' not in f.split('Video')[-1][0:2] else f.split('Video')[-1][0]
        video_id = int(video_id)
        if video_id in video_to_excluded_frames_dict:
            video_to_excluded_frames_dict[video_id].append(frame_id)
        else:
            video_to_excluded_frames_dict[video_id] = []
            video_to_excluded_frames_dict[video_id].append(frame_id)
    # sanity check
    assert(list(video_to_excluded_frames_dict.keys()) == train_videos)
    return video_to_excluded_frames_dict

