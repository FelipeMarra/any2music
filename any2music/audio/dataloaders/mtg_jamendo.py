import os
import csv
import typing as tp

from torch.utils.data import Dataset

CATEGORIES = ['genre', 'instrument', 'mood/theme']
TAG_HYPHEN = '---'

class MTGJamendoDataset(Dataset):
    """
    Args:
        dataset_root (str): Path to your download of the MTG-Jamendo  Dataset (downloaded with the `--dataset raw_30s` tag)
        mtg_clone (str): Path to your clone of https://github.com/MTG/mtg-jamendo-dataset 
        split_type (str): Choose between 'train', 'test' or 'validation'
        split_num (int): Choose between one of the 5 splits: 0, 1, 2, 3 or 4. Defaults to 0
        desc_format (str): Choose between 'tags' or 'phrase'. Defaults to 'tags'
    """
    def __init__(self, dataset_root:str, mtg_clone:str, split_type:str, split_num:int=0, desc_format='tags') -> None:
        super().__init__()

        self.dataset_root = dataset_root
        self.mtg_clone = mtg_clone
        self.tsv_file = os.path.join(self.mtg_clone, 'data', 'splits', f'split-{split_num}', f'autotagging-{split_type}.tsv')
        self.desc_format = desc_format
        self.audios_list = self.get_audios_and_meta()


    def get_audios_and_meta(self) -> list[dict[str, tp.Any]]:
        """
        Returns:
            list[dict[str, any]]: A list with dicts in the format 
        >>> {
                'path': 'path/to/audio', 
                'duration': float,
                'desc': 'music description'
            }
        """
        tracks = []

        with open(self.tsv_file) as fp:
            reader = csv.reader(fp, delimiter='\t')
            next(reader, None)  # skip header
            for row in reader:
                # Get dict in the format category: tags - e.g., genre: eletronic, pop
                raw_tags = row[5:]
                tags_dict = {category: set() for category in CATEGORIES}

                for tag_str in raw_tags:
                    category, tag = tag_str.split(TAG_HYPHEN)
                    tags_dict[category].update(set(tag.split(",")))

                # Transform tags_dict in a music description
                if self.desc_format == 'phrase':
                    desc = 'A song with the following '
                else:
                    desc = ''

                for idx, category in enumerate(tags_dict):
                    if len(tags_dict[category]) > 0:
                        tags_str = ', '.join(tags_dict[category])

                        if self.desc_format == 'phrase':
                            if idx > 0:
                                desc += ' and '

                            desc += f'{category+"s"}: {tags_str}'
                        else:
                            desc += tags_str

                track_dict = {
                    'path': os.path.join(self.dataset_root, row[3]),
                    'duration': float(row[4]),
                    'desc': desc
                }

                tracks.append(track_dict)

        return tracks


    def __len__(self):
        return len(self.audios_list)


    def __getitem__(self, index) -> dict[str, tp.Any]:
        return self.audios_list[index]