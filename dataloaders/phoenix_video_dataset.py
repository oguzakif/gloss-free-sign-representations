from torch.utils.data import Dataset
import lmdb
from PIL import Image
import io
import pickle
import numpy as np
import torch
from dataloaders.data_utils.my_concat_dataset import MyConcatDataset
from dataloaders.data_utils.lmdb_utils import LMDBUtility
from tqdm import tqdm
import pandas as pd
import ignite.distributed as idist


class PhoenixVideoDataset(Dataset):
    def __init__(
        self,
        df,
        lmdb_video_dir,
        dict_gloss_to_id,
        transform,
        isValid,
        dict_sentence=None,
        dict_lem_to_id=None,
        dict_lem_counter=None,
        pseudo_gloss_from_orth=False,
        pseudo_gloss_filter_max_ratio=None,
    ):
        self.items = df.to_dict("records")
        self.lmdb_dir = lmdb_video_dir
        self.transform = transform
        self.isValid = isValid
        self.lmdb_util_video = None
        self.dict_gloss_to_id = dict_gloss_to_id
        self.dict_sentence = dict_sentence
        self.dict_lem_to_id = dict_lem_to_id
        self.dict_lem_counter = dict_lem_counter
        self.pseudo_gloss_from_orth = pseudo_gloss_from_orth
        self.pseudo_gloss_filter_max_ratio = pseudo_gloss_filter_max_ratio

    def __len__(self):
        return len(self.items)

    def collate_fn(self, batch):
        key_set = {k for k in batch[0].keys()}
        val_list = lambda k: [d.get(k) for d in batch if d.get(k) is not None]
        return {k: val_list(k) for k in key_set}

    def _get_frame_selection(self):
        if self.num_frames <= 0:
            raise RuntimeError(
                f"LMDB segment has no frames: {self.lmdb_dir}/{self.items[0]['name']}"
            )

        stride = max(int(self.transform.stride), 1)

        if self.isValid or self.num_frames <= self.transform.random_shift:
            start_frame = 0
            end_frame = self.num_frames
        else:
            start_frame = np.random.randint(0, self.transform.random_shift)
            end_low = max(start_frame + 1, self.num_frames - self.transform.random_shift)
            end_frame = np.random.randint(end_low, self.num_frames + 1)

        selection = np.arange(start_frame, end_frame, stride).astype(int)
        if len(selection) == 0:
            selection = np.array([0], dtype=int)

        if len(selection) > self.transform.max_seq_len:
            selection = np.random.choice(
                selection, size=self.transform.max_seq_len, replace=False
            )
            selection = np.sort(selection)

        return selection

    def __getitem__(self, idx):
        item = self.items[idx]
        file_name = item["name"]

        if not self.lmdb_util_video:
            self.lmdb_util_video = LMDBUtility(
                f"{self.lmdb_dir}/{file_name}",
            )

            self.num_frames = self.lmdb_util_video.details["num_frames"]

        selection = self._get_frame_selection()

        frames = self.lmdb_util_video.get_frames(selection)
        if len(frames) == 0:
            raise RuntimeError(
                f"No frames loaded for {file_name} with selection {selection.tolist()} "
                f"from {self.lmdb_dir}/{file_name}"
            )

        if self.transform:
            frames = self.transform.aug_video(frames, self.isValid)
        else:
            frames = torch.tensor(np.stack(frames)).float()
        # Sanitize translation and orth fields (may be NaN/None/non-str)
        sentence = item.get("translation", "")
        if not isinstance(sentence, str):
            try:
                sentence = "" if (isinstance(sentence, float) and sentence != sentence) else str(sentence)
            except Exception:
                sentence = ""

        orth_val = item.get("orth", "")
        if not isinstance(orth_val, str):
            try:
                orth_val = "" if (isinstance(orth_val, float) and orth_val != orth_val) else str(orth_val)
            except Exception:
                orth_val = ""
        glosses = orth_val.split(" ") if len(orth_val) > 0 else []
        pseudo_gloss_ids = []
        if self.pseudo_gloss_from_orth and self.dict_lem_to_id is not None:
            pseudo_gloss_ids = [
                self.dict_lem_to_id[gloss]
                for gloss in glosses
                if gloss in self.dict_lem_to_id
                and (
                    self.pseudo_gloss_filter_max_ratio is None
                    or self.dict_lem_counter is None
                    or self.dict_lem_counter.get(gloss, 0) / max(len(self.items), 1)
                    < self.pseudo_gloss_filter_max_ratio
                )
            ]
        elif self.dict_sentence is not None:
            if sentence in self.dict_sentence:
                lems = self.dict_sentence[sentence]
                sentence_filter_max_ratio = (
                    0.4
                    if self.pseudo_gloss_filter_max_ratio is None
                    else self.pseudo_gloss_filter_max_ratio
                )
                pseudo_gloss_ids = [
                    self.dict_lem_to_id[lem]
                    for lem in lems
                    if lem in self.dict_lem_to_id
                    and (
                        sentence_filter_max_ratio is None
                        or self.dict_lem_counter[lem] / len(self.dict_sentence)
                        < sentence_filter_max_ratio
                    )
                ]
                # pseudo_gloss_ids.append(self.dict_lem_to_id[lem])

        if self.dict_gloss_to_id:
            gloss_ids = [self.dict_gloss_to_id[gloss] for gloss in glosses]

        return {
            "index": torch.tensor(idx).long(),
            "frames": frames,
            "sentence": sentence,
            "file_name": file_name,
            **(
                {"gloss_ids": torch.tensor(gloss_ids).long()}
                if self.dict_gloss_to_id
                else {}
            ),
            **(
                {"pseudo_gloss_ids": torch.tensor(pseudo_gloss_ids).long()}
                if self.dict_lem_to_id
                else {}
            ),
        }


def get_ds(ds_params, transform):
    from dataloaders.data_utils.file_utils import read_json, read_pickle

    # Read CSV and sanitize to ensure string types
    df = pd.read_csv(ds_params["csv_dir"], sep=ds_params["sep"], keep_default_na=False)
    if "translation" not in df.columns:
        df["translation"] = ""
    if "orth" not in df.columns:
        df["orth"] = ""
    df["translation"] = df["translation"].astype(str)
    df["orth"] = df["orth"].astype(str)
    if "gloss_dir" in ds_params:
        dict_gloss_to_id = read_json(ds_params["gloss_dir"])
    else:
        dict_gloss_to_id = None

    if "pseudo_gloss_dir" in ds_params:
        dict_processed_words = read_pickle(ds_params["pseudo_gloss_dir"])

        dict_sentence = dict_processed_words["dict_sentence"]
        dict_lem_to_id = dict_processed_words["dict_lem_to_id"]
        dict_lem_counter = dict_processed_words["dict_lem_counter"]
    else:
        dict_sentence = None
        dict_lem_to_id = None
        dict_lem_counter = None
    list_of_ds = []
    length = 0
    for gp, d in tqdm(df.groupby("name")):
        ds = PhoenixVideoDataset(
            d,
            **ds_params["ds_params"],
            dict_gloss_to_id=dict_gloss_to_id,
            dict_sentence=dict_sentence,
            dict_lem_to_id=dict_lem_to_id,
            dict_lem_counter=dict_lem_counter,
            pseudo_gloss_from_orth=ds_params.get("pseudo_gloss_from_orth", False),
            pseudo_gloss_filter_max_ratio=ds_params.get(
                "pseudo_gloss_filter_max_ratio", None
            ),
            transform=transform,
        )
        list_of_ds.append(ds)
        collate_fn = ds.collate_fn
        length += len(ds)

    ds = MyConcatDataset(list_of_ds)
    dl = idist.auto_dataloader(
        ds,
        shuffle=ds_params["shuffle"],
        sampler=None,
        num_workers=ds_params["num_workers"],
        batch_size=ds_params["bs"],
        drop_last=ds_params["drop_last"],
        collate_fn=collate_fn,
        pin_memory=True,
    )
    return dl, {"length": length, "dict_lem_to_id": dict_lem_to_id}
