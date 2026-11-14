import io
import pickle

import lmdb
import numpy as np
from PIL import Image


class LMDBUtility(object):
    def __init__(self, lmdb_dir, landmark_keys=None):
        self.lmdb_dir = lmdb_dir

        env = self._open_environment()
        try:
            with env.begin(write=False) as txn:
                raw_details = txn.get(key=b"details")
                if raw_details is None:
                    raise KeyError(f"Missing LMDB details key in {self.lmdb_dir}")
                self.details = pickle.loads(raw_details)
        finally:
            # Do not leave thousands of short-lived LMDB mappings for the
            # process/OS to collect during a long epoch.
            env.close()
        self.landmark_keys = {} if landmark_keys is None else landmark_keys

    def _open_environment(self):
        return lmdb.open(
            path=f"{self.lmdb_dir}",
            readonly=True,
            readahead=False,
            lock=False,
            meminit=False,
        )

    def get_frames(self, selection):
        env = self._open_environment()
        try:
            frames = []
            with env.begin(write=False) as txn:
                for idx in selection:
                    frame_index = int(idx)
                    raw_frame = txn.get(key=str(frame_index).encode("ascii"))
                    if raw_frame is None:
                        raise KeyError(
                            f"Missing frame {frame_index} in LMDB {self.lmdb_dir}"
                        )

                    # Force decoding while the byte buffer and transaction are
                    # alive.  np.array also makes the result independent of the
                    # PIL object before it is closed.
                    with Image.open(io.BytesIO(raw_frame)) as image:
                        image.load()
                        frames.append(np.array(image))
            return frames
        finally:
            env.close()
