import os
import random
import glob
import shutil
import json
import filecmp

import numpy as np
import torch

from src.data_src.experiment_src.experiment_create import create_experiment
from src.models.model_utils.cnn_big_images_utils import create_tensor_image, create_CNN_inputs_loop
from src.utils import maybe_makedir
from src.data_grouping import (
    CACHE_MANIFEST_FILENAME,
    SCENE_BATCH_FORMAT_VERSION,
    batch_cache_manifest,
    batch_cache_path,
    requires_scene_window_batches,
)
from src.batch_cache_io import (
    ZSTD_PICKLE_SUFFIX,
    batch_cache_files,
    dump_batch_cache,
)


def is_legitimate_traj(traj_df, step):
    """
    check if the candidate trajectory satisfies the requirement
    """
    agent_id = traj_df.agent_id.values
    # check if I only have 1 agent (always the same)
    if not (agent_id[0] == agent_id).all():
        print("not same agent")
        return False
    # frame_ids = traj_df.frame_id.values
    # equi_spaced = np.arange(frame_ids[0], frame_ids[-1] + 1, step, dtype=int)
    # # check that frame rate is evenly-spaced
    # if not (frame_ids == equi_spaced).all():
    #     print("not equi_spaced")
    #     return False
    # if checks are passed
    return True


class Trajectory_Data_Pre_Process(object):
    def __init__(self, args):
        self.args = args

        # Trajectories and data_batches folder
        self.data_batches_path = batch_cache_path(self.args)
        self.cache_manifest_path = os.path.join(
            self.data_batches_path, CACHE_MANIFEST_FILENAME)
        expected_manifest = batch_cache_manifest(self.args)
        rebuild_reason = None
        if os.path.isdir(self.data_batches_path):
            if getattr(self.args, 'force_reprocess', False):
                rebuild_reason = '--force_reprocess True'
            else:
                try:
                    with open(self.cache_manifest_path, 'r') as handle:
                        cached_manifest = json.load(handle)
                except (FileNotFoundError, json.JSONDecodeError, OSError):
                    cached_manifest = None
                if cached_manifest != expected_manifest:
                    rebuild_reason = 'preprocessing settings/cache schema changed'
        if rebuild_reason is not None:
            print(f'Rebuilding generated batch cache: {rebuild_reason}.')
            shutil.rmtree(self.data_batches_path)
        maybe_makedir(self.data_batches_path)

        # Creating batches folders and files
        self.batches_folders = {}
        self.batches_confirmation_files = {}
        for set_name in ['train', 'valid', 'test']:
            # batches folders
            self.batches_folders[set_name] = os.path.join(
                self.data_batches_path, f"{set_name}_batches")
            maybe_makedir(self.batches_folders[set_name])
            # batches confirmation file paths
            self.batches_confirmation_files[set_name] = os.path.join(
                self.data_batches_path, f"finished_{set_name}_batches.txt")

        required_sets = ['train', 'valid', 'test']
        # A cache is reusable only when every split completed. Social caches
        # have a distinct versioned directory and can never be confused with
        # the legacy independently packed fragments.
        if all(os.path.exists(self.batches_confirmation_files[name])
               for name in required_sets):
            print('Data batches already created!\n')
            return

        print("Loading dataset and experiment ...")
        self.experiment = create_experiment(self.args.dataset)(
            self.args.test_set, self.args)
        # Every experiment already owns the matching dataset. Reusing it is
        # important for SDD, whose 47 full-resolution scene maps consume many
        # GiB when constructed twice.
        self.dataset = self.experiment.dataset
        print("Done.\n")

        print("Preparing data batches ...")
        self.num_batches = {}
        for set_name in required_sets:
            if not os.path.exists(self.batches_confirmation_files[set_name]):
                for stale_path in batch_cache_files(
                        self.batches_folders[set_name]):
                    os.remove(stale_path)
                self.num_batches[set_name] = 0
                print(f"\nPreparing {set_name} batches ...")
                if set_name == 'test' and self._link_identical_eval_batches():
                    print(
                        f"Reused {self.num_batches[set_name]} validation "
                        "batches for the byte-identical test split via hard "
                        "links.")
                else:
                    self.create_data_batches(set_name)

        temporary_manifest_path = self.cache_manifest_path + '.tmp'
        with open(temporary_manifest_path, 'w') as handle:
            json.dump(expected_manifest, handle, indent=2, sort_keys=True)
        os.replace(temporary_manifest_path, self.cache_manifest_path)

        print('Data batches created!\n')

    def _eval_source_files_are_identical(self):
        """Check whether validation and test use the same ordered raw files."""
        if self.args.shuffle_test_batches:
            return False
        valid_data = self.experiment.data['valid']
        test_data = self.experiment.data['test']
        if len(valid_data) != len(test_data):
            return False
        for valid_item, test_item in zip(valid_data, test_data):
            valid_path = valid_item['file_path']
            test_path = test_item['file_path']
            if (valid_item['scene_name'] != test_item['scene_name'] or
                    valid_item['downsample_frame_rate'] !=
                    test_item['downsample_frame_rate'] or
                    os.path.basename(valid_path) !=
                    os.path.basename(test_path) or
                    not filecmp.cmp(valid_path, test_path, shallow=False)):
                return False
        return True

    def _link_identical_eval_batches(self):
        """Hard-link validation batches when the test split is identical.

        ETH/UCY distributes identical validation and test files. The cached
        tensors are therefore identical too, and hard links avoid storing a
        second multi-GiB copy without changing any tensor values.
        """
        if not self._eval_source_files_are_identical():
            return False
        valid_paths = batch_cache_files(self.batches_folders['valid'])
        if not valid_paths:
            return False
        linked_paths = []
        try:
            for batch_number, valid_path in enumerate(valid_paths, start=1):
                suffix = (ZSTD_PICKLE_SUFFIX if valid_path.endswith(
                    ZSTD_PICKLE_SUFFIX) else '.pkl')
                test_path = os.path.join(
                    self.batches_folders['test'],
                    f"test_batch_{batch_number:04d}{suffix}")
                os.link(valid_path, test_path)
                linked_paths.append(test_path)
        except OSError:
            for linked_path in linked_paths:
                os.remove(linked_path)
            return False
        self.num_batches['test'] = len(linked_paths)
        with open(self.batches_confirmation_files['test'], 'w') as handle:
            handle.write(f"Number of test batches: {len(linked_paths)}")
        return True

    def create_data_batches(self, set_name):
        """
        Create data batches for the DataLoader object
        """
        for scene_data in self.experiment.data[set_name]:
            # break if fast_debug
            if self.args.fast_debug and self.num_batches[set_name] >= \
                    self.args.fast_debug_num:
                break
            self.make_batches(scene_data, set_name)
            print(f"Saved a total of {self.num_batches[set_name]} {set_name} "
                  f"batches ...")

        with open(self.batches_confirmation_files[set_name], "w") as f:
            f.write(f"Number of {set_name} batches: "
                    f"{self.num_batches[set_name]}")

    def make_batches(self, scene_data, set_name):
        """
        Query the trajectories fragments and make data batches.
        Notes: Divide the fragment if there are too many people; accumulate some
        fragments if there are few people.
        """
        if requires_scene_window_batches(self.args):
            return self.make_scene_window_batches(scene_data, set_name)

        scene_name = scene_data["scene_name"]
        scene = self.dataset.scenes[scene_name]
        delta_frame = scene.delta_frame
        downsample_frame_rate = scene_data["downsample_frame_rate"]

        df = scene_data['raw_pixel_data'] # from pd.read_csv()

        if set_name == 'train':
            shuffle = self.args.shuffle_train_batches
        elif set_name == 'test':
            shuffle = self.args.shuffle_test_batches
        else:
            shuffle = self.args.shuffle_test_batches
        assert scene_data["set_name"] == set_name

        fragment_list = []  # container for a batch of data (list of fragments)

        for agent_i in set(df.agent_id):
            hist = df[df.agent_id == agent_i] # get the hist of agent_i, a ndarray records the 
            # downsample frame rate happens here, at the single agent level
            hist = hist.iloc[::downsample_frame_rate] # [i:j:s] from i to j, step=s

            for start_t in range(0, len(hist), self.args.skip_ts_window):
                candidate_traj = hist.iloc[start_t:start_t + self.args.seq_length]
                if len(candidate_traj) == self.args.seq_length:
                    if is_legitimate_traj(candidate_traj,
                                          step=downsample_frame_rate * delta_frame):
                        fragment_list.append(candidate_traj)

        if shuffle:
            random.shuffle(fragment_list)

        batch_acculumator = []
        batch_ids = {
            "scene_name": scene_name,
            "starting_frames": [],
            "agent_ids": [],
            "data_file_path": scene_data["file_path"]}

        # Divide the fragment if there are too many people; accumulate some
        # fragments if there are few people.
        # fragment_df represents the traj of a single agent
        for fragment_df in fragment_list:
            # break if fast_debug
            if self.args.fast_debug and self.num_batches[set_name] >= \
                    self.args.fast_debug_num:
                break

            batch_ids["starting_frames"].append(fragment_df.frame_id.iloc[0])
            batch_ids["agent_ids"].append(fragment_df.agent_id.iloc[0])

            batch_acculumator.append(fragment_df[["x_coord", "y_coord"]].values)

            # save batch if big enough
            if len(batch_acculumator) == self.args.batch_size:
                # create and save batch
                self.massup_batch_and_save(batch_acculumator,
                                           batch_ids, set_name)

                # reset batch_acculumator and ids for new batch
                batch_acculumator = []
                batch_ids = {
                    "scene_name": scene_name,
                    "starting_frames": [],
                    "agent_ids": [],
                    "data_file_path": scene_data["file_path"]}

        # save last (incomplete) batch if there is some fragment left
        if batch_acculumator:
            # create and save batch
            self.massup_batch_and_save(batch_acculumator,
                                       batch_ids, set_name)

    def make_scene_window_batches(self, scene_data, set_name):
        """Save one exact synchronized frame window per pickle.

        The original GDTS cache mixes arbitrary per-agent fragments. Social
        interaction is meaningful only when columns share the same complete
        frame-id sequence, so non-baseline ablations group fragments by that
        exact tuple and never split a window. A window may contain a variable
        number of agents and may exceed the legacy ``batch_size``.
        """
        scene_name = scene_data["scene_name"]
        scene = self.dataset.scenes[scene_name]
        expected_step = (scene_data["downsample_frame_rate"] *
                         scene.delta_frame)
        df = scene_data['raw_pixel_data']
        # Use one global temporal phase for the whole scene/window. Sampling
        # every track from its own first row (the legacy behavior) gives
        # different phases when agents enter at different frames and can turn
        # a genuinely simultaneous crowd into many false N=1 batches.
        global_frame_origin = int(np.asarray(df.frame_id.values).min())

        windows = {}
        aligned_histories = []
        scene_frame_grid = set()
        for agent_i in sorted(set(df.agent_id), key=str):
            hist = df[df.agent_id == agent_i].sort_values('frame_id')
            frame_values = np.asarray(hist.frame_id.to_numpy())
            aligned_frame_mask = np.mod(
                frame_values - global_frame_origin, expected_step) == 0
            hist = hist[aligned_frame_mask]
            if len(hist) == 0:
                continue
            aligned_histories.append(hist)
            scene_frame_grid.update(int(frame) for frame in hist.frame_id.values)

        # Apply the window stride once on a scene-level temporal grid. If each
        # track starts its own range(0, ..., skip), late-entering pedestrians
        # can be assigned a different phase and disappear from shared windows.
        allowed_start_frames = set(
            sorted(scene_frame_grid)[::self.args.skip_ts_window])
        for hist in aligned_histories:
            for start_t in range(len(hist)):
                if int(hist.frame_id.iloc[start_t]) not in allowed_start_frames:
                    continue
                fragment = hist.iloc[start_t:start_t + self.args.seq_length]
                if len(fragment) != self.args.seq_length:
                    continue
                frame_ids = fragment.frame_id.to_numpy()
                if not np.all(np.diff(frame_ids) == expected_step):
                    continue
                frame_key = tuple(int(frame) for frame in frame_ids)
                windows.setdefault(frame_key, []).append(fragment)

        window_items = list(windows.items())
        if set_name == 'train' and self.args.shuffle_train_batches:
            random.shuffle(window_items)

        for frame_key, fragments in window_items:
            if self.args.fast_debug and self.num_batches[set_name] >= \
                    self.args.fast_debug_num:
                break
            batch_ids = {
                "scene_name": scene_name,
                "starting_frames": [frame_key[0]] * len(fragments),
                "frame_ids": list(frame_key),
                "agent_ids": [fragment.agent_id.iloc[0]
                              for fragment in fragments],
                "data_file_path": scene_data["file_path"],
                "synchronized_window": True,
                "batch_format_version": SCENE_BATCH_FORMAT_VERSION,
            }
            trajectories = [
                fragment[["x_coord", "y_coord"]].values
                for fragment in fragments
            ]
            self.massup_batch_and_save(
                trajectories, batch_ids, set_name,
                frame_ids=np.asarray(frame_key, dtype=np.int64))

    def massup_batch_and_save(self, batch_acculumator, batch_ids, set_name,
                              frame_ids=None):
        """
        Mass up data fragments to form a batch and then save it to disk.
        From list of dataframe fragments to saved batch.
        """
        abs_pixel_coord = np.stack(batch_acculumator).transpose(1, 0, 2) # batch_acculumator [agent_ids, x_coord, y_coord]
        seq_list = np.ones((abs_pixel_coord.shape[0],
                            abs_pixel_coord.shape[1]))

        data_dict = {
            "abs_pixel_coord": abs_pixel_coord,
            "seq_list": seq_list,
        }
        if frame_ids is not None:
            num_agents = abs_pixel_coord.shape[1]
            data_dict.update({
                # All columns belong to one simultaneous scene-window. These
                # explicit tensors survive the outer DataLoader collation.
                "scene_index": np.zeros(num_agents, dtype=np.int64),
                "scene_ptr": np.asarray([0, num_agents], dtype=np.int64),
                "frame_ids": np.repeat(frame_ids[:, None], num_agents, axis=1),
                "batch_format_version": np.asarray(
                    SCENE_BATCH_FORMAT_VERSION, dtype=np.int64),
            })

        # add cnn maps and inputs
        data_dict = self.add_pre_computed_cnn_maps(data_dict, batch_ids)

        # increase batch number count
        self.num_batches[set_name] += 1
        compressed = bool(getattr(
            getattr(self, 'args', None), 'compress_batch_cache', False))
        suffix = (ZSTD_PICKLE_SUFFIX if compressed
                  else '.pkl')
        batch_name = os.path.join(
            self.batches_folders[set_name],
            f"{set_name}_batch" + "_" + str(
                self.num_batches[set_name]).zfill(4) + suffix)
        # save batch
        dump_batch_cache(
            (data_dict, batch_ids), batch_name,
            compressed=compressed)

    def add_pre_computed_cnn_maps(self, data_dict, batch_ids):
        """
        Pre-compute CNN maps used by the goal modules and add them to data_dict
        """
        abs_pixel_coord = data_dict["abs_pixel_coord"]
        scene_name = batch_ids["scene_name"]
        scene = self.dataset.scenes[scene_name]

        # numpy semantic map from 0 to 1
        img = scene.semantic_map_pred
        map = scene.RGB_image

        tensor_map = create_tensor_image(
            big_numpy_image=map,
            down_factor=self.args.down_factor)
        
        tensor_image = create_tensor_image(
            big_numpy_image=img,
            down_factor=self.args.down_factor)   

        input_traj_maps = create_CNN_inputs_loop(
            batch_abs_pixel_coords=torch.tensor(abs_pixel_coord).float() /
                                   self.args.down_factor,
            tensor_image=tensor_image)

        data_dict["tensor_image"] = tensor_image
        data_dict["input_traj_maps"] = input_traj_maps
        data_dict["tensor_map"] = tensor_map

        return data_dict
