import numpy as np
dataset_split = '/wild6d_data/zubair/MAE_complete_data/front3d_split.npz'
with np.load(dataset_split) as split:
    train_scenes = split["train_scenes"]

scenes_remove = '00326-u9rPN5cHWBg_9.npz'
scenes_remove = scenes_remove.split('.')[0]

train_scenes = [scene for scene in train_scenes if scene != scenes_remove]

#save the new split
np.savez('/wild6d_data/zubair/MAE_complete_data/front3d_split.npz', train_scenes=train_scenes, test_scenes=split["test_scenes"], val_scenes=split["val_scenes"])