import numpy as np

# Load the split file
dataset_split = '/wild6d_data/zubair/MAE_complete_data/front3d_split.npz'
with np.load(dataset_split) as split:
    train_scenes = split["train_scenes"]
    test_scenes = split["test_scenes"]
    val_scenes = split["val_scenes"]

# Remove the specific scene from train_scenes
scene_to_remove = "00326-u9rPN5cHWBg_9"
train_scenes = [scene for scene in train_scenes if scene != scene_to_remove]

# Save the modified train_scenes into a new file
new_train_scenes_file = '/wild6d_data/zubair/MAE_complete_data/new_train_scenes.npz'
np.savez(new_train_scenes_file, train_scenes=train_scenes, test_scenes=test_scenes, val_scenes=val_scenes)

print("New train scenes file saved successfully.")