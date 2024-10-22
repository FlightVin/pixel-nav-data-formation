# Forming training trajectories for PixelNav based controller

Changes made in `.../habitat-lab/habitat-lab/habitat/core/env.py` to get the semantic sensor working:

Comment out the lines indicated here (Starting from 105)

```python
        # load the first scene if dataset is present
        if self._dataset:
            assert (
                len(self._dataset.episodes) > 0
            ), "dataset should have non-empty episodes list"
            self._setup_episode_iterator()
            self.current_episode = next(self.episode_iterator)
            with read_write(self._config):
                # self._config.simulator.scene_dataset = (
                #     self.current_episode.scene_dataset_config
                # )
                self._config.simulator.scene = self.current_episode.scene_id
```

Reference - https://github.com/facebookresearch/habitat-lab/issues/2090