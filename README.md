# Raster Vision Experiments 🔬 🔭

This repo is for experiments using [Raster Vision](https://github.com/azavea/raster-vision) to measure the performance of algorithms under a variety of different circumstances. The repo is public for the sake of repeatability and transparency, but doesn't aim to be very well-maintained, supported, or documented. In contrast, the [raster-vision-examples](https://github.com/azavea/raster-vision-examples) repo contains examples of using Raster Vision on open datasets, and is a designed to be a good starting point for beginners.

To run these experiments, we recommend checking out [raster-vision-examples](https://github.com/azavea/raster-vision-examples) and then placing this repo inside the `other` directory. That way, the scripts and Docker setup in `raster-vision-examples` can be used to run the experiments in this repo.

## Experiments

* [noisy-buildings-semseg](noisy_buildings_semseg/): How does error in training labels affect accuracy for semantic segmentation?
