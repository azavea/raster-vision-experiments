# noisy-buildings-semseg

The purpose of this experiment set is to measure how errors in training labels affect accuracy in the learned model. This uses the [Spacenet Vegas](https://spacenetchallenge.github.io/AOI_Lists/AOI_2_Vegas.html) building semantic segmentation dataset, based on the [Vegas](https://github.com/azavea/raster-vision-examples#spacenet-vegas) example.

For each noise type (ie. randomly shifting and deleting buildings) and noise level, we create a noisy label set based on the original ground truth labels. For each noisy label set, we train a model and validate on the original, uncorrupted labels.

## Instructions

* Running this requires https://github.com/azavea/raster-vision-fastai-plugin/tree/defb6d396d189b515efce5b1479c7a0161c6a869. To run this, place this repo directory within the `examples` directory, and follow instructions in the fastai plugin for running an example.
* Run this inside the container: `export PYTHONPATH=/opt/src/examples/raster-vision-experiments/:"$PYTHONPATH"`
* Download the data following instructions in the [Vegas](https://github.com/azavea/raster-vision-examples#spacenet-vegas) example.
* Set paths to data and RV output by modifying the constants at the start of [data.py](data.py).
* Generate the noisy labels by running `python -m noisy_buildings_semseg.prep`
* Sync noisy labels to cloud using aws cli.
* Run a small test local experiment using `rastervision -p fastai run local -e noisy_buildings_semseg.exp -a test True -a use_remote_data False`
* Run a full remote experiment using `rastervision -p fastai run aws_batch -e noisy_buildings_semseg.exp -a test False -a use_remote_data True --splits 4`
* When the individual experiments finish running, plot some curves based on the evaluations using `python -m noisy_buildings_semseg.plot_separate_curves` and `python -m noisy_buildings_semseg.plot_separate_curves`, and images of noisy labels and predictions using `python -m noisy_buildings_semseg.plot_images`.
