# Dataset and Training Guide

## Dataset management

- Use DatasetRegistry to register datasets.
- Use DatasetVersion to version datasets with metadata and checksums.
- Use DatasetValidator to validate missing files, annotations, timestamps, and duplicates.

## Labeling

- LabelManager supports multi-label annotation for species, behaviour, distress, pain, fear, aggression, occupancy, pose, health, and environment.

## Augmentation

- VisionAugmenter supports crop, flip, brightness, blur, noise, rotation, and occlusion.
- AudioAugmenter supports pitch shift, time stretch, background noise, gain, and SpecAugment.
- SensorAugmenter supports Gaussian noise, interpolation, temporal jitter, and missing-value simulation.

## Training pipelines

- VisionTrainingPipeline, AudioTrainingPipeline, SensorTrainingPipeline, and FusionTrainingPipeline provide modular training entry points.
- TrainingConfig captures hyperparameters, random seed, dataset version, and split information.
- ModelRegistry tracks model versions and metadata.
- ExperimentTracker stores metrics, configurations, and experiment history.
