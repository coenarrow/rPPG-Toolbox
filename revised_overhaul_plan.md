# Project Overview

This project, `remote-physiology` is a repo built on the ideas of `rPPG-Toolbox`. Where `rPPG Toolbox` was built for a standardised, reproducible way of estimating heart rate, this packages is building on it so it extends to other forms of physiology as well. There are a range of changes made along the way, but primarily is the overall structure of the repo.

## Standardised Cache

Instead of having separate dataloaders for each dataset, what this repo does is explicitly require a standardised cached dataset structure using `zarr`. This allows multiple read/write for distributed, parallel training of models. The grouping of a zarr file is the assumption that this all covers the same period of recording, but could have multiple cameras, modalities etc. The zarr structure is as follows.

```
root/
 |-- attrs
 |-- <perspective_1>/                       
 |    |--attrs
 |    |-- <modality_1>/
 |    |    |--timestamps_us
 |    |    |    |--data     # (T,)
 |    |    |--video/
 |    |    |    | data    # (C, T, H, W)
 |    |    |--<trace_1>
 |    |    |    |--data     # (T,)
 |    |    |--<trace_2>
 |    |    |    |--data     # (T,)
 |    |    |...
 |    |    |--<trace_n>
 |    |    |    |--data     # (T,)
 |    |-- <modality_2>/
 |    |    |--timestamps_us
 |    |    |    |--data     # (T,)
 |    |    |--video/
 |    |    |    | data    # (C, T, H, W)
 |    |    |--<trace_1>
 |    |    |    |--data     # (T,)
 |    |    |...
 |-- <perspective_2>/                       
 |    |-- attrs
 |    |-- <modality_1>/
 |    |    |--timestamps_us
 |    |    |    |--data     # (T,)
 |    |    |--video/
 |    |    |    | data    # (C, T, H, W)
 |    |    |--<trace_1>
 |    |    |    |--data     # (T,)
 |    |    |...
 |...
 |-- <perspective_m>/
 |...
```

### `root`

Suppose we open up a `.zarr` file as `root`:

```python
root = zarr.open_group(<recording.zarr>)
attributes = dict(root.attrs)
```

#### Required Attributes

- `participant`
    - This is used by the dataloader so we can do splits between train/val/test or leave-one-out folds.

### `perspective`

The next level in the `.zarr` archive is the camera perspective (or individual camera). We've referred to it as perspective, because there may have been some pre-processing in the dataset to move multiple videos into a specific perspective, such as with the `Neckflix` dataset, where IR and Depth videos were reprojected into the RGB camera perspective. The assumption is that all frames in the perspective have physically aligned pixel representations. So each perspective could have multiple modalities.

#### Requirements

The perspective attribute has a key called `'fps'` and is the nominal fps for **all** modalities from this perspective. The first frame from all modalities must be aligned to less than 1/fps.

### `modality`

We have assumed a list of modalities based on the general literature.

- `'gr'` for Grayscale cameras
- `'rgb'` for RGB videos
- `'ir'` for Infrared
- `'depth'` for Depth
- `'t'` for Thermal
- `ev` for the event camera

Each modality would likely be coming from a different sensor (as it was with the Microsoft Kinect in the `Neckflix` dataset). As such, each sensor actually could have slightly different timestamps, so we require a `timestamps_us` group, with data.

#### Requirements

All modalities **must** have the same traces.

### `video`

To access the video frames for a specific modality, we can do that via.

```python
root[<perspective>][<modality>]["video"]["data"] # (C, T, H, W)
```

The video **must** be stacked in that order.

### `trace`

Much like the list of modalities, we've assumed a list of possible trace keys:

- `'ecg'` for ECG traces
- `'abp'` for Arterial Blood Pressure
- `'cvp'` for Central Venous Pressure
- `'ppg'` for Finger-based PPG
- `'rr'` for Respiratory traces on chest

#### Required attributes

The `<trace>` group must contain a `units` key in the dict, describing the units of the trace.The trace contains the units the trace was measured in. The most important ones for `abp` and `cvp` (mmHg typically, but could be cmH2O), with the rest likely being arbitrary. 'arb' is acceptable as a unit for things like PPG.

## Dataset + Dataloader

There's only a single type of dataset and dataloader since we've standardised the cache. The dataset works as the intermediate between the model and the cache, and so it is required to do all the pre-processing and/or augmentation for the model.

From the perspective attribute requirements about fps, although all cameras must have the same fps, they may not necessarily have the same duration. For the `Neckflix` dataset for instance, there may be some times where the ir camera stopped working prior to the rgb camera. In these instances, we drop frames from the longer video array.

If a model asks for a specific set of channels and/or traces, and the dataset doesn't actually have those. What the dataset should do, is create zero-masks for those traces/frames. The idea here is that if we have a model that was originally trained on 5 channels of data, it should still be able to work with only 3 channels.

The dataloader should also have the ability to do some sort of pre-processing functionality. So far, the only preprocessing functionality we want to do is resizing, and interpolating. That is, if a model is requesting 5 seconds of video at 30fps, and the underlying video was at 60fps, the dataloader should make sure it can match what the model is asking for.

## Models and model trainer

Unlike the original models in rPPG-Toolbox, the models here *must* be able to predict multiple signals at once. The typical way of implementing this, is that for models that were originally only predicting 1 signal, we just set it up such that there are 2 models in parallel. This means that the models are still true to the original implementation, but at the cost of having a bigger model (more params).

The models must also now be able to not only predict the shape of the waveform, but now, we're looking at making it learn the scale of the waveforms as well. As such, the original implementation of many of these models was just using negative pearson loss. This loses scale information completely. What we now want to do, is to have loss functions that learn scale.

The models now also must be able to handle dictionaries as both inputs and outputs. Ideally, what happens, is that the dataloader gives the model a *batch* (dict), consisting of the input frames, labels and metadata. The models will make the predictions, and actually add the prediction as a key to the batch dict, and that's what gets returned during training and inference. On test, or debugging, this dict can be stored.

The loss functions are different depending on signal type. PPG and ECG signals are unlikely to require scale, but scale is **mandatory** to learn for the pressure signals.

The idea with the models, is that they get saved, alongside their configuration, such that it's easy to then specify to any new dataloader, what the type/shape of the inputs need to be, and whether this model has been trained on it.

## Evaluation

Where `rPPG Toolbox` was only focused on the standardised evaluation of heart rates, this package aims to be more than that. We retain the standard evaluation framework for heart rate, as it's fundamental, but extend this to (currently) pressure signals of `abp` and `cvp`. ABP is clinically useful for 3 specific measures, systolic, diastolic and mean arterial pressure, and there are some standards that have been used to guide our evaluation of it here. We have used the standards ISO-81060 to guide our evaluation of non-invasive blood pressure estimation - (see `standards\ISO-81060` in this project).

## Extending the package

This package aims to be easily extendible. To make this happen, what we should have, is a template for what the addition of new features looks like. This includes a new;

- dataset
- model
- trace (label)

The benefit for clearly describing how to do this, is that when agents are going about migrating existing models, they can reuse the template. 