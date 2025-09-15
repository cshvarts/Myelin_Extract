# Myelin_Extract



"""# Myelin Extraction Pipeline

![Example neurons with myelin shown](example_myelin_vis.png)

This repository contains tools and pipelines for extracting myelin from the [MICrONS](https://www.microns-explorer.org/).

## Classifier Training

The classifiers used for myelin detection are stored in `DNN_classifiers` (model weights) and `classifiers.py` (classifier classes).  

These models were trained on annotation data using the following notebooks:
- `create_dataset_v2.ipynb`
- `make_test_train.ipynb`
- `train_test_model.ipynb`

You generally do **not** need to retrain the classifiers unless you want to:
- Improve performance
- Train on a new EM dataset

## Pipeline Overview

The main myelin extraction pipeline works as follows:

1. **Skeleton Point Selection**  
   - For each neuron, points are selected along the axon skeleton.  
   - If two adjacent skeleton points are more than *3 microns* apart (threshold adjustable), an additional point is inserted.

2. **Image Preprocessing**  
   - Compute the most perpendicular plane to the axon’s direction to obtain a circular cross-section.  
   - Identify the boundary of this cross-section from the segmentation mask.  
   - Unwrap the EM image along the contour to “straighten” the boundary.

3. **Classification**  
   - The processed image is classified using the trained DNN classifier.

## Usage

The helper functions for myelin extraction are provided in `myelin_extraction.py`.  

For batch extraction, use the notebook:  
- `Auto_myelin_v2.ipynb` → Calls `process_neurons()` in `myelin_extraction.py` to extract myelin for a list of neurons.

### Output Format

Results are stored in folders named: segments_myelin_{version}

where `{version}` refers to the client version (e.g., `1507`).  

Each folder contains **extracted myelin traces in segments format**:
- Each segment corresponds to a portion of the axon between two branch points.
- Each segment includes sampled points along the axon and the classifier’s myelin predictions.
- Only axonal segments are included (skeletons filtered accordingly).




