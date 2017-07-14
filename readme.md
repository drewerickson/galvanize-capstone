# Multimodal Brain Tumor Segmentation from Magnetic Resonance Imaging (MRI) Scans

Drew Erickson  
[Galvanize, Inc.](https://www.galvanize.com/san-francisco)  

This repository is for the Galvanize Capstone Project built in July 2017 for the Data Science Immersive, Cohort 42.

### Summary

Medical imaging data is growing in scale, but the primary source of analysis is human.  Teaching machines to detect
anomalies in these images will reduce the time to appropriate treatment.  MRI scan data for patients with glioblastomas 
represent an optimal challenge for this work, due to the quality of data and the advancement of the processing tools.
Here, I used labeled MRI scans provided by the [BraTS 2017 Challenge](http://braintumorsegmentation.org/), consisting of 
T1, T1-Gd, T2, and FLAIR scans with labels for necrotic tumor, Gd-enhancing tumor, and peritumoral edema.  My approach
was to use a U-Net convolutional neural network model framework, with the addition of batch normalization and inverted 
dropout.  Scan types were stacked together (similar to visual channels), and an additional brain tissue label was 
generated.  Optimal 2D slices were selected for each volume.  Results for this approach will be added soon.

### Methods and Discussion

More detailed description of the work will be added soon.
