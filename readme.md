# Galvanize Capstone Project

## Drew Erickson

This repository is for the Galvanize Capstone Project that I will be completing in July 2016 for the Data Science Immersive, Cohort 42.

## Capstone Project Proposals:

### 1) Detection of patients at risk for hospital readmission
 
Hospital readmissions are a costly portion of hospital and insurance costs, so detection of patients at risk for 
readmission can allow preventative interventions to be taken.  This work can be generalized to any readmission in an
urgent care setting, or specified towards a specific health condition (like diabetes).  Research such as this 
(https://arxiv.org/pdf/1602.04257.pdf) address methods to approach this problem.  I will likely approach this as a 
classification problem, with a number of interpretive and predictive models.  The presentation will likely involve
graphs showing the accuracy, precision and recall of various models, along with comparisons of the usefulness of 
different attrtibutes.  One possible source of datasets for this work is here:
- https://mimic.physionet.org/

### 2) Identification of anomalies in radiological scan images

Radiological data is growing in scale, but the primary source of analysis is human.  Teaching machines to detect
anomalies in these images will reduce the time to appropriate treatment.  This problem can have a number of specific
variations, based on anatomical location focus and/or anomaly type.  A common one would be cancerous tumors in a region
of the torso, such as this work on breast cancer detection (http://stm.sciencemag.org/content/3/108/108ra113).  This is
primarily an image classification problem.  Some possible ways to approach this problem are discussed here 
(https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3372692/#S14title), although it also seems likely that this problem may 
involve more recent advances in deep learning, as seen discussed here (https://rubinlab.stanford.edu/publications).  
The presentation will involve graphs explaining the model selection and showing the effectiveness of the model.  Some 
possible datasets for this work are here:
- http://www.cancerimagingarchive.net/
- http://langlotzlab.stanford.edu/projects/medical-image-net/
- http://www.cardiacatlas.org/studies/sunnybrook-cardiac-data/
 
### 3) Identifying health issues through wearable-collected biometrics
 
Wearable devices and mobile phone apps can collect data through common inputs (accelerometers, microphone) that
can be used to detect health issues before they lead to a hospital visit, allowing users to intervene early and improve
their future health.  This problem has a number of specific formulations to tackle, depending on the type of wearable
you are collecting data from.  This is likely a classification problem in which a particular set of health conditions
(such as depression, insomnia, sleep apnea, and/or cardio health) relevant to the data are selected for detection.
Presentation will involve showing how the limited dataset can distinguish the health condition(s) and showing the
effectiveness of the model.  Sources for this data are undetermined at this time, but would likely come from a company
producing a wearable device or mobile phone app.

