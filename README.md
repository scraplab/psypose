# psypose
Tools for processing pose estimation for psychological research.

## Installation 
The easiest way to get started with PsyPose at this point is to install the PARE branch (actively in development) using pip:
```
pip install git+https://github.com/scraplab/psypose.git@psypose_pare
```

## Basic Annotation
Below are a few simple lines to get you started. 
```
from psypose import data, extract
vid_path = 'path/to/video'
t = data.pose()
t.load_video(vid_path)
extract.annotate(t, output_path = 'where/to/save/ouputs')
```
This will save one .pkl file and one .csv, which correspond to body and face data, respectively. 

*Note: Psypose requires some pre-trained model files - these will be downloaded automatically the first time psypose is imported, and they take up a couple gigs of space. 
