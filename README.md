# psypose
This is a smaller, more light-weight version of Psypose. It retains the ability to make pose objects and use display funcitons, but it cannot run any annotations. Use this if you need some basic processing functions on the fly.

## Installation 
The easiest way to get started with PsyPose at this point is to install the psypose_minimal branch using pip:
```
pip install git+https://github.com/scraplab/psypose.git@psypose_minimal
```

## Load a pose object with video file, pose file (.pkl), and face data (.csv).
Below are a few simple lines to get you started. 
```
from psypose import data
t = data.pose()
t.load_video('path_to_video.mp4')
t.load_pkl('path_to_pkl.pkl')
t.load_face_data('path_to_faces.csv')
```

You can now use the pose object (here labeled `t`) in any of the display functions to look at 3D pose, face annotations, or just to view specific frames from the video. 

```
from psypose import display

# View a single frame from the video 
display.frame(t, 100)

# View an interactive 3D pose skeleton
display.track3d(t, 1) # Look at track 1

# View a face
display.face(t, 0) # The 0 corresponds to the 0th row of the face data csv
```

There are also some functions to generate n_people regressors and measures of synchrony. 

```
from psypose import features

# Get a synchrony matrix for every frame. Every idx in the retuned list is n_people x n_people.
sync_mat = features.synchrony_matrix(t)

# Get the average synchrony for all bodies on screen for every frame. Empty frames will be filled with NaNs. 
sync_vec = features.average_synchrony(t)
```


