# just counting how many sessions have pose

from pathlib import Path

mydir = Path('/safestore/users/landry/SCRAP/data/andromeda_storage/conversations_unconstrained')
# get all the sessions in the data directory
dirs = [f for f in mydir.iterdir() if f.is_dir()]
dirs = [i for i in dirs if '_' in i.name]
dirs = [i for i in dirs if len(i.name.split('_')[1]) == 3]
dirs = [i for i in dirs if (i / 'derivatives' / 'cam1_concatenated_trimmed.mp4').exists()]
dirs = [i for i in dirs if (i / 'derivatives' / 'cam2_concatenated_trimmed.mp4').exists()]

total = len(dirs)
processed = 0
for i in dirs:
    p1, p2 = i / 'processed' / 'cam1_pose.pkl', i / 'processed' / 'cam2_pose.pkl'
    if p1.exists() and p2.exists():
        processed += 1
print(f'{processed}/{total} eligible sessions have been processed for pose.')

processed = 0
for i in dirs:
    cam1_faces = i / 'processed' / 'cam1_faces.txt'
    cam2_faces = i / 'processed' / 'cam2_faces.txt'
    if cam1_faces.exists() and cam2_faces.exists():
        processed += 1

print(f'{processed}/{total} eligible sessions have been processed for face detection.')
print('\nEligibility is determined by the presence of both cam1 and cam2 videos that have been concatenated, aligned, and trimmed.')

