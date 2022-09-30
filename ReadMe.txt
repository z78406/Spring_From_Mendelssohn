# Created on 9/3/2022







# ----------------- About Data Structure ------------------
for frame: (see frame.hpp)
	unique_pixel_id that saves "global ID" of every keypoint in the
	current frame.
	unique_pixel_has_match:


# ----------------- Recording matching info ---------------

1. pair-based feature match will return "relative-index" matching result
e.g., Match2viewSIFT((frame 1, frame 2, matches) will return a maching result (for frame 1)
tells for each keypoint in 1(queryIdx), the relative-index of matched keypoint in 2(trainIdx).
Optionally, result will be refined by EstimateE5points_RANSAC() and provide inlider matches.

2. assign unified ID to matched keypoints.
e.g., LinkMatchedPointID() / AssignUnmatchedPointId
If a frame is an initial frame or has unmatched point (-1 in vector: unique_pixel_id),
It will be assign a "global unique ID" by AssignUnmatchedPointId.
Otherwise, every matched keypoint in main frame will be assigned the same global ID
to its reference frame (usually frame index) unless there is 1-many point match
between the main frame and reference frame. This is provided in LinkMatchedPointID().
Certeinly, unmatched keypoints in each main frame will go through AssignUnmatchedPointId
which assigns them a unique global ID. Finally, both matched and unmatched keypoints
will be given their global ID (unique).

In sum, matcher of the frame will provide info of matched keypoints at relative order
use frame.unique_pixel_ids[matcher[i].queryIdx] to find ith keypoint's global ID.


3. correspondence between keypoints in the frames and point cloud points.



