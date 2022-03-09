# vim: expandtab:ts=4:sw=4
'''
Filename: groupTracker.py
Purpose: This file is the main REGROUP tracker driver
Author: Angelique Taylor <amt062@eng.ucsd.edu;amt298@cornell.edu>
Note: If you use this in your work, please cite our HRI 2022 paper (see README.md for bibtex)
'''
from __future__ import absolute_import
import numpy as np
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from regroup.tracker_mod import TrackerMod
from .grouptrack import GroupTrack
import numpy as np
from application_util import preprocessing
from . import utils

from . import detection
import statistics
from sklearn.preprocessing import *
from scipy import stats

CROWDED = 0

class GroupTracker(TrackerMod):
    """
    This is the multi-target tracker.

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.

    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories in image space.
    tracks : List[Track]
        The list of active tracks at the current time step.

    """

    def __init__(self, metric, cost_metric):
        TrackerMod.__init__(self, metric, max_age=30, n_init=3, cost_metric=cost_metric)
        self.pedTracker = TrackerMod(metric,cost_metric=cost_metric)
        self.time = 0
        self.cif = 0
        
    def predict_groups(self):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """        
        self.pedTracker.predict()
        self.predict()

    def setparams(self, cif, gp_dist, h_ratio):
        self.cif = cif
        self.pedTracker.setparams(cif, gp_dist, h_ratio)

    def update_groups(self, detections,group_labels=None):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.
        """
        self.time += 1

        ped_detections = detections
        ped_group_matches, group_labels, v = self.pedTracker.update(ped_detections,group_labels=group_labels)

        CROWDED = 0
        # Estimate Crowd feature
        if(len(v) > self.cif):
            CROWDED = 1
        
        detections, ped_group_idx = self.UpdateBoxes(detections, group_labels, 0) # BE CAREFUL WHEN CHANGING T
        
        boxes = np.array([d.tlwh for d in detections])

        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections = \
            self._match(detections)

        group_track_ids = np.zeros(len(ped_group_idx), dtype=int)

        # Update track set.
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(
                self.kf, detections[detection_idx])
            self.tracks[track_idx].group_track_id = self.tracks[track_idx].track_id
            group_track_ids[detection_idx] = self.tracks[track_idx].track_id 
            self.tracks[track_idx].time_since_group_update = 0
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        for detection_idx in unmatched_detections:
            id = self._initiate_group_track(detections[detection_idx], ped_group_matches[ped_group_idx[detection_idx]])
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Update distance metric.
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            track.features = []
        self.metric.group_partial_fit(
            np.asarray(features), np.asarray(targets), active_targets)
        
        # Update pedestrian group ids
        for group_idx, group in enumerate(ped_group_idx):
            group_id = group_track_ids[group_idx]
            for ped in group:
                ped_track_id  = ped_group_matches[ped]
                for t in self.pedTracker.tracks:
                    if t.is_confirmed() and t.track_id == ped_track_id:
                        t.group_track_id = group_id
                        if(group_id != 0):
                            if(len(t.group_track_freq) == 0):
                                t.group_track_freq_value.append(group_id)
                                t.group_track_freq_bin_count.append(1)
                            else:
                                if(t.group_track_freq[len(t.group_track_freq)-1] == group_id):
                                    t.group_track_freq_bin_count[len(t.group_track_freq_bin_count)-1] += 1
                                else:
                                    t.group_track_freq_bin_count.append(1)
                                    t.group_track_freq_value.append(group_id)
                        
                            t.group_track_freq.append(t.group_track_id)
                            t.group_track_freq_unique = np.unique(t.group_track_freq)
                            t.group_time.append(self.time)
                            t.time_since_group_update=0
        
        return ped_detections, boxes, group_track_ids, ped_group_matches

    def UpdateGroupDetections(self):
        """
        Update group detections
        Input:
            - group_tracks:  group tracks
        Output:
            - group_tracks: updated group tracks
            - u: track ids
        """
        # Get ped group time updates
        group_time = np.array([t.group_time for t in self.pedTracker.tracks if t.is_confirmed()])
        
        gtt=[]
        for i in range(0, len(group_time)):
            gt = group_time[i]
            if(len(gt) > 1):
                if(gt[len(gt)-1] == self.time):
                    gtt.append(gt[len(gt)-1])
                else:
                    gtt.append(-1)
            else:
                gtt.append(-1)
        
        # Get ped detections
        ped_det = np.array([t.to_tlwh() for t in self.pedTracker.tracks if t.is_confirmed()])
        
        # Get group track state
        group_state = np.array([t.group_state for t in self.pedTracker.tracks if t.is_confirmed()])
        ped_features = np.array([t.features for t in self.pedTracker.tracks if t.is_confirmed()])
        
        # Get group detections
        group_det = np.array([t.to_tlwh() for t in self.tracks if t.is_confirmed()])
        
        # Get ped group track ids
        ped_group_track_id = [t.group_track_id for t in self.pedTracker.tracks if t.is_confirmed()]
        
        # Get group track ids
        group_track_id = [t.track_id for t in self.tracks if t.is_confirmed()]
        u = np.unique(ped_group_track_id)
        
        updated_group_det = []
        updated_group_track_ids = []
        counter = 0
        for i in range(0, len(u)):
            idx, = np.where(ped_group_track_id == u[i])
            idxx = [id for id in idx if gtt[id] != -1]
            
            if(len(idxx) > 1):
                gt = group_time[idxx]
                
                # Get ped detections
                boxes = ped_det[idxx,:]
                
                # Get ped features
                features = ped_features[idxx,:]
                fsum = np.sum([features for i in idxx], axis=0)
                
                # Merge boxes
                mboxes = self.MergeBox(boxes)
                updated_group_det.append(np.array(mboxes))
                updated_group_track_ids.append(ped_group_track_id[idxx[0]])

        return updated_group_det, updated_group_track_ids

    def _initiate_group_track(self, detection, group_id):
        id = self._next_id
        mean, covariance = self.kf.initiate(detection.to_xyah())
        self.tracks.append(GroupTrack(
            mean, covariance, self._next_id, group_id, self.n_init, self.max_age,
            detection, detection.feature))
        self._next_id += 1
        return id

    def UpdateBoxes(self, detections, group_labels, T):
        """
        Updates feature map of groups
        """
        boxes = np.array([d.tlwh for d in detections])
        unique_labels = np.unique(group_labels)
        mergedBB = np.array([])
        features = np.array([])
        mergedBB_idx = []
        group_detections = []

        for i in range(0, len(unique_labels)):
            f=[]
            if(unique_labels[i] != -1):
                idx, = np.where(group_labels == unique_labels[i])
                if(len(idx) > 1):
                    mergedBB_idx.append(idx)
                    
                    fsum = np.sum([detections[i].feature for i in idx], axis=0) # sum > mean
                    
                    f = [detections[i].feature for i in idx]
                    
                    x, y, w, h = self.MergeBox(boxes[idx,:])
                    area = w*h
                    if(area > T):
                        box = ((x - w/2), (y - h/2), (x + w/2), (y + h/2))
                        x_ = (x - w/2)
                        y_ = (y - h/2)
                        features = np.array([ ((x),(y),(w),(h)) ])
                        if(mergedBB.size == 0):
                            mergedBB = np.array([ ((x),(y),(w),(h)) ])
                        else:
                            mergedBB = np.vstack((mergedBB, np.array([((x),(y),(w),(h))]) ))
                        group_detections.append(detection.Detection(((x),(y),(w),(h)), -1, fsum))
		                
        boxes = np.array([d.tlwh for d in group_detections])
        return group_detections, mergedBB_idx
        
    def MergeBox(self, boxes): 
        """
        Return x <left row>, y <top col>, w <width> and h <height>
        """
        min_row = min(boxes[:,1])
        max_row = max(boxes[:,1] + boxes[:,3])
        min_col = min(boxes[:,0])
        max_col = max(boxes[:,0] + boxes[:,2])

        x = min_col
        if(x < 0):
            x = 1
            
        y = min_row
        if(y < 0):
            y = 1
        w = max_col - min_col
        h = max_row - min_row
        
        return x, y, w, h
