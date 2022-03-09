# vim: expandtab:ts=4:sw=4
'''
Filename: tracker_mod.py
Purpose: This file does handles pedestrian tracking for REGROUP
Author: Angelique Taylor <amt062@eng.ucsd.edu;amt298@cornell.edu>
Note: If you use this in your work, please cite our HRI 2022 paper (see README.md for bibtex)
'''
from __future__ import division, print_function, absolute_import
import numpy as np
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from .track import Track
from .track import TrackState
from scipy import stats
from . import utils
import statistics
from sklearn.preprocessing import *
from scipy import stats
from numpy import linalg as LA

class TrackerMod:
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
    def __init__(self, metric, max_iou_distance=0.9, max_age=30, n_init=5, cost_metric=2): # mod
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init

        self.kf = kalman_filter.KalmanFilter()
        self.tracks = []
        self._next_id = 1
        self.time = 0
        self.cif = 0
        self.gp_dist = 0
        self.h_ratio = 0
        self.cost_metric=cost_metric

    def predict(self):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict(self.kf)

    def setparams(self, cif, gp_dist, h_ratio):
        self.cif = cif
        self.gp_dist = gp_dist
        self.h_ratio = h_ratio

    def update(self, detections,group_labels=None):
        """Perform measurement update and track management for pedestrians.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        self.time += 1
        
        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections = \
            self._match(detections)
      
        ped_group_track_ids = np.zeros(len(detections), dtype=int)
        
        boxes = np.array([d.tlwh for d in detections])
        matched_tracks = []
        matched_detections = []
        # Reset track iou
        for t in self.tracks:
            t.time_since_group_update=-1
            t.iou = 0
        
        # Update track set.
        for track_idx, detection_idx in matches:
            matched_tracks.append(self.tracks[track_idx].track_id)
            self.tracks[track_idx].update(
                self.kf, detections[detection_idx])
            self.tracks[track_idx].time.append(self.time)
            self.tracks[track_idx].iou = utils.Compute_IOU(self.tracks[track_idx].to_tlwh(), detections[detection_idx].tlwh)
            ped_group_track_ids[detection_idx] = self.tracks[track_idx].track_id
            self.tracks[track_idx].time_since_update = 0
            self.tracks[track_idx].state = TrackState.Confirmed
            self.tracks[track_idx].time_since_group_update = 0
            box = detections[detection_idx].tlwh
            self.tracks[track_idx].ground_plane_x.append(box[0] + box[2]*0.5)
            self.tracks[track_idx].ground_plane_y.append(box[1] + box[3])
            self.tracks[track_idx].group_track_id = 0
        
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        for detection_idx in unmatched_detections:
            matched_detections.append(detections)
            self._initiate_track(detections[detection_idx], group_track_id=0)
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

        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets)   
        
        # Estimate Group detections
        spencer_group_labels=group_labels
        group_labels, adj_matrix = self.detect_groups(detections) 
        if(spencer_group_labels is not None):
            group_labels=spencer_group_labels

        CROWDED=0
        v=[]
        if(self.cif!=0):       
            v = utils.GraphDriver(adj_matrix)
            if(len(v) > self.cif):
                CROWDED = 1
        
        for t in self.tracks:
            counter = 0
            for p in ped_group_track_ids:
                if(p == t.track_id):
                    if(CROWDED):
                        if(len(t.group_track_freq_bin_count)>0):
                            idx = max(t.group_track_freq_bin_count)
                            idx = np.where(t.group_track_freq_bin_count == idx)
                            maxi = 0
                            count = 0
                            for i in t.group_track_freq_bin_count:
                                if(i>maxi):
                                    maxi=i
                                    idx = count
                                count+=1
                            group_labels[counter] = t.group_track_freq_value[idx]
                counter+=1       
        
        self.UpdateTrajectoryMatrix(ped_group_track_ids)
        return ped_group_track_ids, group_labels, v
        
    def _match(self, detections):

        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric.distance(features, targets)
            cost_matrix = linear_assignment.gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets, track_indices,
                detection_indices)

            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        if(self.cost_metric==1): # motion
            unmatched_detections = list(range(len(detections)))
            matches_a = []
            unmatched_tracks_a = []
        else: # app and both
            # Associate confirmed tracks using appearance features.
            matches_a, unmatched_tracks_a, unmatched_detections = \
                linear_assignment.matching_cascade(
                    gated_metric, self.metric.matching_threshold, self.max_age,
                    self.tracks, detections, confirmed_tracks)

        # Associate remaining tracks together with unconfirmed tracks using IOU.
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update == 1]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update != 1]
        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)

        if(self.cost_metric==0):
            matches = matches_a + matches_b
            unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        elif(self.cost_metric==1):
            matches = matches_b
            unmatched_tracks = list(set(unmatched_tracks_b))
        else:
            matches = matches_a + matches_b
            unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection, group_track_id):
        mean, covariance = self.kf.initiate(detection.to_xyah())
        self.tracks.append(Track(
            mean, covariance, self._next_id, group_track_id, self.n_init, self.max_age,
            detection, detection.feature))
        
        self._next_id += 1
    
    def MergeBoxes(self, boxes): 
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
        return np.array([x, y, w, h])
        
    def UpdateTrajectoryMatrix(self, ped_group_track_ids):
        traj = np.zeros((len(ped_group_track_ids),len(ped_group_track_ids)))+100
        
        c1=0
        for i in ped_group_track_ids:
            for track1 in self.tracks:
                if(track1.track_id == i):
                    c2=0
                    for j in ped_group_track_ids:
                        for track2 in self.tracks:
                            if(track2.track_id == j):
                                unique_intersection = np.intersect1d(track1.group_time,track2.group_time)
                                
                                a1, a2 = [], []
                                if(len(unique_intersection) >= 10):
                                    for i in range(len(unique_intersection)-10,len(unique_intersection)):
                                        unique_val = unique_intersection[i]
                                        idx1, = np.where(track1.group_time == unique_val)
                                        idx2, = np.where(track2.group_time == unique_val)
                                        a1.append((track1.ground_plane_x[idx1[0]], track1.ground_plane_y[idx1[0]]))
                                        a2.append((track2.ground_plane_x[idx2[0]], track2.ground_plane_y[idx2[0]]))
                                    euclidean_dist = np.sqrt(LA.norm((a1, a2),axis=None))
                                    if(euclidean_dist != 0):
                                        traj[c1][c2] = euclidean_dist
                        c2+=1                        
            c1+=1

    def detect_groups(self, detections):
   
        boxes = np.array([d.tlwh for d in detections])
        boxes_shape = boxes.shape
        group_predictions = []
        adj_matrix = []
        
        # Run Group Detector
        if(boxes_shape[0] > 1):
            group_predictions, adj_matrix = self.SplitBadBoxes(boxes)
        return group_predictions, adj_matrix
    
    def SplitBadBoxes(self, boxes):
        """
        boxes in format <left x, top y, w h>
        """
        boxesShape = np.shape(boxes)
        dist_w = np.ndarray(shape=(len(boxes),len(boxes)))
        dist_h = np.ndarray(shape=(len(boxes),len(boxes)))
        labels = np.zeros(boxesShape[0])
        labels_visited = np.zeros(boxesShape[0])
        
        x_adj = np.ndarray(shape=(len(boxes),len(boxes)))
        y_adj = np.ndarray(shape=(len(boxes),len(boxes)))
        theta_adj = np.ndarray(shape=(len(boxes),len(boxes)))
        
        for j in range(0,len(boxes)):
            for k in range(0,len(boxes)):
                b1 = boxes[j,:]
                b2 = boxes[k,:]
                b1_center = b1[0] + b1[2]/2
                b2_center = b2[0] + b2[2]/2 

                dist_w[j,k] = abs(b1_center - b2_center) - (b1[2]/2 + b2[2]/2)
                dist_h[j,k] = abs(b1[3] - b2[3])
                     
        # Evaluate box distances
        dist_w_temp = dist_w
        next_id = 1
        dist_bottom = np.ndarray(shape=(len(boxes),len(boxes)))
        aggregate = np.ndarray(shape=(len(boxes),len(boxes)))
        
        for j in range(0, len(dist_h)):
            for k in range(0, len(dist_h)):
                maxWidth = statistics.mean([boxes[j,2],boxes[k,2]])
                bottomDiff = abs((boxes[j,1] + boxes[j,3]) - (boxes[k,1] + boxes[k,3]))
                
                minHeight = min(boxes[j,3],boxes[k,3])
                maxHeight = max(boxes[j,3],boxes[k,3])
                hRatio = round((minHeight/maxHeight), 2)

                if(dist_w[j,k] > maxWidth):
                    dist_w[j,k] = 0
                    dist_w[k,j] = 0

                dist_h[j,k] = hRatio
                dist_h[k,j] = hRatio
                dist_bottom[j,k] = bottomDiff
                dist_bottom[k,j] = bottomDiff

                if(bottomDiff >= self.gp_dist):
                    dist_w[j,k] = 0
                    dist_w[k,j] = 0

                if(hRatio <= self.h_ratio):
                    dist_w[j,k] = 0
                    dist_w[k,j] = 0
            
        adj_matrix = np.ndarray(shape=(len(boxes),len(boxes)))

        # Convert dist_w to adj matrix
        for j in range(0, len(dist_w)):
            for k in range(0, len(dist_w)):
                if(dist_w[j,k] == 0 or j == k):
                    adj_matrix[j,k] = 0
                else:
                    adj_matrix[j,k] = 1

        # Normalize metrics
        dist_w = normalize(dist_w, norm='l1')
        dist_h = normalize(dist_h, norm='l1')
        dist_bottom = normalize(dist_bottom, norm='l1')

        for j in range(0, len(dist_h)):
            for k in range(0, len(dist_h)):
                aggregate[j,k] = dist_h[j,k]*dist_bottom[j,k]*dist_w[j,k]
                aggregate[k,j] = dist_h[j,k]*dist_bottom[j,k]*dist_w[j,k]
                if(abs(aggregate[j,k]) <= 0):
                    aggregate[j,k] = 100
                    aggregate[k,j] = 100
        
        pairs = np.zeros(boxesShape[0])
        
        for j in range(0, len(dist_h)):
            pairs[j] = np.argmin(aggregate[j,:], axis=0)
            if(np.sum([aggregate[j,:]]) == len(dist_h)*100):
                pairs[j] = -1
            
        labels = np.zeros(boxesShape[0])
        counter = 0
        for j in range(0, len(pairs)):
            p = j
            while(p != -1):
                p = int(pairs[j])
                if(p != -1):
                    if(labels[j] != 0 or labels[p] != 0):
                        id = max(labels[p], labels[j])
                        labels[j] = id
                        labels[p] = id
                    else:
                        labels[j] = next_id
                        labels[p] = next_id
                        next_id+=1
                counter+=1
                if(counter > len(pairs)):
                    break
                    
        for i in range(0, len(labels)):
            if(labels[i] == 0):
                labels[i] = next_id
                next_id+=1
        
        if(np.sum([labels]) == 0):
            labels = []   
        else:            
            u = np.unique(labels)
            for i in range(0, len(u)):
                idx, = np.where(labels == u[i])
                if(len(idx) > 1):
                    continue
                else:
                    labels[idx] = -1
                    
        return labels, adj_matrix
