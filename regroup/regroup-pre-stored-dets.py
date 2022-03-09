# vim: expandtab:ts=4:sw=4
from __future__ import division, print_function, absolute_import
'''
Filename: regroup.py
Purpose: This file does runs REGROUP on pre-stored pedestrian detections
Author: Angelique Taylor <amt062@eng.ucsd.edu;amt298@cornell.edu>
Note: If you use this in your work, please cite our HRI 2022 paper (see README.md for bibtex)

Usage: python regroup-pre-stored-dets.py --sequence_dir=../data/test/[sequence]/  --display=1 --output_file=../data/test/[sequence]/[sequence].txt
'''
import argparse
import os
import cv2
import numpy as np
import errno
import tensorflow as tf
import time
from application_util import preprocessing
from application_util import visualization
from regroup import nn_matching
from regroup.detection import Detection
from regroup.tracker import Tracker
from regroup.groupTracker import GroupTracker
from regroup import utils

save_peds=True

def _run_in_batches(f, data_dict, out, batch_size):
    data_len = len(out)
    num_batches = int(data_len / batch_size)

    s, e = 0, 0
    for i in range(num_batches):
        s, e = i * batch_size, (i + 1) * batch_size
        batch_data_dict = {k: v[s:e] for k, v in data_dict.items()}
        out[s:e] = f(batch_data_dict)
    if e < len(out):
        batch_data_dict = {k: v[e:] for k, v in data_dict.items()}
        out[e:] = f(batch_data_dict)

def extract_image_patch(image, bbox, patch_shape):
    """Extract image patch from bounding box.
    Parameters
    ----------
    image : ndarray
        The full image.
    bbox : array_like
        The bounding box in format (x, y, width, height).
    patch_shape : Optional[array_like]
        This parameter can be used to enforce a desired patch shape
        (height, width). First, the `bbox` is adapted to the aspect ratio
        of the patch shape, then it is clipped at the image boundaries.
        If None, the shape is computed from :arg:`bbox`.
    Returns
    -------
    ndarray | NoneType
        An image patch showing the :arg:`bbox`, optionally reshaped to
        :arg:`patch_shape`.
        Returns None if the bounding box is empty or fully outside of the image
        boundaries.
    """
    bbox = np.array(bbox)
    if patch_shape is not None:
        # correct aspect ratio to patch shape
        target_aspect = float(patch_shape[1]) / patch_shape[0]
        new_width = target_aspect * bbox[3]
        bbox[0] -= (new_width - bbox[2]) / 2
        bbox[2] = new_width

    # convert to top left, bottom right
    bbox[2:] += bbox[:2]
    bbox = bbox.astype(np.int)

    # clip at image boundaries
    bbox[:2] = np.maximum(0, bbox[:2])
    bbox[2:] = np.minimum(np.asarray(image.shape[:2][::-1]) - 1, bbox[2:])
    if np.any(bbox[:2] >= bbox[2:]):
        return None
    sx, sy, ex, ey = bbox
    image = image[sy:ey, sx:ex]
    image = cv2.resize(image, tuple(patch_shape[::-1]))
    return image

class ImageEncoder(object):

    def __init__(self, checkpoint_filename, input_name="images",
                 output_name="features"):
        self.session = tf.Session()
        with tf.gfile.GFile(checkpoint_filename, "rb") as file_handle:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(file_handle.read())
        tf.import_graph_def(graph_def, name="net")
        self.input_var = tf.get_default_graph().get_tensor_by_name(
            "net/%s:0" % input_name)
        self.output_var = tf.get_default_graph().get_tensor_by_name(
            "net/%s:0" % output_name)

        assert len(self.output_var.get_shape()) == 2
        assert len(self.input_var.get_shape()) == 4
        self.feature_dim = self.output_var.get_shape().as_list()[-1]
        self.image_shape = self.input_var.get_shape().as_list()[1:]

    def __call__(self, data_x, batch_size=32):
        out = np.zeros((len(data_x), self.feature_dim), np.float32)
        _run_in_batches(
            lambda x: self.session.run(self.output_var, feed_dict=x),
            {self.input_var: data_x}, out, batch_size)
        return out

def create_box_encoder(model_filename, input_name="images",
                       output_name="features", batch_size=32):
    image_encoder = ImageEncoder(model_filename, input_name, output_name)
    image_shape = image_encoder.image_shape

    def encoder(image, boxes):
        image_patches = []
        for box in boxes:
            patch = extract_image_patch(image, box, image_shape[:2])
            if patch is None:
                print("WARNING: Failed to extract image patch: %s." % str(box))
                patch = np.random.uniform(
                    0., 255., image_shape).astype(np.uint8)
            image_patches.append(patch)
        image_patches = np.asarray(image_patches)
        return image_encoder(image_patches, batch_size)

    return encoder

def gather_sequence_info(sequence_dir, group_detector):
    """Gather sequence information, such as image filenames, detections,
    groundtruth (if available).

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.

    Returns
    -------
    Dict
        A dictionary of the following sequence information:

        * sequence_name: Name of the sequence
        * image_filenames: A dictionary that maps frame indices to image
          filenames.
        * detections: A numpy array of detections in MOTChallenge format.
        * groundtruth: A numpy array of ground truth in MOTChallenge format.
        * image_size: Image size (height, width).
        * min_frame_idx: Index of the first frame.
        * max_frame_idx: Index of the last frame.

    """
    rgb_image_dir = os.path.join(sequence_dir, "rgb")
    rgb_image_filenames = {
        int(os.path.splitext(f)[0]): os.path.join(rgb_image_dir, f)
        for f in os.listdir(rgb_image_dir)}

    groundtruth_file = os.path.join(sequence_dir, "gt/gt.txt")
    detection_file_new = os.path.join(sequence_dir, "det/det.txt")
    detections = np.loadtxt(detection_file_new, delimiter=',')
    groundtruth = None
    
    if os.path.exists(groundtruth_file):
        groundtruth = np.loadtxt(groundtruth_file, delimiter=',')

    if len(rgb_image_filenames) > 0:
        rgb_image = cv2.imread(next(iter(rgb_image_filenames.values())),
                           cv2.IMREAD_GRAYSCALE)
        rgb_image_size = rgb_image.shape
    else:
        rg_image_size = None

    image_size = rgb_image_size

    if len(rgb_image_filenames) > 0:
        min_frame_idx = min(rgb_image_filenames.keys())
        max_frame_idx = max(rgb_image_filenames.keys())
    else:
        min_frame_idx = int(detections[:, 0].min())
        max_frame_idx = int(detections[:, 0].max())

    info_filename = os.path.join(sequence_dir, "seqinfo.ini")
    if os.path.exists(info_filename):
        with open(info_filename, "r") as f:
            line_splits = [l.split('=') for l in f.read().splitlines()[1:]]
            info_dict = dict(
                s for s in line_splits if isinstance(s, list) and len(s) == 2)

        update_ms = 1000 / int(info_dict["frameRate"])
    else:
        update_ms = None

    feature_dim = detections.shape[1] - 10 if detections is not None else 0
    seq_info = {
        "prev_image": [],
        "sequence_name": os.path.basename(sequence_dir),
        "rgb_image_filenames": rgb_image_filenames,
        "detections": detections,
        "groundtruth": groundtruth,
        "image_size": image_size,
        "min_frame_idx": min_frame_idx,
        "max_frame_idx": max_frame_idx,
        "feature_dim": feature_dim,
        "update_ms": update_ms
    }
    return seq_info


def create_detections(detection_mat, frame_idx, image, min_height=0):
    """Create detections for given frame index from the raw detection matrix.

    Parameters
    ----------
    detection_mat : ndarray
        Matrix of detections. The first 10 columns of the detection matrix are
        in the standard MOTChallenge detection format. In the remaining columns
        store the feature vector associated with each detection.
    frame_idx : int
        The frame index.
    min_height : Optional[int]
        A minimum detection bounding box height. Detections that are smaller
        than this value are disregarded.

    Returns
    -------
    List[tracker.Detection]
        Returns detection responses at given frame index.

    """
    frame_indices = detection_mat[:, 0].astype(np.int)
    mask =  frame_indices == frame_idx
    rows = detection_mat[mask]
    
    features = encoder(image, rows[:, 2:6].copy())
    confidence = rows[:,1]
    detection_list = []
    for row, feature in zip(rows, features):
        bbox, confidence = row[2:6], row[6]
        if bbox[3] < min_height:
            continue
        detection_list.append(Detection(bbox, confidence, feature))
    
    return detection_list


def run(sequence_dir, output_file, min_confidence,
        nms_max_overlap, min_detection_height, distance_metric, max_cosine_distance,
        nn_budget, display, img_output_path, cif, gp_dist, h_ratio, encoder, cost_metric, group_detector):
    """Run multi-target tracker on a particular sequence.

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
    detection_file : str
        Path to the detections file.
    output_file : str
        Path to the tracking output file. This file will contain the tracking
        results on completion.
    min_confidence : float
        Detection confidence threshold. Disregard all detections that have
        a confidence lower than this value.
    nms_max_overlap: float
        Maximum detection overlap (non-maxima suppression threshold).
    min_detection_height : int
        Detection height threshold. Disregard all detections that have
        a height lower than this value.
    max_cosine_distance : float
        Gating threshold for cosine distance metric (object appearance).
    nn_budget : Optional[int]
        Maximum size of the appearance descriptor gallery. If None, no budget
        is enforced.
    display : bool
        If True, show visualization of intermediate tracking results.
    cif : crowd indication feature
    gp_dist : ground plane
    h_ratio : height ratio threshold
    """
    seq_info = gather_sequence_info(sequence_dir,group_detector)
    metric = nn_matching.NearestNeighborDistanceMetric(
        distance_metric, max_cosine_distance, nn_budget)
    
    groupTracker = GroupTracker(metric, cost_metric)
    groupTracker.setparams(cif, gp_dist, h_ratio)
    results = []
    
    
    def frame_callback(vis, frame_idx):

        if frame_idx in seq_info["rgb_image_filenames"]:

            print("Processing frame %05d" % frame_idx)

            # Load image and generate detections.
            image = cv2.imread(
                    seq_info["rgb_image_filenames"][frame_idx], cv2.IMREAD_COLOR)

            detections = create_detections(
                seq_info["detections"], frame_idx, image, min_detection_height)
            detections = [d for d in detections if d.confidence >= min_confidence]

            # Run non-maxima suppression.
            boxes = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            indices = preprocessing.non_max_suppression(
                boxes, nms_max_overlap, scores)
            detections = [detections[i] for i in indices]
            features = np.array([d.feature for d in detections])
            
            # Update tracker.
            groupTracker.predict_groups()
            ped_detections, group_detections, group_track_ids, ped_track_ids = groupTracker.update_groups(detections)
            
            # Update visualization.
            if display==True:
                vis.set_image(image.copy())
                vis.draw_detections(detections)
                vis.draw_trackers(groupTracker.pedTracker.tracks)
                
                if(img_output_path is not None):
                    cv2.imwrite(img_output_path + "/" + str(frame_idx) + ".png", vis.viewer.image)
            idx=None
            
            ped_detections = np.array([d.tlwh for d in detections])

            ped_length=0
            for track in groupTracker.pedTracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 0:
                    continue
                ped_length+=1

            if(save_peds):
                for track in groupTracker.pedTracker.tracks:
                    if(ped_length<2):
                        continue
                    if not track.is_confirmed() or track.time_since_update > 0:
                        continue
                    if(track.group_track_id==0):
                        continue
                    ped_box = track.to_tlwh().astype(np.int)

                    
                    iou=0
                    choice_idx=-1
                    for idx, box in enumerate(ped_detections):
                        iou_temp = utils.Compute_IOU(box,ped_box)
                        if(iou_temp>iou):
                            choice_idx=idx
                            iou=iou_temp
                    results.append([
                            frame_idx,track.track_id,track.group_track_id, ped_box[0],ped_box[1],ped_box[2],ped_box[3]])
            else:
                for i in range(0, len(group_track_ids)):
                    group = group_detections[i]
                    if(group_track_ids[i] != 0):
                        results.append([
                            frame_idx, group_track_ids[i], group[0], group[1], group[2], group[3]])
            
        else:
            print("WARNING could not find image for frame %d" % frame_idx)

    # Run tracker
    if display:
        visualizer = visualization.Visualization(seq_info, update_ms=5)
    else:
        visualizer = visualization.NoVisualization(seq_info)
    visualizer.run(frame_callback)

    # Store results
    f = open(output_file, 'w')
    for row in results:
        if(save_peds):
            print('%d,%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (
                row[0], row[1], row[2], row[3], row[4], row[5], row[6]),file=f)
        else:
            print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (
                row[0], row[1], row[2], row[3], row[4], row[5]),file=f)
    
def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="REGROUP")
    parser.add_argument(
        "--sequence_dir", help="Path to MOTChallenge sequence directory", default=None, required=True)
    parser.add_argument(
        "--detection_file", help="Path to custom detections.", default=None, required=False)
    parser.add_argument(
        "--distance_metric", help="Path to custom detections.", default="cosine", required=False)
    parser.add_argument(
        "--output_file", help="Path to the tracking output file. This file will contain the tracking results on completion.", default="hypotheses.txt")
    parser.add_argument(
        "--min_confidence", help="Detection confidence threshold",default=-1, type=int)
    parser.add_argument(
        "--min_detection_height", help="Threshold on the detection bounding", default=40, type=int)
    parser.add_argument(
        "--cif", help="Crowd indication feature threshold" , default=4, type=int)
    parser.add_argument(
        "--nms_max_overlap", help="Non-maxima suppression threshold", default=0.7, type=float)
    parser.add_argument(
        "--max_cosine_distance", help="Gating threshold for cosine distance metric (object appearance).", type=float, default=0.9)
    parser.add_argument(
        "--nn_budget", help="Maximum size of the appearance descriptors gallery.", type=int, default=100)
    parser.add_argument(
        "--display", help="Show intermediate tracking results", default=0, type=int)
    parser.add_argument(
        "--img_output_path", help="path to write images", default=None)
    parser.add_argument(
        "--group_detector", help="group detection method", default='regroup')
    parser.add_argument(
        "--model", default="./resources/networks/mars-small128.pb", help="Path to freezed inference graph protobuf.")
    parser.add_argument(
        "--gp_dist", help="Ground plane distance threshold", default=20, type=int)
    parser.add_argument(
        "--h_ratio", help="Height ratio threshold [0,1]", default=0.8, type=float)
    parser.add_argument(
        "--cost_metric", help="cost_metric = {0-appearance, 1-motion, 2-both}" , default=2, type=int)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    encoder = create_box_encoder(args.model, batch_size=32)
    if(args.img_output_path is not None):
        img_create_dir = args.img_output_path
        try: 
            os.makedirs(img_create_dir)
        except OSError:
            if not os.path.isdir(img_create_dir):
                raise
    
    start = time.time()
    # Run group tracker
    run(
        args.sequence_dir, args.output_file, args.min_confidence, args.nms_max_overlap, args.min_detection_height, 
        args.distance_metric, args.max_cosine_distance, args.nn_budget, args.display, args.img_output_path, args.cif, 
        args.gp_dist, args.h_ratio, encoder, args.cost_metric,args.group_detector)
    
    # Print runtime
    end = time.time()
    total_time = end-start
    print('Total time: {}'.format(total_time))
