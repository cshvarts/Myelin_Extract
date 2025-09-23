
import numpy as np
import imageryclient as ic
import pandas as pd
from caveclient import CAVEclient
from meshparty.skeleton import Skeleton
import matplotlib.pyplot as plt
import preprocess_images as pre
from myelin_dataset import MyelinDataset
from classifiers import SimpleClassifier
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError, ProcessPoolExecutor
import logging
import os
import pickle
import time
# import signal


# _pull_pool = ProcessPoolExecutor(max_workers=16)  # limit concurrent pulls

""" workflow: I give a set of neuron ids to process
    First, check if segments folder exists, if not create it with version in name.
    Then, for each pt_root_id,
        if pt_root_id.pkl exists in this folder, continue. If not, process it process.
            create a segs variable to hold segments
            load the skeleton for this neuron.
            From skeleton, get axon segments
            for each axon segment 
                (process_axon_segment)
                create temp_segment dict w/ pt_position, and 'myelin'
                create_list_of_points_to_check (adds new points if length < thresh)
                check myelin status (on points to check and in parallel) (outputs results)
                Store and append results to temp_segment
                append temp_segment to segs
            save segs to pt_root_id.pkl in segments folder.

"""


def check_myelin_at_point(ctr, pt_root_id, sk_df, img_client, box_sz_microns, model):
    try:
        # print(f"Point: {ctr}")

        # if(box_sz_microns == "adaptive"):
        #     box_sz = 3 #TEMP
        # else:
        #     box_sz = box_sz_microns

        #pull unwrapped image
        plane = pre.pick_normal_plane(ctr, sk_df)
        image, segs = pre.pull_image_and_segmentation(
            img_client, ctr, pt_root_id, plane, box_sz_microns
        )
        image_unwr, contour_sm = pre.unwrap_image_along_boundary(image, segs)

        #resize to make dataset_ready
        image_unwr_ready = pre.resize_and_add_channel(image_unwr, target_size=(40, 100))
        image_unwr_ready = torch.from_numpy(image_unwr_ready).float()             # shape: [40, 100]
        image_unwr_ready = image_unwr_ready.unsqueeze(0)  # Add batch dimension

        #classify myelin
        logits = model(image_unwr_ready)
        prob = torch.sigmoid(logits).squeeze()  # [B, 1] -> [B]
        pred = (prob > 0.5).float().item()

        return ctr, pred


    except Exception as e:
        print(f"Error processing point {ctr}: {e}")
        # return None
        return ctr, -1
    

def check_myelin_at_point_with_timeout(ctr, pt_root_id, sk_df, img_client, box_sz_microns, model):
    timeout_s=15
    attempt = 0
    while True:  # infinite retries
        attempt += 1
        try:
            # only wrap the potentially slow server call
            with ThreadPoolExecutor(max_workers=1) as ex:
                future = ex.submit(
                    pre.pull_image_and_segmentation,
                    img_client, ctr, pt_root_id,
                    pre.pick_normal_plane(ctr, sk_df),
                    box_sz_microns
                )
                image, segs = future.result(timeout=timeout_s)

            # rest of the pipeline (fast, local)
            image_unwr, contour_sm = pre.unwrap_image_along_boundary(image, segs)
            image_unwr_ready = pre.resize_and_add_channel(image_unwr, target_size=(40, 100))
            image_unwr_ready = torch.from_numpy(image_unwr_ready).float().unsqueeze(0)

            logits = model(image_unwr_ready)
            prob = torch.sigmoid(logits).squeeze()
            pred = (prob > 0.5).float().item()

            if attempt > 1:
                print(f"Point {ctr} succeeded on attempt {attempt}")
            return ctr, pred

        except TimeoutError:
            print(f"Point {ctr} attempt {attempt} timed out (> {timeout_s}s), retrying...")
            continue  # retry

        except Exception as e:
            print(f"Error processing point {ctr}: {e}")
            return ctr, -1


#YET ANOTHER WAY...
_pull_thread_pool = ThreadPoolExecutor(max_workers=64)  # bigger than outer pool
def check_myelin_at_point_with_timeout_3(ctr, pt_root_id, sk_df, img_client, box_sz_microns, model):
    timeout_s=15
    attempt = 0
    while True:
        attempt += 1
        try:
            plane = pre.pick_normal_plane(ctr, sk_df)
            fut = _pull_thread_pool.submit(pre.pull_image_and_segmentation,
                                           img_client, ctr, pt_root_id, plane, box_sz_microns)
            image, segs = fut.result(timeout=timeout_s)

            # rest of the pipeline
            image_unwr, contour_sm = pre.unwrap_image_along_boundary(image, segs)
            image_unwr_ready = pre.resize_and_add_channel(image_unwr, target_size=(40, 100))
            image_unwr_ready = torch.from_numpy(image_unwr_ready).float().unsqueeze(0)

            logits = model(image_unwr_ready)
            prob = torch.sigmoid(logits).squeeze()
            pred = (prob > 0.5).float().item()

            if attempt > 1:
                print(f"Point {ctr} succeeded on attempt {attempt}")
            return ctr, pred
        except TimeoutError:
            print(f"Point {ctr} attempt {attempt} timed out (> {timeout_s}s), retrying...")
            continue
        except Exception as e:
            print(f"Error processing point {ctr}: {e}")
            return ctr, -1

# def check_myelin_at_point_with_timeout_2(ctr, pt_root_id, sk_df, img_client, box_sz_microns, model):
    
#     timeout_s=15
#     attempt = 0
#     while True:
#         attempt += 1
#         try:
#             # compute plane outside the subprocess
#             plane = pre.pick_normal_plane(ctr, sk_df)

#             # submit only the pull to the pool
#             future = _pull_pool.submit(
#                 pre.pull_image_and_segmentation,
#                 img_client, ctr, pt_root_id, plane, box_sz_microns
#             )
#             image, segs = future.result(timeout=timeout_s)

#             # rest of your pipeline
#             image_unwr, contour_sm = pre.unwrap_image_along_boundary(image, segs)
#             image_unwr_ready = pre.resize_and_add_channel(image_unwr, target_size=(40, 100))
#             image_unwr_ready = torch.from_numpy(image_unwr_ready).float().unsqueeze(0)

#             logits = model(image_unwr_ready)
#             prob = torch.sigmoid(logits).squeeze()
#             pred = (prob > 0.5).float().item()

#             if attempt > 1:
#                 print(f"Point {ctr} succeeded on attempt {attempt}")
#             return ctr, pred

#         except TimeoutError:
#             print(f"Point {ctr} attempt {attempt} timed out (> {timeout_s}s), retrying…")
#             continue
#         except Exception as e:
#             print(f"Error processing point {ctr}: {e}")
#             return ctr, -1

def check_myelin_in_parallel(points_to_check, pt_root_id, sk_df, img_client, box_sz_microns, model, max_workers):
    
    results = [None] * len(points_to_check)  # preallocate ordered results
    #Check myelin status on all points_to_check (and in parallel)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(
                check_myelin_at_point, ctr, pt_root_id, sk_df, img_client,
                box_sz_microns, model
            ): idx
            for idx, ctr in enumerate(points_to_check)
        }

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            result = future.result()
            if result is not None:
                ctr, pred = result
                results[idx] = (ctr, pred)  # store at the correct position
                print(f"Processed point: {ctr}: Myelin={pred}")
    
    #Store results
    temp_segment = {
                'pt_position': [],
                'myelin': []
    }
    for ctr, pred in results:
        temp_segment['pt_position'].append(ctr.tolist())
        temp_segment['myelin'].append(pred)
    
    return temp_segment

def get_adaptive_radius(sk_dict, axon_segment):
    #box size is axon_radius * 16 /1000, and constrained to be between min and max

    max = 5
    min = 1.6
    axon_radius = sk_dict['radius'][axon_segment[0]]
    scale_factor = 12 #default I started with was 16.

    box_sz = axon_radius * scale_factor / 1000 #convert nm to microns
    if box_sz > max:
        box_sz = max
    if box_sz < min:
        box_sz = min

    return box_sz



def create_list_of_points_to_check(axon_segment, sk, length_thresh=3000):
    #Create list of points to check, adding a new point (midpoint) if dist between points above threshold.
        
    points_to_check = sk.vertices[axon_segment] # Get the vertices of the current segment
    points_to_check = np.vstack((points_to_check, sk.vertices[sk.parent_nodes(axon_segment[-1])])) #Add parent of last vertex

    j = 0
    while j != points_to_check.shape[0]-1:  # While not at the last point
        pt_1 = points_to_check[j]
        pt_2 = points_to_check[j + 1]
        distance = np.linalg.norm(pt_2 - pt_1)  # Calculate the distance
        if distance > length_thresh:  # If distance is above threshold
            # Split the segment into two points
            mid_point = (pt_1 + pt_2) / 2
            points_to_check = np.insert(points_to_check, j + 1, mid_point, axis=0)  # Insert the midpoint
            j-=1 # Stay at the same index to check the new segment
        j+= 1  # Move to the next point

    #remove last point, since it will be at the start of a different segment.
    points_to_check = points_to_check[:-1]
    return points_to_check

def load_skel_get_axon_segs(pt_root_id, client):

    #load skeleton
    sk_df = client.skeleton.get_skeleton(pt_root_id, output_format='swc')
    sk_dict = client.skeleton.get_skeleton(pt_root_id, output_format='dict')
    sk = Skeleton.from_dict(sk_dict)

    #get axon segments
    axon_segments = []
    for i in range(len(sk.segments)):
        if sk_dict['compartment'][sk.segments[i][0]] == 2:  # Check if the segment is axon
            axon_segments.append(sk.segments[i])

    return sk_df, sk_dict, sk, axon_segments


def process_neurons(neur_ids, client, img_client, classifier_model, model_weights_file, max_workers=9, box_sz_microns = "adaptive", length_thresh=3000):
    #box_sz_microns can be an int, or the string "adaptive"

    #suppress warnings with caveclient and urllib3
    logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)

    #check if "segments_" + client.version exists. If not, create it.
    folder_name = f"segments_myelin_{client.version}"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    #initialize model once
    model = classifier_model
    model.load_state_dict(torch.load(model_weights_file))

    #go through and process each neuron
    for pt_root_id in neur_ids:

        #check if neuron already processed
        file_path = os.path.join(folder_name, f"{pt_root_id}.pkl")
        if os.path.exists(file_path):
            print(f"File {file_path} already exists. Skipping neuron {pt_root_id}.")
            continue
        print(f"Processing neuron {pt_root_id}")
        start_neur = time.time()  


        #load skeleton and get axon segments (extents of axon between branch points)
        sk_df, sk_dict, sk, axon_segments = load_skel_get_axon_segs(pt_root_id, client)
        if len(axon_segments) == 0:
            print(f"No axon segments found for neuron {pt_root_id}. Skipping.")
            continue


        segs = [] #this will hold myelin dicts for each axon segment

        #Extract myelin at each point along each axon segment.
        for i in range(len(axon_segments)):
            print(f"Starting segment {i+1}/{len(axon_segments)}")

            #Get points to check
            points_to_check = create_list_of_points_to_check(axon_segments[i], sk, length_thresh) #adds new points if distance b/w pts < thresh.
            points_to_check = (points_to_check / np.array([4, 4, 40])).astype(int)  #convert to voxel coords

            #Get adapative radius (if selected)
            if box_sz_microns == "adaptive":
                box_sz = get_adaptive_radius(sk_dict, axon_segments[i])
            else:
                box_sz = box_sz_microns

            #Check myelin status on all points_to_check (in parallel)
            start = time.time()                  
            temp_segment = check_myelin_in_parallel(points_to_check, pt_root_id, sk_df, img_client, box_sz, model, max_workers)
            end = time.time()

            segs.append(temp_segment)

            print(f"Finished segment {i+1}/{len(axon_segments)} for neuron {pt_root_id}")
            print(f"Time per point: {(end - start)/len(points_to_check):.2f} seconds")

        #save segs to pt_root_id.pkl in segments folder.
        with open(file_path, 'wb') as f:
            pickle.dump(segs, f)
        end_neur = time.time()  

        print(f"Finished neuron {pt_root_id} in {(end_neur - start_neur)/60:.2f} minutes")
