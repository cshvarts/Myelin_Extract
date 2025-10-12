import numpy as np
import pandas as pd
from caveclient import CAVEclient
from meshparty.skeleton import Skeleton
import os
import pickle

def denoise_segs(segs):
    #Goes through with a rolling window of 5.
    
    segs_denoise = []

    for i in range(len(segs)):
        temp = segs[i]
        temp['myelin'] = np.where(temp['myelin'] == -1, 0.5, temp['myelin']) #set error entries to 0.5
        temp['myelin'] = np.concatenate(([0], [0], temp['myelin'], [0], [0])) #pad with two zeros on each side
        temp['myelin'] = pd.Series(temp['myelin']).rolling(window=5, center=True).mean().to_numpy() #apply rolling mean with window of 5
        temp['myelin'] = np.where(temp['myelin'] >= 0.5, 1, 0) #threshold at 0.5
        temp['myelin'] = temp['myelin'][2:-2] #remove padding
        segs_denoise.append(temp)
    return segs_denoise

def get_total_myelin_length(segs):
    total_length = 0
    for seg in segs:
        positions = np.array(seg['pt_position'])
        myelin_flags = np.array(seg['myelin'])
        for i in range(len(positions) - 1):
            if myelin_flags[i] == 1 and myelin_flags[i + 1] == 1:
                p1 = positions[i] * np.array([4, 4, 40])  # Convert back to nm
                p2 = positions[i + 1] * np.array([4, 4, 40])
                length = np.linalg.norm(p2 - p1)
                total_length += length
    return total_length

def create_myelin_info_df(directory, cell_type_df):
    # makes a myelin df from all .pkl files in given directory.
    #   For now has columns:
    # 'pt_root_id', 'total_myelin_length','cell_type', 'number_of_segments'
    myelin_info_df = pd.DataFrame(columns=['pt_root_id', 'total_myelin_length','cell_type', 'number_of_segments'])

    for filename in os.listdir(directory):
        if filename.endswith('.pkl'):
            pt_root_id = int(filename.split('.')[0])
            print(pt_root_id)
            
            with open(os.path.join(directory, filename), 'rb') as f:
                segs = pickle.load(f)
            segs_denoised = denoise_segs(segs)
            total_myelin_length = get_total_myelin_length(segs_denoised)/1000 #convert to microns 
            total_myelin_length = round(total_myelin_length, 2)  #round to 2 decimal places
            # cell_type = cell_type_df[cell_type_df['pt_root_id'] == pt_root_id]['cell_type'].values[0]
            #if cell type not found, set to unknown
            if len(cell_type_df[cell_type_df['pt_root_id'] == pt_root_id]) == 0:
                cell_type = 'unknown'
            else:
                cell_type = cell_type_df[cell_type_df['pt_root_id'] == pt_root_id]['cell_type'].values[0]
            number_of_segments = len(segs_denoised)

            myelin_info_df = pd.concat([
                myelin_info_df,
                pd.DataFrame([{
                    'pt_root_id': pt_root_id,
                    'total_myelin_length': total_myelin_length,
                    'cell_type': cell_type
                    ,'number_of_segments': number_of_segments
                    
                }])
            ], ignore_index=True)
    return myelin_info_df


def get_delay_from_l_and_r(length, radius, myelin_status, k_myelinated = 6, k_unmyelinated = .3):
    #Get delay in ms from length and radius in microns.
    #radius in microns.
    #unmyelinated -> v = k_m * sqrt(d)    d in microns, v in m/s
    #myelinated -> v = k_u * d    d in microns, v in m/s
    d = 2 * radius
    if myelin_status == 1:
        v = k_myelinated * d  # in microns/us
    if myelin_status == 0:
        v = k_unmyelinated * np.sqrt(d) # in microns/us
    delay = length / v  # in us
    return delay / 1000  # convert to ms

def get_info_about_each_segment(sk_dict, segs_myelin):

    ''' Get's info about each axonal segment
        including path length, myelin length, delay, parent segment, radius
    '''
    sk = Skeleton.from_dict(sk_dict)
    segs = segs_myelin
    
    #make segs_info dictionary
    segs_info = {}
    segs_info['path_length'] = []
    segs_info['myelin_length'] = []
    segs_info['delay'] = []
    segs_info['delay_unmyel'] = []
    segs_info['parent'] = []
    segs_info['radius'] = []
    segs_info['map_to_skel_segments'] = []
    # segs_info['root'] = []

    #map myelin segments (along axon) to skeleton segments (whole neuron)
    parent_seg_skel = []
    skel_to_axon_map = {}
    axon_idx = 0
    for i in range(len(sk.segments)):
        if sk_dict['compartment'][sk.segments[i][0]] == 2:  # Check if the segment is axon
            segs_info['map_to_skel_segments'].append(i)
            segs_info['radius'].append(sk_dict['radius'][sk.segments[i][0]] / 1000)  # Get radius of the first node in the segment

            #also get parent of last node - this will be the branch point, and start of parent segment
            last_pt = sk.segments[i][-1]
            temp_parent = sk.parent_nodes(last_pt)
            parent_seg_skel.append(sk.segment_map[temp_parent])
            skel_to_axon_map[i] = axon_idx
            axon_idx += 1

    for i in range(len(parent_seg_skel)):
        if parent_seg_skel[i] not in skel_to_axon_map:
            segs_info['parent'].append(-1) #if parent not in axon segments, set to -1
        else:
            segs_info['parent'].append(skel_to_axon_map[parent_seg_skel[i]])
    # segs_info['parent'] = [skel_to_axon_map[p] for p in parent_seg_skel if p in skel_to_axon_map]


    for i in range(len(segs)):
        seg = segs[i]
        positions = np.array(seg['pt_position'])
        positions = positions * np.array([4, 4, 40]) / 1000 # Convert to microns
        myelin_flags = np.array(seg['myelin'])
        path_length = 0
        myelin_length = 0
        delay = 0
        delay_unmyel = 0

        #get path length of segment
        for j in range(len(positions) - 1):
            p1 = positions[j]
            p2 = positions[j + 1]
            length = np.linalg.norm(p2 - p1)
            #if myelinated, add to myelin length
            if myelin_flags[j] == 1 and myelin_flags[j + 1] == 1:
                myelin_length += length
            path_length += length
        #add length of last node to branch point to path length
        parent_seg = segs_info['parent'][i]
        if parent_seg != -1:
            #get position of branch point
            p1 = positions[-1]
            p2 = segs[parent_seg]['pt_position'][0] * np.array([4, 4, 40]) / 1000 # Convert to microns
            length = np.linalg.norm(p2 - p1)
            path_length += length
            if myelin_flags[-1] == 1 and segs[parent_seg]['myelin'][0] == 1: #but since branch point, this should not be reached.
                myelin_length += length

        
        #add to dict
        segs_info['path_length'].append(path_length)
        segs_info['myelin_length'].append(myelin_length)
        #get delay of segment
        radius = segs_info['radius'][i]

        delay = get_delay_from_l_and_r(path_length-myelin_length, radius, 0) + get_delay_from_l_and_r(myelin_length, radius, 1)
        delay_unmyel = get_delay_from_l_and_r(path_length, radius, 0)
        segs_info['delay'].append(delay)
        segs_info['delay_unmyel'].append(delay_unmyel)

    return segs_info


def create_synapse_len_delay_df(segs_myelin, segs_info, output_syn_df, cell_type_df):

    segs = segs_myelin

    synapses_len_delay_df = pd.DataFrame(columns=[
        'pre_pt_position', 'pre_pt_root_id', 'post_pt_root_id', 
        'post_cell_type', 'post_excit_inhib', 'size',
        'path_len_from_soma', 'vector_from_soma',
        'delay', 'delay_unmyelin'
    ])
    
    #iterate through the synapses for the neuron.
    for index, row in output_syn_df.iterrows():
        # print(row['pre_pt_position'])
        #Get basic attributes
        syn_position = row['pre_pt_position']
        syn_position = np.array(syn_position) * np.array([4, 4, 40]) / 1000 # Convert to microns

        pre_pt_root_id = row['pre_pt_root_id']
        post_pt_root_id = row['post_pt_root_id']
        if len(cell_type_df[cell_type_df['pt_root_id'] == post_pt_root_id]) == 0:
            post_cell_type = 'unknown'
            post_excit_inhib = 'unknown'
        else:
            post_cell_type = cell_type_df[cell_type_df['pt_root_id'] == post_pt_root_id]['cell_type'].values[0]
            post_excit_inhib = cell_type_df[cell_type_df['pt_root_id'] == post_pt_root_id]['classification_system'].values[0]
        syn_size = row['size']

        #find index in segs that syn_position is closest to.
        closest_seg_idx = -1
        closest_pt_in_seg_idx = -1
        closest_dist = float('inf')
        for i in range(len(segs)):
            seg = segs[i]
            positions = np.array(seg['pt_position']) * np.array([4, 4, 40]) / 1000
            dists = np.linalg.norm(positions - syn_position, axis=1)
            #get min_dist and idx_min
            min_dist = np.min(dists)
            idx_min = np.argmin(dists)
            # min_dist = np.min(dists)
            if min_dist < closest_dist:
                closest_dist = min_dist
                closest_seg_idx = i
                closest_pt_in_seg_idx = idx_min
        

        #get length and delay from synapse to soma.

        delay = 0
        delay_unmyel = 0

        #***** get path_length and myelin_length, delays for the portion of segment that synapse is on. *****
        #find for this segment.
        curr_seg = closest_seg_idx
        path_length = 0
        myelin_length = 0
        for j in range(closest_pt_in_seg_idx, len(segs[curr_seg]['pt_position']) - 1):
            p1 = segs[curr_seg]['pt_position'][j] * np.array([4, 4, 40]) / 1000
            p2 = segs[curr_seg]['pt_position'][j + 1] * np.array([4, 4, 40]) / 1000
            length = np.linalg.norm(p2 - p1)
            path_length += length
            if segs[curr_seg]['myelin'][j] == 1 and segs[curr_seg]['myelin'][j+1] == 1:
                myelin_length += length
        #Now add lengths to branch point.
        parent_seg = segs_info['parent'][curr_seg]
        if parent_seg != -1:
            #get position of branch point
            p1 = segs[curr_seg]['pt_position'][-1] * np.array([4, 4, 40]) / 1000
            p2 = segs[parent_seg]['pt_position'][0] * np.array([4, 4, 40]) / 1000 # Convert to microns
            length = np.linalg.norm(p2 - p1)
            path_length += length
            if segs[parent_seg]['myelin'][-1] == 1 and segs[parent_seg]['myelin'][0] == 1: #but since branch point, this should not be reached.
                myelin_length += length
        radius = segs_info['radius'][curr_seg]
        # if path_length - myelin_length < 0:
        #     print("Warning: path_length - myelin_length < 0")
        delay += get_delay_from_l_and_r(path_length-myelin_length, radius, 0) + get_delay_from_l_and_r(myelin_length, radius, 1)
        delay_unmyel += get_delay_from_l_and_r(path_length, radius, 0)
        # path_leng_from_soma += path_length

        #(GET-AROUND) some unclean dendrites may have "synapses" from incorrect merges. This handles a small number of those in a hacky way.
        if parent_seg == -1:
            #this would imply that the synapse is on the soma.
            continue

        #**** Get path length, delays for all parent segments up to soma. *****
        while parent_seg != -1:
            curr_seg = parent_seg
            path_length += segs_info['path_length'][curr_seg]
            delay += segs_info['delay'][curr_seg]
            delay_unmyel += segs_info['delay_unmyel'][curr_seg]
            parent_seg = segs_info['parent'][curr_seg]
            if parent_seg == -1:
                root_pt_position = segs[curr_seg]['pt_position'][0] * np.array([4, 4, 40]) / 1000
                vector_from_soma = syn_position - root_pt_position
        
        #add to dataframe
        synapses_len_delay_df = pd.concat([
            synapses_len_delay_df,
            pd.DataFrame([{
                'pre_pt_position': syn_position,
                'pre_pt_root_id': pre_pt_root_id,
                'post_pt_root_id': post_pt_root_id,
                'post_cell_type': post_cell_type,
                'post_excit_inhib': post_excit_inhib,
                'size': syn_size,
                'path_len_from_soma': path_length,
                'vector_from_soma': vector_from_soma,
                'delay': delay,
                'delay_unmyelin': delay_unmyel
            }])
        ], ignore_index=True)

    return synapses_len_delay_df

def create_synapse_len_delay_df_all_neurons(pt_root_ids, client, cell_type_df, directory_segs_myelin, directory_synapse_len_delay, overwrite = False):
    '''
    get cell_type_df.
    if not exists, create directory with name: directory_synapse_len_delay.
    for each pt_root_id,
        if pt_root_id.pkl exists in directory_synapse_len_delay and overwrite = False, skip
        else:
        get segs_myelin, (denoise it)
        get skeleton dict,
        get segs_info,
        get output_syn_df,

        then call create_synapse_len_delay_df
        save synapses_len_delay_df to directory_synapse_len_delay with filename pt_root_id.pkl
    
    '''
    if not os.path.exists(directory_synapse_len_delay):
        os.makedirs(directory_synapse_len_delay)
    for pt_root_id in pt_root_ids:
        print(f"Processing neuron {pt_root_id}")
        if os.path.exists(os.path.join(directory_synapse_len_delay, f"{pt_root_id}.pkl")) and not overwrite:
            print(f"File {pt_root_id}.pkl already exists. Skipping...")
            continue
        #get segs_myelin
        with open(os.path.join(directory_segs_myelin, f"{pt_root_id}.pkl"), 'rb') as f:
            segs_myelin = pickle.load(f)
        segs_myelin = denoise_segs(segs_myelin)
        #get skeleton dict
        sk_dict = client.skeleton.get_skeleton(pt_root_id, output_format='dict')        #get segs_info
        segs_info = get_info_about_each_segment(sk_dict, segs_myelin)
        #get output_syn_df
        output_syn_df = client.materialize.synapse_query(pre_ids=pt_root_id)
        print(f"Number of synapses for {pt_root_id}: ", len(output_syn_df))

        #call create_synapse_len_delay_df
        synapses_len_delay_df = create_synapse_len_delay_df(segs_myelin, segs_info, output_syn_df, cell_type_df)
        #save synapses_len_delay_df to directory_synapse_len_delay with filename pt_root_id.pkl
        with open(os.path.join(directory_synapse_len_delay, f"{pt_root_id}.pkl"), 'wb') as f:
            pickle.dump(synapses_len_delay_df, f)


