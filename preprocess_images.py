
import numpy as np
from skimage.measure import block_reduce
from scipy.stats import mode
from scipy.ndimage import map_coordinates
from skimage.measure import find_contours
from scipy.ndimage import gaussian_filter1d
from matplotlib.path import Path
import pandas as pd



def pick_normal_plane(ctr, sk_df):
    """Picks the plane perpendicular to the axon direction at a given point.
    Gets closest segment

    Parameters
    ----------
    ctr : array-like
        x,y,z coordinates of some point along axon. In voxel coordinates.
    sk_df : pandas.DataFrame
        DataFrame containing the skeleton data with columns 'x', 'y', 'z', and 'parent'.
        The coordinates should be in microns (should be default when using client.skeleton.get_skeleton)

    Returns
    -------
    plane : str
        The plane that is perpendicular to the axon direction at that point.
        Possible values are 'xy', 'xz', or 'yz'.
    """
    ctr = np.array(ctr) * np.array([4, 4, 40]) // 1000 # convert ctr to micron coordinates
    planes = ['yz', 'xz', 'xy']

    # Step 1: Find closest point in sk_df to ctr
    coords = sk_df[['x', 'y', 'z']].values
    dists = np.linalg.norm(coords - ctr, axis=1)
    closest_idx = np.argmin(dists)

    # Step 2: Get parent and children of that point
    connected = []

    # Parent segment
    parent = sk_df.loc[closest_idx, 'parent']
    if pd.notna(parent) and int(parent) >= 0:
        A = sk_df.loc[int(parent), ['x', 'y', 'z']].values
        B = coords[closest_idx]
        connected.append((A, B))

    # Child segments
    children = sk_df.index[sk_df['parent'] == closest_idx].tolist()
    for child_idx in children:
        A = coords[closest_idx]
        B = sk_df.loc[child_idx, ['x', 'y', 'z']].values
        connected.append((A, B))

    # Step 3: Find the closest segment among connected ones
    def closest_segment_dir(A, B):
        AB = B - A
        t = np.clip(np.dot(ctr - A, AB) / np.dot(AB, AB), 0, 1)
        proj = A + t * AB
        return B - A, np.linalg.norm(ctr - proj)

    direction, _ = min((closest_segment_dir(A, B) for A, B in connected), key=lambda x: x[1])

    # Step 4: Get plane most perpendicular to the direction

    #If 2 nodes are the same or close to each other, might need this.
    # if np.allclose(direction, 0):
    #     return None

    main_axis = np.argmax(np.abs(direction))
    return planes[main_axis]

def pull_image_and_segmentation(img_client, ctr, pt_root_id, plane, box_sz_microns):
    """Pulls the image and segmentation for a given point on an axon, and a chosen plane.
       Resamples the segmentation and image to 40nm isotropic resolution.

    Parameters
    ----------
    img_client : ImageryClient
        The client to use for pulling images and segmentations.
        Should be an instance of the ImageryClient class from the ic module.
    ctr : array-like
        x,y,z coordinates of some point along axon.
    pt_root_id : int
        The root ID of the neuron/cell.
    plane : str
        The plane to pull the image and segmentation for. "xy", "xz", or "yz".
    box_sz_microns : float
        The size of the box in microns.

    Returns
    -------
    image : np.ndarray
        The image data.
    segs : np.ndarray
        The segmentation mask.
    """
    
    # Pull the image and segmentation data
    x_size = box_sz_microns * 1000 / 4
    y_size = box_sz_microns * 1000 / 4
    z_size = box_sz_microns * 1000 / 40

    if plane == 'xz':
        y_size = 2
    elif plane == 'xy':
        z_size = 2
    elif plane == 'yz':
        x_size = 2

    bounds = np.array([
        [ctr[0] - x_size // 2, ctr[1] - y_size //2, ctr[2] - z_size // 2],  # Lower bound
        [ctr[0] + x_size // 2, ctr[1] + y_size //2, ctr[2] + z_size // 2]  # Upper bound
    ])

    image, segs = img_client.image_and_segmentation_cutout(
        bounds, 
        image_mip = 0,
        root_ids = pt_root_id,
        split_segmentations=False,
        )

    #resamples segmentation and image to 40nm isotropic (segmentation resampled using mode)
    if plane == 'xz':
        # segs = segs[::5,:]
        segs = mode(segs[:segs.shape[0]//5*5, :].reshape(-1, 5, segs.shape[1]), axis=1)[0].squeeze()
        image = block_reduce(image, block_size=(5, 1), func=np.mean)
    elif plane == 'xy':
        segs = segs[:,:,1]
        # segs = segs[::5,::5]
        segs = mode(segs[:segs.shape[0]//5*5, :segs.shape[1]//5*5]
                .reshape(segs.shape[0]//5, 5, segs.shape[1]//5, 5)
                .swapaxes(1,2).reshape(-1, 25), axis=1)[0].reshape(segs.shape[0]//5, segs.shape[1]//5)
        image = block_reduce(image[:,:,1], block_size=(5, 5), func=np.mean)
    elif plane == 'yz':
        # segs = segs[::5,:]
        segs = mode(segs[:segs.shape[0]//5*5, :].reshape(-1, 5, segs.shape[1]), axis=1)[0].squeeze()
        image = block_reduce(image, block_size=(5, 1), func=np.mean)

    return image, segs

def unwrap_image_along_boundary(image, segs, sigma = 3, n_in = 10, n_out = 30, step_size = 0.5):
    """Does normal based unwrapping of the image along boundary of segmentation (cell membrane).

    Parameters
    ----------
    image : np.ndarray
        The image data to unwrap.
    segs : np.ndarray
        The segmentation mask to use for finding the boundary.
    sigma : float, optional
        The standard deviation for Gaussian smoothing of the contour, by default 3.
    n_in : int, optional
        Number of samples to take inward from the contour, by default 10.
    n_out : int, optional
        Number of samples to take outward from the contour, by default 30.
    step_size : float, optional
        The step size in pixels for sampling along the normal, by default 0.5.

    Returns
    -------
    image_unwr : np.ndarray
        The unwrapped image data, (shape will vary based on contour length).
    contour_sm : np.ndarray
        The smoothed contour coordinates used for unwrapping.
    """

    # Get contour of segmentation
    sigma = 3
    mask = segs > 1
    contours = find_contours(mask.astype(float), level=0.5)

    # select the contour that contains the center point
    # contour = next(c for c in contours if Path(c).contains_point((segs.shape[0] // 2, segs.shape[1] // 2)))

    #instead, select contour closest to the center point:
    contour = min(contours, key=lambda c: np.min(np.sum((c - [segs.shape[0] // 2, segs.shape[1] // 2])**2, axis=1)))


    #smooth contour.
    smoothed_x = gaussian_filter1d(contour[:, 1], sigma)
    smoothed_y = gaussian_filter1d(contour[:, 0], sigma)
    contour_sm = np.stack([smoothed_y, smoothed_x], axis=1)

    #Do normal based unwrapping to straighten the contour  
    image_unwr = np.zeros((n_in + n_out + 1, len(contour_sm)))

    # Loop through contour
    for i, (y, x) in enumerate(contour_sm):

        # Compute tangent: difference between neighboring points
        prev_pt = contour_sm[i - 1]
        next_pt = contour_sm[(i + 1) % len(contour_sm)]
        tangent = next_pt - prev_pt

        # Normalize tangent
        tangent = tangent / np.linalg.norm(tangent)

        # Normal is perpendicular to tangent: swap and negate
        normal = np.array([-tangent[1], tangent[0]])

        # Sample along normal
        samples = []
        for j in range(-n_in, n_out + 1):
            offset = normal * j * step_size
            sample_y = y + offset[0]
            sample_x = x + offset[1]

            # Use interpolation to get subpixel intensity
            val = map_coordinates(image, [[sample_y], [sample_x]], order=2, mode='reflect')[0]
            samples.append(val)

        image_unwr[:, i] = samples
    return image_unwr, contour_sm


def unwrap_image_given_contour(image, contour_sm, n_in = 10, n_out = 30, step_size = 0.5):
    """ Helper function if you already have the contour extracted and may want to change params of unwrap.
    Does normal based unwrapping of the image along boundary given contour.

    Parameters
    ----------
    image : np.ndarray
        The image data to unwrap.
    contour : np.ndarray
        The contour around the segmentation. 
    n_in : int, optional
        Number of samples to take inward from the contour, by default 10.
    n_out : int, optional
        Number of samples to take outward from the contour, by default 30.
    step_size : float, optional
        The step size in pixels for sampling along the normal, by default 0.5.

    Returns
    -------
    image_unwr : np.ndarray
        The unwrapped image data, (shape will vary based on contour length).
    """

    #Do normal based unwrapping to straighten the contour  
    image_unwr = np.zeros((n_in + n_out + 1, len(contour_sm)))

    # Loop through contour
    for i, (y, x) in enumerate(contour_sm):

        # Compute tangent: difference between neighboring points
        prev_pt = contour_sm[i - 1]
        next_pt = contour_sm[(i + 1) % len(contour_sm)]
        tangent = next_pt - prev_pt

        # Normalize tangent
        tangent = tangent / np.linalg.norm(tangent)

        # Normal is perpendicular to tangent: swap and negate
        normal = np.array([-tangent[1], tangent[0]])

        # Sample along normal
        samples = []
        for j in range(-n_in, n_out + 1):
            offset = normal * j * step_size
            sample_y = y + offset[0]
            sample_x = x + offset[1]

            # Use interpolation to get subpixel intensity
            val = map_coordinates(image, [[sample_y], [sample_x]], order=2, mode='reflect')[0]
            samples.append(val)

        image_unwr[:, i] = samples
    return image_unwr