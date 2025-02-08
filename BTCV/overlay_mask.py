import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import os

# Function to load the image and label
def load_image_and_label(image_path, label_path):
    # Load image and label
    image = sitk.ReadImage(image_path)
    label = sitk.ReadImage(label_path)
    return image, label

# Function to normalize the image to the range [0, 255] for RGB plotting
def normalize_image(image):
    # Convert the image to a numpy array
    image_np = sitk.GetArrayFromImage(image)
    
    # Normalize the image: Min-Max scaling to [0, 255]
    min_val = np.min(image_np)
    max_val = np.max(image_np)
    
    # Avoid division by zero if the image has constant values
    if max_val > min_val:
        normalized_image = ((image_np - min_val) / (max_val - min_val)) * 255
    else:
        normalized_image = np.zeros_like(image_np)
    
    # Convert to uint8 for proper visualization
    normalized_image = normalized_image.astype(np.uint8)
    
    return normalized_image

# Function to extract a slice along the 147 axis (depth), keeping 512x512 spatial dimensions
def extract_foreground_slice(image, label, z_slice=None):
    # Convert the SimpleITK images to numpy arrays
    image_np = sitk.GetArrayFromImage(image)
    label_np = sitk.GetArrayFromImage(label)
    
    # Check for a specified slice, if not, default to the middle slice
    if z_slice is None:
        z_slice = image_np.shape[0] // 2  # Middle slice along the depth axis
    
    # Rearrange the array to treat the depth as a "slice" (transpose the first axis)
    image_np = np.transpose(image_np, (1, 2, 0))  # Transpose to (height, width, depth)
    label_np = np.transpose(label_np, (1, 2, 0))  # Transpose to (height, width, depth)
    
    # Extract the slice along the new depth axis (after transposing)
    image_slice = image_np[:, :, z_slice]
    label_slice = label_np[:, :, z_slice]
    
    return image_slice, label_slice

# Function to overlay the segmentation mask with unique colors for each label
def overlay_mask_on_slice(image_slice, label_slice):
    # Define a colormap for the overlay (use distinct colors for each class)
    cmap = plt.get_cmap('tab20c')  # For up to 20 classes, you can use 'tab20c'

    # Initialize an empty RGB image with the same height and width as the image slice
    overlay = np.zeros((image_slice.shape[1], image_slice.shape[2], 3), dtype=np.float32)

    # Iterate over the labels (skip background which is 0)
    for label_value in np.unique(label_slice):
        if label_value == 0:
            continue  # Skip background label
        
        # Generate a mask where label_value occurs in the label_slice
        mask = label_slice == label_value
        
        # Get the color corresponding to this label from the colormap
        color = np.array(cmap(label_value / float(np.max(label_slice))))[:3]  # Normalize label index to [0, 1]
        
        # Apply the color to the overlay image where the mask is True
        overlay[mask] = color
    
    return overlay

# Function to display the image with the mask overlayed
def display_image_with_overlay(image_slice, overlay):
    # Display the image slice in grayscale
    plt.imshow(image_slice, cmap='gray')
    
    # Overlay the segmentation mask with some transparency
    plt.imshow(overlay, alpha=0.5)  # The alpha controls the transparency of the overlay
    
    plt.show()

# Main function to perform all tasks
def main(image_path, label_path):
    # Load image and label
    image, label = load_image_and_label(image_path, label_path)
    
    # Normalize the image to [0, 255] range
    image_slice = normalize_image(image)
    
    # Extract a slice containing foreground content (with the rearranged dimensions)
    _, label_slice = extract_foreground_slice(image, label)
    
    # Overlay the segmentation mask on the foreground slice
    overlay = overlay_mask_on_slice(image_slice, label_slice)
    
    # Display the result
    display_image_with_overlay(image_slice, overlay)

# Example usage
root_path = "E:/Brats23 Data/BTCV dataset/RawData/RawData/Training"
images = os.path.join(root_path, "img")  # Replace with actual path
labels = os.path.join(root_path, "label")
image_name = "img0001.nii.gz"
label_name = "label0001.nii.gz"
image_path = os.path.join(images, image_name)
label_path = os.path.join(labels, label_name)
main(image_path, label_path)
