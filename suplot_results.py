import os
import argparse
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random

def load_images_from_id_directory(id_directory):
    """
    Loads all the model images from a given ID directory and returns them as a dictionary.
    Each key will be the model name, and the value will be the image.
    """
    images = {}
    model_names = ['SwinUNETR', 'SegResNet', 'nnFormer', 'UXNet', '3D SegUX-Net', 'ground_truth']
    
    for model_name in model_names:
        img_path = os.path.join(id_directory, f"{model_name}.png")  # Assuming images are named with model names
        if os.path.exists(img_path):
            try:
                images[model_name] = mpimg.imread(img_path)  # Read the image
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
        else:
            print(f"Image for model {model_name} not found in {id_directory}")
    
    return images

def plot_comparison_subplot(id_folders, output_filename):
    """
    Creates a 3x6 subplot with each row corresponding to an ID folder,
    and columns representing model-generated images.
    """
    fig, axes = plt.subplots(3, 6, figsize=(9.6, 5.85), dpi=300)  # 3 rows, 6 columns
    plt.subplots_adjust(wspace=0.03, hspace=0.03)  # Minimized column space, added row separation

    # Define the models in the specified order
    ordered_models = ['SwinUNETR', 'SegResNet', 'nnFormer', 'UXNet', '3D SegUX-Net', 'ground_truth']

    for row, id_folder in enumerate(id_folders):
        images = load_images_from_id_directory(id_folder)

        for col, model in enumerate(ordered_models):
            if model in images:
                ax = axes[row, col]
                ax.imshow(images[model])
                ax.axis('off')  # Turn off axis labels for better appearance
                ax.margins(0, 0)  # Remove any margins between images

    # Add column labels for the models at the top
    for col, model in enumerate(ordered_models):
        axes[0, col].set_title(model, fontsize=10, fontweight='bold')

    # Adjust layout to ensure no white space
    plt.tight_layout()
    plt.savefig(output_filename, bbox_inches='tight', pad_inches=0.0)
    plt.close()

def process_directory(base_dir, output_dir):
    """
    Main function to process the 'brats2023' directory and its subfolders.
    """
    # random.seed(123)
    # Path to the ID folders (subfolders under brats2023)
    id_folders = [os.path.join(base_dir, id_folder) for id_folder in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, id_folder))]
    
    # Randomly shuffle and select 3 directories for comparison
    random.shuffle(id_folders)
    id_folders = id_folders[:3]  # Select only the first 3 ID directories after shuffling
    
    # Generate output filename from base directory name
    base_dir_name = os.path.basename(base_dir)
    output_filename = os.path.join(output_dir, f"{base_dir_name}_comparison_results.png")
    
    # Create the comparison subplot
    plot_comparison_subplot(id_folders, output_filename)
    print(f"Comparison subplot saved as {output_filename}")

if __name__ == '__main__':
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Generate comparison subplots for BRATS dataset models.")
    parser.add_argument('--directory', type=str, help="Directory containing the BRATS dataset directories (e.g., brats2023).")
    parser.add_argument('--output_directory', type=str, help="Directory where the comparison subplot will be saved.")
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)
    
    # Process the specified directory and save the plot
    process_directory(args.directory, args.output_directory)
