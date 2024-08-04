import os
import shutil
from PIL import Image

def move_rename_and_resize_images(main_folder, total_files, target_size=(320,180)):
    # Get a list of all subfolders and files
    all_files = []
    for root, dirs, files in os.walk(main_folder):
        for file in files:
            all_files.append(os.path.join(root, file))
    
    # Sort files to have a predictable order
    all_files.sort()
    
    # Move, rename, and resize images
    for i, file_path in enumerate(all_files):
        if i < total_files:
            # Get file extension
            _, ext = os.path.splitext(file_path)
            # Define new file name with sequential number
            new_file_name = f"{i + 1}{ext}"
            # Define new file path in main folder
            new_file_path = os.path.join(main_folder, new_file_name)
            
            # Move file to new location with new name
            shutil.move(file_path, new_file_path)
            
            # Open, resize, and save the image
            try:
                with Image.open(new_file_path) as img:
                    img = img.resize(target_size)
                    img.save(new_file_path)
            except Exception as e:
                print(f"Error processing {new_file_path}: {e}")
        else:
            # Delete the remaining files
            os.remove(file_path)

# Example usage
main_folder = 'val'
total_files = 2900  # Number of files you want to keep
move_rename_and_resize_images(main_folder, total_files)
