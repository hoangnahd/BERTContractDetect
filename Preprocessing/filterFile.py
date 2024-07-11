import os
import glob

def remove_excess_files(folder_path, keep_count):
    # List all files in the folder
    files = glob.glob(os.path.join(folder_path, '*'))
    
    # Sort files by creation time (oldest first)
    files.sort(key=os.path.getmtime)
    
    # Calculate number of files to delete
    files_to_delete = len(files) - keep_count
    
    # Delete excess files
    for i in range(files_to_delete):
        try:
            os.remove(files[i])
            print(f"Deleted: {files[i]}")
        except Exception as e:
            print(f"Failed to delete: {files[i]}. Error: {e}")

# Example usage:
folder_path = 'data/vulnerable'
keep_count = 1000  # Number of latest files to keep

remove_excess_files(folder_path, keep_count)
