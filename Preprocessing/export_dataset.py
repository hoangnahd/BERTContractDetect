import os
import json
from tqdm import tqdm
import timeout_decorator

from contract_clean_and_format import SourceCodeCleanerAndFormatter  # Assuming this import is correct

@timeout_decorator.timeout(7)
def preprocessData(inputFile):
    contract = SourceCodeCleanerAndFormatter(inputFile)
    contract.read_input_file()
    contract.clean_source_code()
    contract.format_source_code()
    return contract.source_code

def loadFilesFolder(inputFolder, isVul, update=False, file_to_update=None):
    filesList = []

    if update and file_to_update:
        try:
            with open(file_to_update, 'r') as f:
                filesList = json.load(f)
        except Exception as e:
            print(f"Failed to update file list: {str(e)}")
    for file in tqdm(os.listdir(inputFolder)):
        contract_info = {}
        file_path = os.path.join(inputFolder, file)
        contract_info["source"] = preprocessData(file_path)
        contract_info["label"] = isVul
        contract_info["name"] = file
        filesList.append(contract_info)

    return filesList

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Load files from folders.')
    parser.add_argument('inputFolder', type=str, help='Path to the input folder')
    parser.add_argument('--update', action='store_true', help='Update the file list')
    parser.add_argument('--isVul', action='store_true', help='Flag indicating if files are vulnerable')
    parser.add_argument('--o', type=str, default=None, help='Output file for processed data (optional)')
    parser.add_argument('--file', type=str, default=None, help='Name of file to update (required if --update is specified)')

    args = parser.parse_args()

    if args.update and not args.file:
        parser.error("--file is required when --update is specified.")

    filesList = loadFilesFolder(args.inputFolder, args.isVul, args.update, args.file)

    if args.o:
        try:
            with open(args.o, 'w') as f:
                json.dump(filesList, f, indent=4)
            print(f"Processed data saved to {args.o}")
        except Exception as e:
            print(f"Failed to save processed data: {str(e)}")
            
