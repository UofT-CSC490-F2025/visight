import pandas as pd          
from pipelines.configs.config import Config
from pathlib import Path
import hashlib
import re
import argparse
from pipelines.clients.s3_client import s3Client
from tqdm import tqdm
    

CONFIG = Config()
ACTIVE_PREFIX = None
ACTIVE_VERSION = None

def get_prefix(version: str) -> str:
    if version == "raw":
        return Config.raw_prefix
    elif version == "v1":
        return Config.processed_prefix
    elif version == "aug":
        return Config.aug_prefix
    else:
        raise ValueError(f"Unknown version {version}")

def create_brand_catalogue(metadata):
    '''
    Create df of brand names with ids
    
    NOTE: Id is currently a monotonically increasing value, 
    but might alter or enhance approach depending on potential inclusion of different datasets in the future
    '''
    expected_keys = {"names", "nc"}
    missing_keys = expected_keys - set(list(metadata.keys()))
    if missing_keys:
        raise Exception(f"Missing keys from metadata: {missing_keys}")
    
    brand_names = metadata["names"]
    brand_df = pd.DataFrame({"id": [i for i in range(metadata["nc"])], "name": brand_names})
    return brand_df

def get_classes_in_img(client, label_directory, image_stem):
    '''
    Finds and formats class_ids from a label file. 
    Raises an exception if the matching label file is not found in expected s3 location
    '''
    label_key = f"{label_directory}/{str(image_stem)}.txt"
    try:
        label_file = client.get_object(label_key, "txt")
    except Exception as e:
        raise Exception(f"Error fetching file {label_key} from s3: {e}")

    if not label_file or len(label_file.strip()) == 0:
        return label_key, []

    class_ids = set()
    for raw_line in label_file.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            continue
        cls_token = parts[0]
        try:
            cls_id = int(float(cls_token))
        except (ValueError, TypeError):
            continue
        class_ids.add(cls_id)

    classes = sorted(class_ids)
    return label_key, classes

def create_image_catalogue(client, split_directories, images_per_split, base_prefix: str):
    '''
    Input: 
    - client (s3 client)
    - split_directories: a dictionary that maps data splits to their label directories
    - images_per_split: a dictionary that maps data splits to a list of all images
    
    Take all data from existing dataset splits and save each image/label pair into the image catalogue
    An exception is thrown and catalogue doesn't save if a matching label cannot be found for an image. 
    (The same exception is not thrown if there doesn't exist a matching image to a given label file.)
    
    Assumptions: 
    (1) Image and label pairs are uniquely identifiable by their file name (the image id is a hash of this name)
    ex: 
    0_jpeg.rf.7f31cb6315d6f40d5245c8358e01f620.jpg
    0_jpeg.rf.7f31cb6315d6f40d5245c8358e01f620.txt
    (2) Provided labels are already in Yolo format
    
    NOTE in the future might want to add dataset versioning and naming to config or as a commandline input
    NOTE might want to treat image files that don't have a matching label as a non-breaking warning
    
    Image catalogue: csv with columns
    - id (str): hash of image_path stem (aka without the .jpg or .png ending, irregardless of parent directory)
    - s3_img_path (str): image location in s3
    - s3_label_path (str): matching YOLO formatted label location in s3
    - class_ids (array of ints): a deduplicated and ordered list of class ids present in the image
    - split (str): one of ["train", "val", "test"]
    '''

    label_directories = {k: f"{base_prefix}/{v}/labels" for k, v in split_directories.items()}
    catalogue_dict = {
        "id": [], "s3_img_path":[], "s3_label_path":[], "class_ids":[], "split":[]
    }
    for split in images_per_split.keys(): 
        # for each image, confirm that a matching label exists and write it into
        label_directory = label_directories[split]
        print(f"\t{split} with {len(images_per_split[split])} images..")
        for full_img_path in tqdm(images_per_split[split]):
            # get the image stem (without directories or file type)
            rel_path_stem = Path(full_img_path).stem
            hashed_id = hashlib.sha256(str(rel_path_stem).encode()).hexdigest()
            
            # 1: check if a matching label exists, open said file, get classes from it
            full_label_path, classes = get_classes_in_img(client, label_directory, rel_path_stem)
            
            catalogue_dict["id"].append(hashed_id)
            catalogue_dict["s3_img_path"].append(full_img_path)
            catalogue_dict["s3_label_path"].append(full_label_path)
            catalogue_dict["class_ids"].append(classes)
            catalogue_dict["split"].append(split)
                
    image_catalogue = pd.DataFrame(catalogue_dict)
    return image_catalogue

def get_and_format_split_directories(metadata, expected_splits):
    # NOTE: this is a non-clean way to normalize the paths received in data.yaml. 
    # The data.yaml we have has a relative path that goes up a directory to access the datasets, which we might want to change.  
    strip_relative_pathing = lambda s: re.sub(r'/images$', "", re.sub(r'^(\.)*\/', "", s))
    split_directories = {k:f"{strip_relative_pathing(metadata[k])}" 
                         for k in expected_splits if k in metadata.keys()}
    missing_splits = set(expected_splits) - metadata.keys()
    if missing_splits: 
        print(f"WARNING: the following data splits do not exist in metadata: \n{missing_splits}")
        
    return split_directories
    
def extract(client, base_prefix: str, expected_splits=["train", "val", "test"]): 
    metadata_key = f"{base_prefix}/data.yaml"
    
    metadata = client.get_object(metadata_key, "yaml")
    # Process some of the metadata to extract all other files needed
    split_dirs = get_and_format_split_directories(metadata, expected_splits)
    
    image_files = {k:client.batch_get_filenames(f"{base_prefix}/{v}/images") for k, v in split_dirs.items()}
    
    return {
        "metadata": metadata, 
        "split_directory": split_dirs,
        "image_files": {k:v for k,v in image_files.items() if len(v) > 0} 
    }

def transform(client, retrieved_data, base_prefix: str): 
    expected_keys = {"metadata", "split_directory", "image_files"}
    missing_keys = expected_keys - set(list(retrieved_data.keys()))
    if missing_keys:
        raise Exception(f"Missing keys from Extract step: {missing_keys}")
        
    brand_df = create_brand_catalogue(retrieved_data["metadata"])
    print("Created brand catalogue")
    
    print("Creating image and label catalogue..")
    image_df = create_image_catalogue(client, retrieved_data["split_directory"], retrieved_data["image_files"], base_prefix)
    print("Created image catalogue")
    
    return {
        "brand_catalogue": brand_df, 
        "image_catalogue": image_df
    }

def load(result_data, version: str): 
    expected_keys = {"brand_catalogue", "image_catalogue"}
    missing_keys = expected_keys - set(list(result_data.keys()))
    if missing_keys:
        raise Exception(f"Missing keys from Transform step: {missing_keys}")
    
    out_prefix = f"{CONFIG.catalogue_path}/{version}"
    result_data["brand_catalogue"].to_csv(f"{out_prefix}/brand_catalogue.csv", index=False)
    result_data["image_catalogue"].to_csv(f"{out_prefix}/image_catalogue.csv", index=False)

    print(f"Successfully saved data to s3 directory {out_prefix}!")

if __name__ == "__main__": 

    parser = argparse.ArgumentParser()
    parser.add_argument("--version", choices=["raw", "v1", "aug"], required=True)
    args = parser.parse_args()

    ACTIVE_VERSION = args.version
    ACTIVE_PREFIX = get_prefix(args.version)
    print(f"Running ingestion for s3://{Config.s3_bucket}/{ACTIVE_PREFIX}")

    s3 = s3Client(buckets=[CONFIG.s3_bucket])
    
    # Extract step
    retrieved_data = extract(s3, ACTIVE_PREFIX)
    
    # Transform step
    result_data = transform(s3, retrieved_data, ACTIVE_PREFIX)
    
    # Load (aka save) step
    load(result_data, ACTIVE_VERSION)
