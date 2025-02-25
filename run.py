#!/usr/bin/env python3
import os
import sys
import argparse
import subprocess
import venv
import platform
from pathlib import Path

# Define paths
ROOT_DIR = Path(__file__).parent.absolute()
VENV_DIR = ROOT_DIR / "venvs"
REQUIREMENTS_DIR = ROOT_DIR / "requirements"

# Define module-specific requirements
MODULE_REQUIREMENTS = {
    "object": REQUIREMENTS_DIR / "object_requirements.txt",
    "object_vehicle": REQUIREMENTS_DIR / "object_vehicle_requirements.txt",
    "person": REQUIREMENTS_DIR / "person_requirements.txt",
    "scene": REQUIREMENTS_DIR / "scene_requirements.txt",
}

def setup_venv(module_name):
    """Create and set up a virtual environment for the specified module."""
    venv_path = VENV_DIR / module_name
    
    # Create venv directory if it doesn't exist
    VENV_DIR.mkdir(exist_ok=True)
    
    # Check if venv already exists
    if venv_path.exists():
        print(f"Virtual environment for {module_name} already exists.")
    else:
        print(f"Creating virtual environment for {module_name}...")
        venv.create(venv_path, with_pip=True)
    
    # Determine the path to the Python executable in the virtual environment
    if platform.system() == "Windows":
        python_path = venv_path / "Scripts" / "python.exe"
    else:
        python_path = venv_path / "bin" / "python"
    
    # Install requirements
    if module_name in MODULE_REQUIREMENTS:
        req_file = MODULE_REQUIREMENTS[module_name]
        if req_file.exists():
            print(f"Installing requirements for {module_name}...")
            subprocess.run([str(python_path), "-m", "pip", "install", "-r", str(req_file)])
        else:
            print(f"Requirements file {req_file} not found.")
    
    return python_path

def run_object_workflow(args):
    """Run object detection workflow."""
    python_path = setup_venv("object")
    
    if args.subworkflow == "tests":
        print("Running object detection tests...")
        subprocess.run([str(python_path), "-m", "pytest", "object/tests"])
    elif args.subworkflow == "detection":
        print("Running object detection on image...")
        subprocess.run([str(python_path), "object/img_object_detection.py"])
    elif args.subworkflow == "video":
        print("Running object detection on video...")
        subprocess.run([str(python_path), "object/video_object_detection.py"])
    elif args.subworkflow == "tracking":
        print("Running object tracking on video...")
        subprocess.run([str(python_path), "object/video_object_detection_with_tracking.py"])
    else:
        print("Invalid subworkflow for object detection.")

def run_vehicle_workflow(args):
    """Run vehicle classifier workflow."""
    python_path = setup_venv("object_vehicle")
    
    if args.subworkflow == "download_test":
        print("Downloading test data for vehicle classifier...")
        subprocess.run([str(python_path), "object/vehicle_classifier/download_test.py"])
    elif args.subworkflow == "annotations_test":
        print("Testing annotations for vehicle classifier...")
        subprocess.run([str(python_path), "object/vehicle_classifier/annotations_test.py"])
    elif args.subworkflow == "train":
        print("Training vehicle classifier...")
        subprocess.run([str(python_path), "object/vehicle_classifier/train.py"])
    elif args.subworkflow == "inference":
        print("Running vehicle classifier inference...")
        subprocess.run([str(python_path), "object/vehicle_classifier/inference_test.py"])
    else:
        print("Invalid subworkflow for vehicle classifier.")

def run_person_workflow(args):
    """Run person detection workflow."""
    python_path = setup_venv("person")
    
    if args.subworkflow == "tests":
        print("Running person detection tests...")
        subprocess.run([str(python_path), "-m", "pytest", "person/tests"])
    elif args.subworkflow == "download_model":
        print("Downloading person transformer model...")
        # This requires user input
        subprocess.run([str(python_path), "person/upload_download_models/download_person_transformer.py"], 
                      stdin=sys.stdin, stdout=sys.stdout)
    elif args.subworkflow == "everyid":
        print("Running EveryID person recognition...")
        subprocess.run([str(python_path), "person/EveryID_msmt17_top_rank.py"])
    else:
        print("Invalid subworkflow for person detection.")

def run_scene_workflow(args):
    """Run scene analysis workflow."""
    python_path = setup_venv("scene")
    
    if args.subworkflow == "tests":
        print("Running scene analysis tests...")
        subprocess.run([str(python_path), "-m", "pytest", "scene/tests"])
    elif args.subworkflow == "download_transformer":
        print("Downloading scene transformer model...")
        subprocess.run([str(python_path), "scene/upload_download_models/download_scene_transformer.py"])
    elif args.subworkflow == "inference_transformer":
        print("Running scene analysis inference with transformer model...")
        subprocess.run([str(python_path), "scene/inference_transformer.py"])
    else:
        print("Invalid subworkflow for scene analysis.")

def main():
    parser = argparse.ArgumentParser(description="Run computer vision workflows with isolated environments")
    
    subparsers = parser.add_subparsers(dest="module", help="Module to run")
    
    # Object detection parser
    object_parser = subparsers.add_parser("object", help="Object detection workflows")
    object_parser.add_argument("subworkflow", choices=["tests", "detection", "video", "tracking"], 
                              help="Subworkflow to run")
    
    # Vehicle classifier parser
    vehicle_parser = subparsers.add_parser("vehicle", help="Vehicle classifier workflows")
    vehicle_parser.add_argument("subworkflow", 
                               choices=["download_test", "annotations_test", "train", "inference"], 
                               help="Subworkflow to run")
    
    # Person detection parser
    person_parser = subparsers.add_parser("person", help="Person detection workflows")
    person_parser.add_argument("subworkflow", choices=["tests", "download_model", "everyid"], 
                              help="Subworkflow to run")
    
    # Scene analysis parser
    scene_parser = subparsers.add_parser("scene", help="Scene analysis workflows")
    scene_parser.add_argument("subworkflow", 
                             choices=["tests", "download_cnn", "download_transformer", "convert_model", "inference_cnn", "inference_transformer", "inference_onnx"], 
                             help="Subworkflow to run")
    
    args = parser.parse_args()
    
    # Create requirements directory if it doesn't exist
    REQUIREMENTS_DIR.mkdir(exist_ok=True)
    
    if args.module == "object":
        run_object_workflow(args)
    elif args.module == "vehicle":
        run_vehicle_workflow(args)
    elif args.module == "person":
        run_person_workflow(args)
    elif args.module == "scene":
        run_scene_workflow(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 