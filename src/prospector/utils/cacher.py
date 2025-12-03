# src/prospector/utils/cacher.py

from .custom_logging import get_cache_dir
from .custom_logging import get_timestamp
import inspect
import hashlib
import os
import pickle
import pprint

def compute_hash(*args):
    hash_obj = hashlib.md5()
    for arg in args:
        hash_obj.update(repr(arg).encode())
    return hash_obj.hexdigest()

class Cacher:
    """
    This class is a time saver. It caches the results of a computation to disk
    based on the hash of the input arguments. If the cache exists, it will load
    the outputs instead of recomputing them. If the cache does not exist, it will
    save the outputs to disk for future use

    Typical usage:

    cacher = Cacher(computation_inputs)
    if cacher.exists():
        outputs = cacher.load()
    else:
        outputs = compute_outputs(computation_inputs)
        cacher.save(outputs)
    """
    def __init__(self, computation_inputs, tag=""):
        # Nicely print things
        self.printer = pprint.PrettyPrinter(indent=4, width=40)

        # Hash everything and construct the filepath from the hash
        self.computation_inputs = computation_inputs
        self.cache_hash = compute_hash(*computation_inputs)
        if tag:
            self.cache_filepath = f"{get_cache_dir()}/{tag}_{self.cache_hash}.pkl"
        else:
            self.cache_filepath = f"{get_cache_dir()}/cache_{self.cache_hash}.pkl"

    def exists(self):
        # The file both needs to exist and be non-empty
        return os.path.exists(self.cache_filepath) and os.path.getsize(self.cache_filepath) > 0
    
    def load(self):
        # Assumes that existence has been checked for
        print(f"[{self.__class__.__name__}] loading cached results from {self.cache_filepath}")
        with open(self.cache_filepath, "rb") as f:
            return pickle.load(f)
    
    def save(self, outputs_dict):
        print(f"[{self.__class__.__name__}] saving cached results to {self.cache_filepath}")
        # Ensure the cache directory exists
        os.makedirs(os.path.dirname(self.cache_filepath), exist_ok=True)
        with open(self.cache_filepath, "wb") as f:
            pickle.dump(outputs_dict, f)
        # Save as well the computation inputs to a text file bearing the same name
        inputs_txt_path = self.cache_filepath.replace(".pkl", "_computation_inputs.txt")
        with open(inputs_txt_path, "w") as f:
            # Write it nicely with the pretty printer
            f.write(self.printer.pformat(self.computation_inputs))