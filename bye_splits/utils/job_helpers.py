#!/usr/bin/env python

import os
import sys

parent_dir = os.path.abspath(__file__ + 5 * "../")
sys.path.insert(0, parent_dir)

import re
import argparse
import subprocess
import inspect
import importlib.util

_all_ = ['increment_version', 'grab_most_recent', 'compare_file_contents', 'write_file_version', 'conditional_write']

def increment_version(file_path):
    """Accepts a template for a full path to a file and increments the version"""
    dir, file = os.path.split(file_path)
    base, ext = os.path.splitext(file)
    i = 0
    file_path = "{}/{}_v{}{}".format(dir, base, i, ext)
    while os.path.exists(file_path):
        i += 1
        file_path = "{}/{}_v{}{}".format(dir, base, i, ext)
    return file_path

def grab_most_recent(file_path, return_all=False):
    """Grab the most recent version of the file corresponding to the template file_path (or return all matches).
    Returns None if no files mathing template <file_path> have been written."""
    dir, file = os.path.split(file_path)
    base, ext = os.path.splitext(file)
    files = os.listdir(dir)
    version_pattern = re.compile("{}_v(\\d+)\\{}".format(base, ext))
    matches = [version_pattern.search(file) for file in files]
    matches = [match for match in matches if not match is None]
    if len(matches) > 0:
        matches = [int(match.group(1)) for match in matches]
        most_recent = max(matches)
        if not return_all:
            file_path = dir + "/" + base + "_v" + str(most_recent) + ext
        else:
            file_path = [dir + "/" + base + "_v" + str(f) + ext for f in matches]
        return file_path
    else:
        return None

def compare_file_contents(file_path, buffer_list):
    """
    Compares the content in <file_path> with <buffer_list>,
    which should be a list of strings that you wish to write
    to a new file.
    """
    with open(file_path, "r") as file:
        contents = file.readlines()
    return contents==buffer_list

def write_file_version(template, version):
    file_name = increment_version(template)
    with open(file_name, "w") as job_file:
        job_file.writelines(version)
    st = os.stat(file_name)
    os.chmod(file_name, st.st_mode | 0o744)
    return file_name

def conditional_write(file_versions, file_template, current_version):
    """
    Loops through the files in <file_versions>, comparing their contents
    to the current version. If an identical version is found, the function
    breaks and does nothing. Otherwise, it will write the contents in
    <current_version> to an updated version number whose basename corresponds to
    <file_template>. If file_versions is None, writes the v0 version.
    """
    if file_versions is not None:
        identical_version = False
        for file in file_versions:
            if not compare_file_contents(file, current_version):
                continue
            else:
                identical_version = True
                file_path = file
                break
        if not identical_version:
            file_path = write_file_version(file_template, current_version)
    
    else:
        file_path = write_file_version(file_template, current_version)
    return file_path

class Arguments:

  def __init__(self, script):
    self.script       = script
    self.called_file  = inspect.stack()[1].filename
    if self.script != self.called_file:
      spec = importlib.util.spec_from_file_location("script", self.script)
      original_script = importlib.util.module_from_spec(spec)
      spec.loader.exec_module(original_script)
      self.accepted_args = original_script.arg_dict

  def _get_options(self):
    return subprocess.run(["python", self.script, "-h"], capture_output=True, text=True).stdout

  def get_running_args(self):
    """Returns dictionary of command-line
    argument key/value pairs."""
    running_args = sys.argv[1:]
    run_arg_dict = {}
    for i, arg in enumerate(running_args):
      if arg.startswith("--"):
        arg_name = arg
        if i + 1 < len(running_args):
          if running_args[i + 1].startswith("--"):
            run_arg_dict[arg_name] = True
          else:
            run_arg_dict[arg_name] = running_args[i + 1]
        else:
          run_arg_dict[arg_name] = True

    return run_arg_dict

  def combine_args(self, passed_args_dict):
    """Combines the passed argument dictionary
    with the command-line argument dictionary."""
    running_args = self.get_running_args()

    combined_args = {}
    for key in set(passed_args_dict.keys()).union(set(running_args.keys())):
      if key in running_args:
        combined_args[key] = running_args[key]
      else:
        combined_args[key] = passed_args_dict[key]
    
    return combined_args


  def verify_args(self, passed_args_dict):
    """Verifies that the arguments dictionary (passed+command_line)
    is a valid set of arguments for the script."""
    
    combined_args = self.combine_args(passed_args_dict)

    true_keys     = set(self.accepted_args.keys()).union({'"--$val"'})
    combined_keys = set(combined_args.keys())

    if combined_keys.issubset(true_keys):
      for arg, arg_info in self.accepted_args.items():
        if "required" in arg_info.keys() and arg not in \
        combined_keys and arg_info["required"] is True:
          raise Exception("Required argument not passed: {}".format(arg))
      return combined_args
    else:
      raise Exception("Passed arguments not in script: {}\n{}".format(
          combined_keys.difference(true_keys), self._get_options()))

  def write_comm(self, arg_dict):
    """Verifies arguments and writes python command."""
    full_args = self.verify_args(arg_dict)
    comm = ["python", self.script]
    for arg_name, arg_value in full_args.items():
      if "action" in self.accepted_args[arg_name] and \
      self.accepted_args[arg_name]["action"] == "store_true":
        if arg_value is True:
          comm.append(arg_name)
      else:
        comm.append(arg_name)
        comm.append(str(arg_value))
    return comm

  def add_args(self, arg_dict, description=None):
    """Adds the arguments to the script's argument list,
    and returns the updated argument list."""

    parser = argparse.ArgumentParser(description=description)
    for arg_name, arg_info in arg_dict.items():
        parser.add_argument(arg_name, **arg_info)

    self.args = vars(parser.parse_args())
    return self.args

  def run_script(self, arg_dict):
    comm = self.write_comm(arg_dict)
    subprocess.run(comm)