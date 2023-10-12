#!/usr/bin/env python

import re

import os
import sys

parent_dir = os.path.abspath(__file__ + 5 * "../")
sys.path.insert(0, parent_dir)

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
    """Grab the most recent version of the file corresponding to the template file_path (or return all matches)"""
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
        raise ValueError("There are no versions of the passed file: {}".format(file_path))

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
    <file_template>.
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