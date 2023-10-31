import os
import sys

parent_dir = os.path.abspath(__file__ + 3 * "/..")
sys.path.insert(0, parent_dir)

from bye_splits.utils.job_helpers import Arguments

if __name__ == "__main__":
    script = "/home/llr/cms/ehle/NewRepos/bye_splits/tests/submission/dummy_submit.py"

    arg_dict = {"--float_arg": 0.5, "--str_arg": "woop", "--flag": False}

    arg_object = Arguments(script=script)

    arg_object.run_script(arg_dict)