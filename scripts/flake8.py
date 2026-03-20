import os
import shutil
import subprocess
import sys

_IS_WINDOWS = sys.platform == "win32"


def check_flake8_errors(base_dir, filepath):
    if shutil.which("flake8") is None:
        return -1
    flak8_cmd = ["flake8"]  # '--quiet'

    if os.path.isdir(filepath):
        for root, _dirs, files in os.walk(filepath):
            for file in files:
                if file.endswith(".py"):
                    flak8_cmd.append(os.path.join(root, file))
    elif os.path.isfile(filepath):
        flak8_cmd.append(filepath)

    # Check code style.
    ret_flak8 = subprocess.call(flak8_cmd, cwd=base_dir)
    print("status code: ", ret_flak8)

    return ret_flak8


if __name__ == "__main__":
    if _IS_WINDOWS:
        print("skip flake8 check for Windows")
        sys.exit(0)

    base_dir = os.path.abspath(
        os.path.dirname(os.path.join(os.path.abspath(__file__), "../../"))
    )
    setupfile = os.path.join(base_dir, "setup.py")
    base_pydir = os.path.join(base_dir, "src/torch_comp")
    base_scripts = os.path.join(base_dir, "scripts")
    base_test = os.path.join(base_dir, "test")

    Check_dir = [setupfile, base_pydir, base_scripts, base_test]
    ret = sum([check_flake8_errors(base_dir, path) for path in Check_dir])
    if ret > 0:
        print("ERROR: flake8 found format errors!")
        sys.exit(1)
    elif ret < 0:
        print("WARNING: Please check format!")
    else:
        print("Pass!")
