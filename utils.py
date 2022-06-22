"""
Author:      Cameron Johnson
Modified:    06/21/2022
Description: Utility methods.
"""

import subprocess

def runcmd(cmd, verbose=False):
    """
    Runs the provided terminal command.
    """
    process = subprocess.Popen(
        cmd,
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
        text = True,
        shell = True)
    std_out, std_err = process.communicate()
    if verbose:
        print(std_out.strip(), std_err)
    pass