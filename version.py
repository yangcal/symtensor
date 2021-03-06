#!/usr/bin/env python
#
# Author: Yang Gao <younggao1994@gmail.com>
#

"""
This module imports the version number.
"""

import os

with open(os.path.join(os.path.dirname(__file__), 'VERSION')) as version_file:
    VERSION = version_file.read().strip()
