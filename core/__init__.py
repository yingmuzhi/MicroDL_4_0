# ABS path 
import sys, os
module_path = os.path.dirname(__file__)
project_path = os.path.dirname(os.path.dirname(module_path))
sys.path.extend([module_path, project_path])

# version
VERSION = 1.0

import core.hyper_parameters as hyper_parameters, core.module as module, core.progress_board as progress_board