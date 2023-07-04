import sys, os
module_path = os.path.dirname(__file__)
project_path = os.path.dirname(os.path.dirname(module_path))
sys.path.extend([module_path, project_path])

import HyperParameters