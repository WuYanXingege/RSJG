"""Rerun the full previous GPU suite into a NEW subfolder, including dense N57."""
import importlib.util
import common
common.setup(cuda=True)
common.DOC=common.DOC/'gpu_full_regression';common.DOC.mkdir(parents=True,exist_ok=True)
spec=importlib.util.spec_from_file_location('_unchanged_full_gpu_suite',common.SOURCE/'scripts/ssec_acceptance/gpu.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
module.main()
