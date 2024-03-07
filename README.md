Satellite repository of https://github.com/PetrGlad/statmoments
and https://github.com/akochepasov/statmoments.

Used for GPU experimentation to reduce dependencies and avoid handling unnecessary details.

Using cupy to access CUDA functionality.
 
# Instructions

Assuming Linux environment.

Install CUDA toolkit (Libux/Debian)
```
apt install nvidia-cuda-toolkit
```

Also may help: nvidia-cuda-dev, nvidia-cuda-gdb


Get installed CUDA version
```
nvcc --version
```

Get CUDA capabilities supported by the hardware
```
nvidia-smi
```

Install cupy wheel (package name suffix depends on the installed CUDA version):
```
pip install cupy-cuda11x
```

Ensure the CUDA driver can be actually used
```python
import cupy
cupy.cuda.get_local_runtime_version()
cupy.cuda.device.Device().compute_capability
```

Reinstallation, library changes, or drive upgrade may cause CUDA to load. 
For example, you may get an "CUDA error: unknown error" or "Error: cudaErrorUnknown: unknown error".
In that case cleaning compiled kernel cache (and in some cases also rebooting the system) may help:
```
rm -r "$HOME/.nv"
```

# Links

* [Installing latest ncu profiler (the one from Ubuntu does not work)](https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters)
* [Allow using performance counters](https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters)
* [Installing latest CUDA toolkit on Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)


