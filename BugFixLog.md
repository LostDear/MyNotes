# C Environment

> [!bug]
>
> File "/opt/env/anaconda3/envs/py38/lib/python3.8/site-packages/torch/utils/cpp_extension.py", line 1853, in verify_ninja_availability
>     raise RuntimeError("Ninja is required to load C++ extensions")
> RuntimeError: Ninja is required to load C++ extensions

pip install ninja

**References**

[“Ninja is required to load C++ extensions”解决方案-CSDN博客](https://blog.csdn.net/qq_61888524/article/details/123830941)



> [!bug]
>
> subprocess.CalledProcessError: Command '['which', 'c++']' returned non-zero exit status 1.

sudo apt-get install build-essential



> [!bug]
>
> ImportError: /opt/env/anaconda3/envs/py38/bin/../lib/libstdc++.so.6: version `GLIBCXX_3.4.32' not found 

```bash
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt update
sudo apt install gcc-9
sudo apt install libstdc++6

ls -l /usr/lib/x86_64-linux-gnu/libstdc++.so* #查libc++版本
```

然后参考[知乎](https://zhuanlan.zhihu.com/p/685165815)



> [!bug]
>
>  CMake Error at CMakeLists.txt:36 (pybind11_add_module):
>         Unknown CMake command "pybind11_add_module".

把[pybind/pybind11](https://github.com/pybind/pybind11)拉下来手动安装



# HuggingFace Connection

> [!bug]
>
> 【huggingface无法连接】We couldn‘t connect to 'https://huggingface.co'

```bash
echo export HF_ENDPOINT=https://hf-mirror.com >> ~/.bashrc && source ~/.bashrc
```

**References**

[【huggingface无法连接】We couldn‘t connect to 'https://huggingface.co' - 知乎](https://zhuanlan.zhihu.com/p/30237671978)



# Pytorch

> [!bug]
>
> FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
>   return torch.cuda.amp.autocast() if self.activated else NullContextManager()