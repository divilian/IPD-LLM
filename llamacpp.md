
1. Install stuff

```
$ sudo apt update
$ sudo apt install -y git build-essential cmake python3 python3-pip
$ sudo apt install -y libopenblas-dev
```

2. Clone llama:

```
git clone https://github.com/ggerganov/llama.cpp
```

3. Now build:

```
$ cd llama.cpp
$ rm -rf build    # shouldn't exist, but in case you need to redo this step
$ cmake -S . -B build \
  -DGGML_BLAS=ON \
  -DGGML_BLAS_VENDOR=OpenBLAS
$ cmake --build build -j
```

(if you want CPU only, omit the `-DGGML_CUDA=ON` flag.)

4. Obtain a model in .gguf format. I grabbed this small one from HF:

```
$ hf download Qwen/Qwen2.5-0.5B-Instruct-GGUF qwen2.5-0.5b-instruct-q5_k_m.gguf --local-dir . --local-dir-use-symlinks False
```

(Need to install `huggingface-cli` first for this to work, of course.)

5. Finally, to actually run it, you:

```
$ ./build/bin/llama-cli -m qwen2.5-0.5b-instruct-q5_k_m.gguf -p "My prompt"
```



### GPU-enabled: didn't work

Now regarding GPU support...it turns out my existing CUDA runtime is too new
   (13.1), and `llama.cpp` needed an older one (12.4-ish). This required:

```
$ wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.54.14_linux.run
$ chmod +x cuda_12.4.0_550.54.14_linux.run
$ sudo ./cuda_12.4.0_550.54.14_linux.run
```

Then I built with:

```
$ cmake -S . -B build \
  -DGGML_CUDA=ON \
  -DCUDAToolkit_ROOT=/usr/local/cuda-12.4 \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DGGML_BLAS=ON \
  -DGGML_BLAS_VENDOR=OpenBLAS
$ cmake --build build -j
```

and this worked (produced an executable), but running it gave me nasty core
dumps Chat couldn't figure out. So I bailed out and went with CPU only.
