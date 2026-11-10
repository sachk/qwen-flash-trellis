export T3_WORK_DIR="${T3_WORK_DIR:-${XDG_DATA_HOME:-$HOME/.local/share}/qwen-flash-trellis}"
export T3_RUNTIME="${T3_RUNTIME:-$T3_WORK_DIR/runtime}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$T3_RUNTIME/.cache/uv}" TMPDIR="$T3_RUNTIME/tmp" UV_HTTP_TIMEOUT=600
export PYTHONPATH="$T3_REPO/scripts:$T3_REPO/t3${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID TORCH_CUDA_ARCH_LIST=8.0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_BREAKABLE_CUDAGRAPH=1
export VLLM_NO_USAGE_STATS=1 VLLM_DO_NOT_TRACK=1 DO_NOT_TRACK=1
export TORCH_EXTENSIONS_DIR="$T3_RUNTIME/extensions" VLLM_CACHE_ROOT="$T3_RUNTIME/.cache/vllm"
export TRITON_LIBCUDA_PATH="${TRITON_LIBCUDA_PATH:-/run/opengl-driver/lib}"
export PYTORCH_ALLOC_CONF=expandable_segments:True MAX_JOBS="${MAX_JOBS:-2}"
export KUDA_T3MOE_LIB="$T3_KERNELS/lib/libt3moe.so" T3_TESTS_LIB="$T3_KERNELS/lib/libt3moe.so"
export KUDA_HC_FUSE_LIB="$T3_KERNELS/lib/libhcfuse.so"
export KUDA_TOPK_MASK_LIB="$T3_KERNELS/lib/libtopkmask.so"
export KUDA_HC_FUSE=1 KUDA_TOPK_MASK=1 KUDA_T3MTP=1 KUDA_INT8=embed,shared
export KUDA_DRAFT_VOCAB="$T3_KERNELS/share/draft_vocab.json" T3_LEVELS=c2
command_name="${1:-help}"
if (($#)); then shift; fi
case "$command_name" in
  help|--help|-h)
    printf '%s\n' 't3 setup                          Install pinned runtime into T3_RUNTIME' \
      't3 serve --model PATH [OPTIONS]    Serve an existing full T3 checkpoint' \
      't3 python SCRIPT [ARGS]            Run quantisation or verification tools' \
      't3 download REPO [REVISION]        Download all model files to T3_WORK_DIR/model' \
      't3 benchmark [OPTIONS]             Benchmark an already-running API' \
      'T3_WORK_DIR, T3_RUNTIME, T3_BASE_MODEL and T3_FP8_SOURCE configure writable paths.'
    ;;
  setup)
    mkdir -p "$UV_CACHE_DIR" "$TMPDIR" "$T3_RUNTIME/optimization" "$TORCH_EXTENSIONS_DIR"
    if [[ -f "$T3_RUNTIME/installed" ]]; then
      if [[ "$(cat "$T3_RUNTIME/installed")" == "$T3_STAMP" ]]; then
        printf '%s\n' 'Pinned runtime already installed.'
        exit 0
      fi
      printf '%s\n' 'Runtime version differs; select a fresh T3_RUNTIME directory.' >&2
      exit 1
    fi
    if [[ -d "$T3_RUNTIME/src" || -d "$T3_RUNTIME/.venv" ]]; then
      printf '%s\n' 'Incomplete runtime exists; select a fresh T3_RUNTIME or remove only that failed installation.' >&2
      exit 1
    fi
    cp -r --no-preserve=mode "$T3_SOURCE" "$T3_RUNTIME/src"
    cp -r --no-preserve=mode "$T3_OVERLAY"/. "$T3_RUNTIME/src/"
    uv venv --python "$T3_PYTHON" "$T3_RUNTIME/.venv"
    uv pip install --python "$T3_RUNTIME/.venv/bin/python" --torch-backend=cu130 -c "$T3_CONSTRAINTS" "$T3_WHEEL_FILE"
    uv pip install --python "$T3_RUNTIME/.venv/bin/python" --torch-backend=cu130 -c "$T3_CONSTRAINTS" \
      -r "$T3_RUNTIME/src/requirements/build/cuda.txt" 'huggingface-hub==1.32.0'
    VLLM_USE_PRECOMPILED=1 VLLM_VERSION_OVERRIDE="$T3_VLLM_VERSION" VLLM_PRECOMPILED_WHEEL_LOCATION="$T3_WHEEL" \
      uv pip install --python "$T3_RUNTIME/.venv/bin/python" --reinstall --config-settings editable_mode=compat \
      --no-deps --no-build-isolation -e "$T3_RUNTIME/src"
    "$CC" -O3 -shared -fPIC -Wall -Wextra -Werror \
      "$T3_RUNTIME/src/vllm/models/qwen4_exp/nvidia/ple_ssd_io.c" -o "$T3_RUNTIME/optimization/ple_ssd_io.so"
    for tool in "$T3_RUNTIME"/.venv/lib/python3.12/site-packages/triton/backends/nvidia/bin/*; do
      patchelf --set-interpreter "$(cat "$T3_DYNAMIC_LINKER")" "$tool"
    done
    patchelf --set-interpreter "$(cat "$T3_DYNAMIC_LINKER")" \
      "$T3_RUNTIME/.venv/lib/python3.12/site-packages/humming/_native/x86_64/nvrtc_compile"
    printf '%s' "$T3_STAMP" > "$T3_RUNTIME/installed"
    ;;
  serve|python|download|benchmark)
    if [[ ! -f "$T3_RUNTIME/installed" ]]; then
      printf '%s\n' 'Run t3 setup first.' >&2
      exit 1
    fi
    case "$command_name" in
      serve) exec "$T3_RUNTIME/.venv/bin/python" "$T3_REPO/scripts/serve.py" "$@" ;;
      python) exec "$T3_RUNTIME/.venv/bin/python" "$@" ;;
      download)
        export T3_DOWNLOAD_REVISION="${2:-main}"
        exec "$T3_RUNTIME/.venv/bin/hf" download "$1" --revision "$T3_DOWNLOAD_REVISION" --local-dir "$T3_WORK_DIR/model"
        ;;
      benchmark) exec "$T3_RUNTIME/.venv/bin/python" "$T3_REPO/scripts/benchmark.py" "$@" ;;
    esac
    ;;
  *) printf 'Unknown command: %s\n' "$command_name" >&2; exit 2 ;;
esac
