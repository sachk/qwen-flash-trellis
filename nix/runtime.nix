{ pkgs }:
let
  inherit (pkgs) lib;
  cuda = pkgs.cudaPackages_13_0;
  gcc = pkgs.gcc14;
  python = pkgs.python312;
  cudaHome = pkgs.symlinkJoin {
    name = "trellis-cuda-13.0";
    paths = lib.concatMap (p: map (output: p.${output}) (lib.remove "static" p.outputs)) [
      cuda.cccl cuda.cuda_crt cuda.cuda_cudart cuda.cuda_nvcc cuda.cuda_nvrtc
      cuda.libcublas cuda.libcurand cuda.libcusparse cuda.libcusolver cuda.libcufft
    ];
    postBuild = "ln -s lib $out/lib64";
  };
  rev = "a5a30471ff2bb7f0824f2da10e358af98d304472";
  version = "0.29.1rc1.dev402+ga5a30471f.ple1";
  constraints = ../patches/runtime-lock.txt;
  upstream = pkgs.fetchFromGitHub {
    owner = "vllm-project"; repo = "vllm"; inherit rev;
    hash = "sha256-ANhXsd7yuRM6vBm6n8DMrv+/otSAfAmzwkLWrZfD1pM=";
  };
  source = pkgs.applyPatches {
    name = "vllm-trellis-source";
    src = upstream;
    patches = [ ../patches/qwen38-ple-ssd.patch ../patches/sm80-fp8.patch ../patches/sm80-head.patch ../patches/sm80-moe-sum.patch ../patches/sm80-router.patch ];
  };
  wheel = pkgs.fetchurl {
    url = "https://wheels.vllm.ai/${rev}/vllm-0.29.1rc1.dev402%2Bga5a30471f-cp38-abi3-manylinux_2_28_x86_64.whl";
    hash = "sha256-7Ar7bbn2A1+SJvt/SXRn2wEKw8WcPsZyUHU1daD+SFU=";
  };
  wheelFile = "${pkgs.linkFarm "vllm-wheel" [{ name = "vllm-0.29.1rc1.dev402+ga5a30471f-cp38-abi3-manylinux_2_28_x86_64.whl"; path = wheel; }]}/vllm-0.29.1rc1.dev402+ga5a30471f-cp38-abi3-manylinux_2_28_x86_64.whl";
  t3Src = lib.fileset.toSource {
    root = ../t3;
    fileset = lib.fileset.unions [
      ../t3/t3moe.cu ../t3/hcfuse.cu ../t3/topkmask.cu ../t3/t3gemm.py ../t3/draft_vocab.json
      ../t3/vllm_patch/apply.py ../t3/vllm_patch/t3_method.py
      ../t3/vllm_patch/hc_fuse.py ../t3/vllm_patch/topk_mask.py
    ];
  };
  kernels = pkgs.runCommand "trellis-kernels-sm80" { nativeBuildInputs = [ gcc ]; } ''
    export PATH=${cudaHome}/bin:$PATH CPATH=${cudaHome}/include LIBRARY_PATH=${cudaHome}/lib
    mkdir -p $out/lib $out/share
    flags=(-O3 -arch=sm_80 -std=c++17 -ccbin ${gcc}/bin/g++ -shared -Xcompiler -fPIC)
    nvcc "''${flags[@]}" -DT3_C2 -DT3_NO_AHEAD -DT3_BLOCK_ROWS=4 -DT3_W13_BLOCKS=6 -DT3_W2_BLOCKS=6 \
      -DT3_W2_TILE -DT3_W13_TILE -o $out/lib/libt3moe.so ${t3Src}/t3moe.cu
    nvcc "''${flags[@]}" -o $out/lib/libhcfuse.so ${t3Src}/hcfuse.cu
    nvcc "''${flags[@]}" -o $out/lib/libtopkmask.so ${t3Src}/topkmask.cu
    cp ${t3Src}/draft_vocab.json $out/share/draft_vocab.json
  '';
  overlay = pkgs.runCommand "vllm-trellis-overlay" { nativeBuildInputs = [ python ]; } ''
    mkdir tree
    cp -r --no-preserve=mode ${source}/vllm tree/vllm
    python3 ${t3Src}/vllm_patch/apply.py tree
    install -Dm444 ${../patches/sm80_head.py} tree/vllm/models/qwen4_exp/nvidia/sm80_head.py
    cd tree
    find vllm -type f | sort | while IFS= read -r f; do
      cmp -s "$f" "${source}/$f" || install -Dm444 "$f" "$out/$f"
    done
  '';
  chatTemplate = pkgs.concatText "qwen-flash-chat.jinja" [ ../chat-template-prelude.jinja ../chat_template.jinja ];
  libraryPath = "/run/opengl-driver/lib:${cudaHome}/lib:${lib.makeLibraryPath [ pkgs.stdenv.cc.cc.lib pkgs.zlib cuda.cuda_cudart ]}";
  package = pkgs.writeShellApplication {
    name = "t3";
    runtimeInputs = [ pkgs.uv pkgs.patchelf pkgs.coreutils pkgs.findutils pkgs.gnugrep gcc cudaHome pkgs.ninja pkgs.cmake ];
    text = ''
      export T3_REPO=${../.}
      export T3_SOURCE=${source} T3_OVERLAY=${overlay} T3_KERNELS=${kernels}
      export T3_WHEEL=${wheel} T3_WHEEL_FILE=${wheelFile}
      export T3_CONSTRAINTS=${constraints}
      export T3_CHAT_TEMPLATE=${chatTemplate}
      export T3_PYTHON=${python}/bin/python3.12 T3_VLLM_VERSION=${version}
      export T3_STAMP=${lib.escapeShellArg "${source} ${overlay} ${wheel} ${constraints}"}
      export T3_DYNAMIC_LINKER=${pkgs.stdenv.cc}/nix-support/dynamic-linker
      export CUDA_HOME=${cudaHome} CPATH=${cudaHome}/include LIBRARY_PATH=${cudaHome}/lib
      export CC=${gcc}/bin/gcc CXX=${gcc}/bin/g++ CUDAHOSTCXX=${gcc}/bin/g++
      export LD_LIBRARY_PATH=${lib.escapeShellArg libraryPath}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
      ${builtins.readFile ../scripts/runtime.sh}
    '';
  };
in { inherit package kernels overlay source cudaHome; }
