# Provenance and licensing

This private distribution extracts the trellis serving/quantisation work from sachk/homelab-clan. No blanket relicensing of existing original code is asserted by this extraction; the owner should choose the original-code license before making this code repository public. Existing third-party copyright and SPDX notices remain in force.

vLLM is from vllm-project/vllm revision a5a30471ff2bb7f0824f2da10e358af98d304472 (Apache-2.0). The PLE-SSD support patch and tested constraints are the exact selected files from iIIusi0n/qwen3.8-flash-next-cmp170hx revision c7fe762c415c9f53fabccdadc25be949eb92052f. Only those generic model/runtime support files are copied; none of that repository's hardware utilities is included or fetched by the flake. Vendored patch additions retain their Apache-2.0 notices. https://github.com/vllm-project/vllm/blob/a5a30471ff2bb7f0824f2da10e358af98d304472/LICENSE

The model artifacts are derivatives of Qwen/Qwen3.8-Flash-Next-FP8 revision 236dfdf285828023ca3bcd3f37366c58a3469b13 and klee100/Qwen3.8-Flash-Next-AutoRound-3bpw-MTP revision ce0e0b94083895bd836b916f29bf105c40a8162a. Their Qwen Community License 1.0 is preserved with the published weights. It includes additional commercial Model-as-a-Service / AI Work Assistant conditions and must not be presented as Apache/MIT.

Public calibration/evaluation datasets and downloaded medical/legal documents retain their respective source terms; no dataset or private conversation is redistributed here. The released draft vocabulary and layer weights are numeric artifacts, not raw conversation transcripts.

No QuIP#/GLQ lattice research implementation or other rejected decoder experiment is bundled into the serving release.
