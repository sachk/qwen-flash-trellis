# Provenance and licensing

This distribution extracts the trellis serving/quantisation work from sachk/homelab-clan. The original-code license has not been selected yet; no blanket relicensing of existing original code is asserted by this extraction. Existing third-party copyright and SPDX notices remain in force.

vLLM is from vllm-project/vllm revision a5a30471ff2bb7f0824f2da10e358af98d304472 (Apache-2.0). The PLE-SSD support patch and tested constraints are the exact selected files from iIIusi0n/qwen3.8-flash-next-cmp170hx revision c7fe762c415c9f53fabccdadc25be949eb92052f. Only those generic model/runtime support files are copied; none of that repository's hardware utilities is included or fetched by the flake. Vendored patch additions retain their Apache-2.0 notices. https://github.com/vllm-project/vllm/blob/a5a30471ff2bb7f0824f2da10e358af98d304472/LICENSE

The model artifacts are derivatives of Qwen/Qwen3.8-Flash-Next-FP8 revision 236dfdf285828023ca3bcd3f37366c58a3469b13 and klee100/Qwen3.8-Flash-Next-AutoRound-3bpw-MTP revision ce0e0b94083895bd836b916f29bf105c40a8162a. Their Qwen Community License 1.0 is preserved with the published weights. It includes additional commercial Model-as-a-Service / AI Work Assistant conditions and must not be presented as Apache/MIT.

Public calibration/evaluation datasets and downloaded medical/legal documents retain their respective source terms; no dataset or private conversation is redistributed here. The released draft vocabulary and layer weights are numeric artifacts, not raw conversation transcripts.

No QuIP#/GLQ lattice research implementation or other rejected decoder experiment is bundled into the serving release.

## GPQA evaluation scaffold

The prompt, choice-shuffle convention and answer-extraction logic in `scripts/gpqa_eval.py` follow [Inspect Evals GPQA](https://github.com/UKGovernmentBEIS/inspect_evals/tree/a68dade7efb803ef50c8442d3f16264b293d1414/src/inspect_evals/gpqa) and are adapted from [Inspect AI](https://github.com/UKGovernmentBEIS/inspect_ai), version 0.3.289. The adapted Inspect AI portions retain its MIT notice:

```text
MIT License

Copyright (c) 2024 UK AI Security Institute

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

