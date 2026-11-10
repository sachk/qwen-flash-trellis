#!/usr/bin/env python3
"""Adds calibration and eval coverage for domains the first attempt never saw: medical and legal text, more languages.

Sources: Europe PMC CC-BY open-access full text; Federal Register rules and notices (US public domain);
non-English rows of the Vemloir chat traces (ODbL), which are separate from the eval traces.
Documents are split by id, never window by window. Output: sets/calib2, sets/eval_medical, sets/eval_legal.
"""
import hashlib, json, re, sys, time, urllib.parse, urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from paths import ROOT, LIVE
from safetensors import safe_open
import numpy as np
from safetensors.numpy import save_file
from transformers import AutoTokenizer

DATA = ROOT / "data"
OUT = DATA / "sets"
RAW = DATA / "domains"
UA = {"User-Agent": "qwen-flash-trellis-calibration/1.0"}
TOPICS = ["randomized controlled trial oncology", "case report cardiology", "systematic review diabetes", "pediatrics infection cohort", "neurology stroke treatment",
          "psychiatry depression trial", "surgery outcomes retrospective", "pharmacology adverse drug reaction", "public health vaccination", "nephrology chronic kidney disease",
          "pulmonary COPD management", "obstetrics pregnancy complications", "emergency medicine sepsis", "dermatology biologic therapy", "gastroenterology inflammatory bowel disease",
          "radiology imaging diagnostic accuracy", "geriatrics dementia care", "anesthesia perioperative pain", "hematology leukemia", "rheumatology arthritis"]
CALIB_TOKENS, EVAL_TOKENS = 60_000, 30_000
WINDOW = 6000


def get(url, retries=3):
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                return r.read()
        except Exception as e:
            if attempt == retries - 1:
                print("failed", url, e, flush=True)
                return None
            time.sleep(3)


def medical():
    docs = {}
    for topic in TOPICS:
        q = urllib.parse.quote(f'{topic} AND OPEN_ACCESS:y AND SRC:PMC AND LICENSE:"cc by"')
        body = get(f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?query={q}&pageSize=8&format=json&sort=CITED%20desc")
        if not body:
            continue
        for hit in json.loads(body)["resultList"]["result"]:
            pmc = hit.get("pmcid")
            if not pmc or pmc in docs:
                continue
            xml = get(f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmc}/fullTextXML")
            if not xml:
                continue
            try:
                root = ET.fromstring(xml)
            except ET.ParseError:
                continue
            bodies = root.findall(".//body")
            if not bodies:
                continue
            title = "".join(root.find(".//article-title").itertext()) if root.find(".//article-title") is not None else ""
            paragraphs = ["".join(p.itertext()).strip() for b in bodies for p in b.iter("p")]
            text = title + "\n\n" + "\n\n".join(p for p in paragraphs if p)
            if len(text) > 6000:
                docs[pmc] = text
            time.sleep(0.2)
        print("medical", topic, len(docs), flush=True)
    return docs


def legal():
    docs = {}
    url = "https://www.federalregister.gov/api/v1/documents.json?per_page=100&order=newest&conditions%5Btype%5D%5B%5D=RULE&conditions%5Btype%5D%5B%5D=NOTICE&fields%5B%5D=document_number&fields%5B%5D=raw_text_url&fields%5B%5D=title"
    for page in range(1, 4):
        body = get(url + f"&page={page}")
        if not body:
            continue
        for row in json.loads(body)["results"]:
            if len(docs) >= 110:
                break
            raw = get(row["raw_text_url"])
            if not raw:
                continue
            text = re.sub(r"<[^>]+>", "", raw.decode(errors="ignore"))
            if len(text) > 6000:
                docs[row["document_number"]] = row["title"] + "\n\n" + text
            time.sleep(0.2)
    print("legal", len(docs), flush=True)
    return docs


def windows_for(docs, tok, source):
    """One window per document: user gives the start, the assistant continues."""
    im_s, im_e = "<|im_start|>", "<|im_end|>"
    out = []
    for key in sorted(docs, key=lambda k: hashlib.sha256(k.encode()).hexdigest()):
        ids = tok.encode(docs[key], add_special_tokens=False)
        if len(ids) < 1500:
            continue
        ids = ids[:WINDOW]
        cut = min(400, len(ids) // 4)
        prefix = tok.encode(f"{im_s}user\nContinue this document exactly in its own style:\n\n", add_special_tokens=False)
        middle = tok.encode(f"{im_e}\n{im_s}assistant\n<think>\n\n</think>\n\n", add_special_tokens=False)
        seq = np.array(prefix + ids[:cut] + middle + ids[cut:], dtype=np.int32)
        out.append((source, key, seq, [[len(prefix) + cut + len(middle), len(seq)]]))
    return out


def write(name, rows, spans=False):
    ids = np.concatenate([r[2] for r in rows]).astype(np.int32)
    offsets = np.cumsum([0] + [len(r[2]) for r in rows], dtype=np.int64)
    meta = {"sources": json.dumps([r[0] for r in rows]), "sessions": json.dumps([r[1] for r in rows])}
    if spans:
        meta["score_spans"] = json.dumps([r[3] for r in rows])
    save_file({"input_ids": ids, "offsets": offsets}, OUT / f"{name}.safetensors", metadata=meta)
    print(name, len(rows), "sequences", len(ids), "tokens", flush=True)


def main():
    tok = AutoTokenizer.from_pretrained(LIVE)
    RAW.mkdir(exist_ok=True)
    sets = {}
    for name, fn in (("medical", medical), ("legal", legal)):
        path = RAW / f"{name}.json"
        if path.exists():
            docs = json.loads(path.read_text())
        else:
            docs = fn()
            path.write_text(json.dumps(docs))
        sets[name] = windows_for(docs, tok, name)
        print(name, "documents", len(docs), "windows", len(sets[name]), flush=True)
    calib = []
    for name, rows in sets.items():
        evals, used_e, used_c = [], 0, 0
        for r in rows:
            digest = int(hashlib.sha256(r[1].encode()).hexdigest(), 16) % 100
            if digest < 30 and used_e < EVAL_TOKENS:
                evals.append(r)
                used_e += len(r[2])
            elif digest >= 30 and used_c < CALIB_TOKENS:
                calib.append(r)
                used_c += len(r[2])
        write(f"eval_{name}", evals, spans=True)
    with safe_open(OUT / "calib2.safetensors", framework="np") as f:
        ids, offsets, meta = f.get_tensor("input_ids"), f.get_tensor("offsets"), f.metadata()
        sources, sessions = json.loads(meta["sources"]), json.loads(meta["sessions"])
        for i, (source, session) in enumerate(zip(sources, sessions)):
            if source not in ("medical", "legal"):
                calib.append((source, session, ids[offsets[i]:offsets[i + 1]], None))
    write("calib2", calib)


if __name__ == "__main__":
    main()
