#!/usr/bin/env python3
import argparse, hashlib, json, re, unicodedata
from collections import defaultdict
from pathlib import Path
import pandas as pd

RELEASE = "partial-dataset-v1.0.0"
FULL = "foodaudit-benchmark-450-v1.0.0"
CATEGORY_OVERRIDES = {
    14: ("04.01.01.03", "alias mapping: peeled fruit"),
    25: ("03.01", "alias mapping: ice cream"),
    38: ("01.06.05", "corrected: question specifies cheese analogue"),
    45: ("12.10.02.01", "corrected: question specifies mayonnaise"),
    55: ("06.09", "corrected: tapioca pudding is a cereal/starch dessert"),
    142: ("05.03", "alias mapping: chocolate coating"),
}
ITEM_FIXES = {
    (7, "\u4e09\u6c2f\u8517\u7cd6"): ("", "SAFE", "ancestor rule 12.10 permits 0.25 g/kg; declared amount is 0.1 g/kg"),
    (44, "\u7ebd\u751c"): ("", "RISK_OVERLIMIT", "04.02.02.03 limit is 0.01 g/kg; declared amount is 0.03 g/kg"),
    (65, "\u963f\u65af\u5df4\u751c"): ("", "SAFE", "12.10.02 limit is 2 g/kg; declared amount is 1.5 g/kg"),
    (65, "\u756a\u8304\u7ea2"): ("", "RISK_OVERLIMIT", "12.10.02 limit is 0.04 g/kg; declared amount is 0.15 g/kg"),
    (85, "\u6297\u574f\u8840\u9178"): ("\u6297\u574f\u8840\u9178\u9499", "", "item name corrected to match the question"),
    (91, "\u756a\u8304\u7ea2\u7d20"): ("", "RISK_FORBIDDEN", "no path-valid authorization for 12.10.01.02"),
}
QUESTION_REPLACEMENTS = {
    63: ("8g/kg", "8g/L", "corrected denominator to match the applicable g/L rule"),
    86: ("0.02g/L", "0.02g/kg", "corrected denominator to match the applicable g/kg rule"),
}

ALIASES = {
    "\u5929\u95e8\u51ac\u9170\u82ef\u4e19\u6c28\u9178\u7532\u916f": "\u963f\u65af\u5df4\u751c",
    "\u4e59\u9170\u78fa\u80fa\u9178\u94be": "\u5b89\u8d5b\u871c",
    "\u7518\u6c28\u9178": "\u6c28\u57fa\u4e59\u9178",
}

def norm(v):
    return "".join(c for c in unicodedata.normalize("NFKC", str(v)).lower() if c.isalnum())

def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1048576), b""): h.update(b)
    return h.hexdigest()

def gold_items(v):
    out = []
    for part in str(v).replace("\uff1a", ":").split("|"):
        if ":" in part:
            a, b = part.split(":", 1); out.append([a.strip(), b.strip().upper()])
    return out

def variants(item):
    vals = [item]
    for sep in ["(", "\uff08", "[", "\u3010"]:
        if sep in item: vals.append(item.split(sep, 1)[0])
    for sep in [",", "\uff0c", "\u3001"]:
        if sep in item: vals.extend(item.split(sep))
    return sorted({x.strip() for x in vals if len(norm(x)) >= 2}, key=len, reverse=True)

def amount(question, item):
    start = -1
    for v in variants(item):
        p = question.find(v)
        if p >= 0: start = p + len(v); break
    if start < 0: return "", None, ""
    tail = question[start:start+70]
    pat = (r"(?i)(\u9002\u91cf|q\.?s\.?|gmp|[-+]?\d+(?:\.\d+)?\s*"
           r"(?:\u03bcg|\u00b5g|ug|mg|g|kg|mL|ml|L|l)?\s*(?:/|\u6bcf)\s*(?:kg|l|100g|100ml))")
    m = re.search(pat, tail)
    if not m:
        m = re.search(r"(?i)(\u9002\u91cf|q\.?s\.?|gmp|[-+]?\d+(?:\.\d+)?\s*(?:\u03bcg|\u00b5g|ug|mg|g|kg|mL|ml|L|l))", tail)
    raw = m.group(1).strip() if m else ""
    n = re.search(r"[-+]?\d+(?:\.\d+)?", raw)
    u = re.search(r"(?i)(\u03bcg|\u00b5g|ug|mg|g|kg|ml|l)(?:\s*/\s*(kg|l|100g|100ml))?", raw)
    unit = (u.group(1) + ("/" + u.group(2) if u.group(2) else "")) if u else ""
    return raw, (float(n.group()) if n else None), unit

def comparable(v, unit):
    if v is None: return None, str(unit).lower()
    u = str(unit).lower().replace("\u00b5", "\u03bc")
    factors = {"mg/kg":.001, "\u03bcg/kg":.000001, "ug/kg":.000001, "g/kg":1,
               "mg/l":.001, "\u03bcg/l":.000001, "ug/l":.000001, "g/l":1}
    if u in factors: return v * factors[u], "g/" + u.split("/",1)[1]
    return v, u

def match_score(item, additive):
    a0, a1, b = norm(item), norm(ALIASES.get(item, item)), norm(additive)
    if a1 == b: return 100
    if a0 == b: return 98
    if len(a1) >= 2 and a1 in b: return 80 + min(15, len(a1))
    if len(b) >= 2 and b in a1: return 75 + min(15, len(b))
    return 0

def consistent(label, av, au, rule):
    lim, unit = rule.get("max_amount"), str(rule.get("unit") or "")
    if label == "SAFE_QS": return lim == -1 or unit.upper() in {"GMP","QS"}
    if label == "RISK_FORBIDDEN": return False
    if lim is None or lim == -1: return label == "SAFE"
    try: lim = float(lim)
    except: return False
    x, xu = comparable(av, au); y, yu = comparable(lim, unit)
    if x is None or xu != yu: return False
    return (label == "SAFE" and x <= y + 1e-12) or (label == "RISK_OVERLIMIT" and x > y + 1e-12)

def main():
    ap = argparse.ArgumentParser()
    for x in ["source_xlsx","rules_csv","nodes_csv","edges_csv","synonyms_csv","output_dir"]:
        ap.add_argument("--" + x.replace("_","-"), required=True)
    a = ap.parse_args()
    paths = {k:Path(getattr(a,k)) for k in vars(a)}
    out = paths["output_dir"]; out.mkdir(parents=True, exist_ok=True)
    src = pd.read_excel(paths["source_xlsx"])
    rules = pd.read_csv(paths["rules_csv"], encoding="utf-8-sig")
    nodes = pd.read_csv(paths["nodes_csv"], encoding="utf-8-sig")
    edges = pd.read_csv(paths["edges_csv"], encoding="utf-8-sig")
    syns = pd.read_csv(paths["synonyms_csv"], encoding="utf-8-sig")

    names = {str(r.category_code):str(r.category_name) for r in nodes.itertuples()}
    parent = {str(r.child_code):str(r.parent_code) for r in edges.itertuples()}
    tree, rulecats, synonyms = defaultdict(set), defaultdict(set), defaultdict(set)
    for r in nodes.itertuples(): tree[norm(r.category_name)].add(str(r.category_code))
    for r in rules.itertuples(): rulecats[norm(r.category_name)].add(str(r.category_code))
    for r in syns.itertuples(): synonyms[norm(r.synonym)].add(str(r.category_code))
    bycode = defaultdict(list)
    for r in rules.to_dict("records"):
        r["category_code"] = str(r["category_code"])
        for f in ["max_amount","remark","calc_basis","cns","ins"]:
            if pd.isna(r.get(f)): r[f] = None if f == "max_amount" else ""
        bycode[r["category_code"]].append(r)

    def path(code):
        ans, seen = [], set()
        while code and code not in seen:
            seen.add(code); ans.append(code); code = parent.get(code)
        return ans

    recipes, items, nested, fixes, issues = [], [], [], [], []
    for _, row in src.sort_values("id").iterrows():
        sid = int(row.id); original_food = str(row.food_entity).strip(); note = ""
        question = str(row.question)
        if sid in QUESTION_REPLACEMENTS:
            old_q, new_q, reason_q = QUESTION_REPLACEMENTS[sid]
            corrected_question = question.replace(old_q, new_q, 1)
            fixes.append({"sample_id":sid,"field":"question_zh","old_value":question,
                          "new_value":corrected_question,"reason":reason_q})
            question = corrected_question
        if sid in CATEGORY_OVERRIDES:
            code, note = CATEGORY_OVERRIDES[sid]; source = "manual_mapping_or_correction"
            if note.startswith("corrected:"):
                fixes.append({"sample_id":sid,"field":"food_category","old_value":original_food,
                              "new_value":f"{names.get(code,'')} ({code})","reason":note})
        else:
            key = norm(original_food); candidates = tree[key]
            if len(candidates)==1: code, source = next(iter(candidates)), "category_name_exact"
            elif len(rulecats[key])==1: code, source = next(iter(rulecats[key])), "rule_category_name_exact"
            elif len(synonyms[key])==1: code, source = next(iter(synonyms[key])), "category_synonym_exact"
            else: code, source = "", "unresolved"; issues.append({"sample_id":sid,"issue":"unresolved_category","value":original_food})
        cname = names.get(code, original_food); pcodes = path(code); pnames = [names.get(x,x) for x in pcodes]
        out_gold, nested_items, changed = [], [], False
        for idx, (raw_item, raw_label) in enumerate(gold_items(row.ground_truth),1):
            fix = ITEM_FIXES.get((sid, raw_item), ("","",""))
            item = fix[0] or raw_item; label = fix[1] or raw_label
            if fix[2]:
                fixes.append({"sample_id":sid,"field":"item_gold","old_value":f"{raw_item}: {raw_label}",
                              "new_value":f"{item}: {label}","reason":fix[2]}); changed = True
            ar, av, au = amount(question, item)
            if not ar and item != raw_item: ar, av, au = amount(question, raw_item)
            if not ar: issues.append({"sample_id":sid,"issue":"amount_not_extracted","value":item})
            candidates = []
            for depth, cc in enumerate(pcodes):
                for rr in bycode.get(cc,[]):
                    score = match_score(item, rr["additive_name"])
                    if score:
                        z = dict(rr); z.update(depth=depth, match_score=score,
                            label_consistent=consistent(label,av,au,rr),
                            category_text_match=int(norm(rr["category_name"]) in norm(original_food+" "+question)))
                        candidates.append(z)
            candidates.sort(key=lambda z:(int(z["label_consistent"]),-z["depth"],z["match_score"],
                                          z["category_text_match"],len(str(z.get("remark") or ""))), reverse=True)
            if label == "RISK_FORBIDDEN":
                primary = None; status = "label_rule_conflict" if candidates else "no_authorization_on_category_path"
                if candidates: issues.append({"sample_id":sid,"issue":"forbidden_label_has_rule","value":item})
            else:
                primary = candidates[0] if candidates else None; status = "supporting_rule_found" if primary else "missing_supporting_rule"
                if not primary: issues.append({"sample_id":sid,"issue":"missing_supporting_rule","value":item})
                elif not primary["label_consistent"]: issues.append({"sample_id":sid,"issue":"label_rule_quantity_conflict","value":item})
            binary = "SAFE" if label in {"SAFE","SAFE_QS"} else "RISK"; out_gold.append(f"{item}: {label}")
            rec = {"release_version":RELEASE,"sample_id":sid,"item_index":idx,"additive_name_raw":raw_item,
                   "additive_name":item,"amount_raw":ar,"amount_value":av,"amount_unit":au,"gold_label":label,
                   "binary_label":binary,"food_category_code":code,"food_category_name":cname,
                   "category_path_codes":" > ".join(pcodes),"category_path_names":" > ".join(pnames),
                   "evidence_status":status,"evidence_rule_category_code":primary["category_code"] if primary else "",
                   "evidence_rule_category_name":primary["category_name"] if primary else "",
                   "evidence_additive_name":primary["additive_name"] if primary else "",
                   "cns":primary.get("cns","") if primary else "","ins":primary.get("ins","") if primary else "",
                   "max_amount":primary.get("max_amount") if primary else "","rule_unit":primary.get("unit","") if primary else "",
                   "remark":primary.get("remark","") if primary else "","calculation_basis":primary.get("calc_basis","") if primary else "",
                   "candidate_rule_count":len(candidates),"source_standard":"GB 2760-2024","correction_applied":bool(fix[2])}
            items.append(rec)
            nested_items.append({k:rec[k] for k in rec if k not in {"release_version","sample_id","food_category_code","food_category_name"}})
        recipe_gold = "RISK" if any(": RISK_" in x for x in out_gold) else "SAFE"
        rr = {"release_version":RELEASE,"sample_id":sid,"question_zh":question,
              "food_entity_original":original_food,"food_category_code":code,"food_category_name":cname,
              "category_mapping_source":source,"category_mapping_note":note,"gold_items":" | ".join(out_gold),
              "recipe_gold_label":recipe_gold,"additive_count":len(out_gold),"bucket_primary":str(row.bucket_primary),
              "bucket_multi":str(row.bucket_multi),"difficulty_level":str(row.difficulty_level),
              "correction_applied":changed or note.startswith("corrected:")}
        recipes.append(rr); nested.append({**rr,"items":nested_items})

    rdf, idf = pd.DataFrame(recipes), pd.DataFrame(items)
    fdf = pd.DataFrame(fixes); qdf = pd.DataFrame(issues,columns=["sample_id","issue","value"])
    files = {}
    for name, df in [("recipes.csv",rdf),("items.csv",idf),("correction_log.csv",fdf),("validation_issues.csv",qdf)]:
        p=out/name; df.to_csv(p,index=False,encoding="utf-8-sig"); files[name]=p
    jp=out/"recipes.jsonl"
    with open(jp,"w",encoding="utf-8") as f:
        for x in nested: f.write(json.dumps(x,ensure_ascii=False)+"\n")
    files[jp.name]=jp
    source_files=[paths[x] for x in ["source_xlsx","rules_csv","nodes_csv","edges_csv","synonyms_csv"]]
    manifest={"release_version":RELEASE,"full_benchmark_version":FULL,
      "description":"Partial dataset release for transparency and independent verification.",
      "declared_full_benchmark_recipe_count":450,"public_partial_recipe_count":len(rdf),"withheld_recipe_count":300,
      "public_item_count":len(idf),"difficulty_counts":rdf.difficulty_level.value_counts().sort_index().to_dict(),
      "bucket_counts":rdf.bucket_primary.value_counts().sort_index().to_dict(),
      "item_label_counts":idf.gold_label.value_counts().sort_index().to_dict(),"correction_count":len(fdf),
      "validation_issue_count":len(qdf),"source_checksums":{p.name:sha(p) for p in source_files},
      "output_checksums":{n:sha(p) for n,p in files.items()},
      "remaining_300_status":"part of the internally curated benchmark; not cleared for unrestricted redistribution"}
    (out/"dataset_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    full={"benchmark_version":FULL,"total_recipe_count":450,"freeze_status":"partial","slices":[
      {"slice_id":"public-partial-150","recipe_count":150,"release_version":RELEASE,"status":"frozen_and_public",
       "dataset_manifest":"data/partial_dataset/v1/dataset_manifest.json"},
      {"slice_id":"internal-curated-300","recipe_count":300,"status":"not_cleared_for_unrestricted_redistribution",
       "note":"Part of the internally curated benchmark and not included in the public partial release."}]}
    (out/"full_benchmark_manifest.json").write_text(json.dumps(full,indent=2)+"\n",encoding="utf-8")
    readme="""# Partial dataset release

This directory publishes a partial dataset from the FoodAudit-AG benchmark.

- Full benchmark described in the manuscript: 450 recipes.
- Publicly released here: 150 recipes.
- Remaining 300 recipes: part of the internally curated benchmark and not cleared for unrestricted redistribution.
- Regulatory reference: GB 2760-2024.
- Release version: partial-dataset-v1.0.0.

The release supports transparency and independent verification. These mechanism-oriented benchmark cases were initialized from generated candidates and manually reviewed. They are not presented as representative commercial or industrial formulations.

Files: recipes.csv, items.csv, recipes.jsonl, correction_log.csv, validation_issues.csv, dataset_manifest.json, full_benchmark_manifest.json.
"""
    (out/"README.md").write_text(readme,encoding="utf-8")
    print(json.dumps(manifest,ensure_ascii=False,indent=2))
    if issues: raise SystemExit("Validation issues remain; inspect validation_issues.csv")

if __name__=="__main__": main()
