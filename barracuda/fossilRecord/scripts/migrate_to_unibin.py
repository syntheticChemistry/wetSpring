#!/usr/bin/env python3
"""Bulk-migrate prokaryotic [[bin]] files into UniBin experiment modules.

For each src/bin/validate_*.rs and benchmark_*.rs:
1. Copy to validation/experiments/exp_<name>.rs (or bench_<name>.rs)
2. Transform: imports, main() → pub fn run(), remove finish()/exit
3. Generate scenario/benchmark registration metadata
4. Output the mod.rs and registry additions

Usage:
    python3 scripts/migrate_to_unibin.py
"""

import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BARRACUDA = ROOT / "barracuda"
BIN_DIR = BARRACUDA / "src" / "bin"
EXP_DIR = BARRACUDA / "src" / "validation" / "experiments"

# Recover feature map from git
def get_feature_map():
    """Parse the committed Cargo.toml for bin name → required-features."""
    result = subprocess.run(
        ["git", "show", "HEAD:barracuda/Cargo.toml"],
        capture_output=True, text=True, cwd=ROOT
    )
    features = {}
    current_name = None
    in_bin = False
    for line in result.stdout.split("\n"):
        stripped = line.strip()
        if stripped == "[[bin]]":
            in_bin = True
            current_name = None
            continue
        if in_bin:
            m = re.match(r'name\s*=\s*"(.+)"', stripped)
            if m:
                current_name = m.group(1)
                features[current_name] = []
            m = re.match(r'required-features\s*=\s*\[(.+)\]', stripped)
            if m:
                feats = [f.strip().strip('"') for f in m.group(1).split(",")]
                if current_name:
                    features[current_name] = feats
            if stripped == "" and current_name:
                in_bin = False
    return features

SKIP_BINS = {
    "wetspring", "wetspring_unibin", "wetspring_guidestone",
    "wetspring_server", "validate_all",
}

SPECIAL_BINS = {
    "wetspring_gonzales_guidestone", "dump_wetspring_scenarios",
    "wetspring_dashboard", "wetspring_science_facade",
}

def classify_track(name):
    """Heuristic track classification from binary name."""
    name_l = name.lower()
    if any(k in name_l for k in ["gonzales", "pharma", "dose", "drug", "ic50"]):
        return "Track::Pharmacology"
    if any(k in name_l for k in ["pipeline", "parity", "ipc", "composition", "nucleus", "primal", "songbird", "trio"]):
        return "Track::Pipeline"
    if any(k in name_l for k in ["benchmark"]):
        return "Track::Science"
    return "Track::Science"

def classify_tier(features):
    """Determine tier from feature requirements."""
    if not features:
        return "Tier::Rust"
    if "gpu" in features or "sovereign-dispatch" in features:
        return "Tier::Both"
    if "ipc" in features or "guidestone" in features:
        return "Tier::Live"
    return "Tier::Rust"

def make_description(name, first_doc_line):
    """Generate a short description from the binary name and doc comment."""
    if first_doc_line:
        desc = first_doc_line.strip().rstrip(".")
        if len(desc) > 80:
            desc = desc[:77] + "..."
        return desc
    pretty = name.replace("validate_", "").replace("benchmark_", "").replace("_", " ")
    return pretty.title()

def extract_first_doc_line(content):
    """Extract the first //! doc comment line (after SPDX)."""
    for line in content.split("\n"):
        stripped = line.strip()
        if stripped.startswith("//!") and "SPDX" not in stripped and "Provenance:" not in stripped:
            text = stripped[3:].strip()
            if text and len(text) > 10:
                return text
    return ""

def transform_file(content, name, is_benchmark):
    """Transform a bin file into an experiment module."""
    lines = content.split("\n")
    
    # === Pass 1: Strip multi-line inner attributes (#![...]) ===
    cleaned = []
    in_inner_attr = False
    bracket_depth = 0
    for line in lines:
        stripped = line.strip()
        if not in_inner_attr and stripped.startswith("#!["):
            in_inner_attr = True
            bracket_depth = stripped.count("[") - stripped.count("]")
            if bracket_depth <= 0:
                in_inner_attr = False
            continue
        if in_inner_attr:
            bracket_depth += stripped.count("[") - stripped.count("]")
            if bracket_depth <= 0:
                in_inner_attr = False
            continue
        cleaned.append(line)
    lines = cleaned
    
    # === Pass 2: Detect patterns ===
    main_start_idx = None
    is_async_main = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if re.match(r'^(pub\s+)?async\s+fn\s+main\s*\(\s*\)', stripped):
            main_start_idx = i
            is_async_main = True
            break
        if re.match(r'^(pub\s+)?fn\s+main\s*\(\s*\)', stripped):
            main_start_idx = i
            break
    
    # === Pass 3: Transform ===
    out = []
    in_main = False
    main_brace_depth = 0
    skipping_validator_new = False  # multi-line Validator::new() skip
    found_validator_new = False
    validator_var = "v"  # name of the Validator variable (detected from let binding)
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # Remove #[tokio::main]
        if stripped == "#[tokio::main]":
            continue
        
        # Transform ALL wetspring_barracuda:: references (imports AND code bodies)
        line = line.replace("wetspring_barracuda::", "crate::")
        
        # Fix include_str! relative paths (moved one directory deeper)
        line = line.replace('include_str!("../../', 'include_str!("../../../')
        line = line.replace('include_str!("../../../graphs/', 'include_str!("../../../../graphs/')
        
        # Remove ExitCode import
        if "use std::process::ExitCode" in stripped:
            continue
        
        # === Inside main() body ===
        if i == main_start_idx:
            out.append(f"/// Run the `{name}` experiment, recording checks into `v`.")
            out.append("pub fn run(v: &mut crate::validation::Validator) {")
            if is_async_main:
                out.append("    let __rt = tokio::runtime::Runtime::new().expect(\"tokio runtime\");")
                out.append("    __rt.block_on(async {")
            in_main = True
            main_brace_depth = stripped.count("{") - stripped.count("}")
            continue
        
        if not in_main:
            out.append(line)
            continue
        
        # We're inside main() — track brace depth
        main_brace_depth += stripped.count("{") - stripped.count("}")
        
        # If we're skipping a multi-line Validator::new(...) statement
        if skipping_validator_new:
            if ";" in stripped:
                skipping_validator_new = False
            continue
        
        # Remove Validator::new() / Validator::silent() — may span multiple lines
        # Detect the variable name (usually 'v' but sometimes 'validator')
        if not found_validator_new:
            let_match = re.match(r'^\s*let\s+mut\s+(\w+)\s*=\s*$', line)
            if let_match:
                validator_var = let_match.group(1)
                found_validator_new = True
                skipping_validator_new = True
                continue
            vn_match = re.match(r'^\s*let\s+mut\s+(\w+)\s*=\s*Validator::(new|silent)\(', stripped)
            if vn_match:
                validator_var = vn_match.group(1)
                found_validator_new = True
                if ";" not in stripped:
                    skipping_validator_new = True
                continue
            if "Validator::new(" in stripped or "Validator::silent(" in stripped:
                found_validator_new = True
                if ";" not in stripped:
                    skipping_validator_new = True
                continue
        
        # Closing brace of main()
        if main_brace_depth <= 0:
            if is_async_main:
                out.append("    });")
            out.append("}")
            in_main = False
            continue
        
        # Remove <var>.finish() / <var>.finish_with_code()
        if stripped == f"{validator_var}.finish();":
            continue
        if f"{validator_var}.finish_with_code()" in stripped:
            continue
        # Only remove process::exit at top level of main (depth 1), not inside closures
        if main_brace_depth == 1 and (stripped.startswith("std::process::exit(") or stripped.startswith("process::exit(")):
            continue
        
        # Fix &mut <var> → <var> (var is now &mut Validator, not owned)
        vn = re.escape(validator_var)
        line = re.sub(rf'\(&mut {vn}\)', f'({validator_var})', line)
        line = re.sub(rf'\(&mut {vn},', f'({validator_var},', line)
        line = re.sub(rf',\s*&mut {vn}\)', f', {validator_var})', line)
        line = re.sub(rf',\s*&mut {vn},', f', {validator_var},', line)
        # Handle standalone &mut v on its own line (multiline arg lists)
        line = re.sub(rf'^\s+&mut {vn},$', f'                {validator_var},', line)
        line = re.sub(rf'^\s+&mut {vn}\)$', f'                {validator_var})', line)
        # Generic: replace &mut v anywhere it appears as a standalone expression
        line = re.sub(rf'(?<![&\w])&mut {vn}(?=\s*[,)\]])', validator_var, line)
        
        # If the validator variable is not 'v', rename it to 'v'
        # so the function signature `pub fn run(v: ...)` matches the body
        if validator_var != "v":
            line = re.sub(rf'\b{vn}\b', 'v', line)
        
        out.append(line)
    
    return "\n".join(out)

def generate_scenario_fn(mod_name, name, track, tier, description, is_benchmark):
    """Generate the run_as_scenario function and SCENARIO/BENCHMARK constant."""
    const_name = "SCENARIO"
    provenance_crate = name
    
    scenario_block = f'''
/// Bridge into [`primalspring::validation::ValidationResult`] for UniBin dispatch.
pub fn run_as_scenario(result: &mut primalspring::validation::ValidationResult) {{
    let mut v = crate::validation::Validator::silent("{name}");
    run(&mut v);
    v.bridge_into(result);
}}

/// Scenario registration for the UniBin registry.
pub const {const_name}: crate::validation::scenarios::registry::Scenario = crate::validation::scenarios::registry::Scenario {{
    meta: crate::validation::scenarios::registry::ScenarioMeta {{
        id: "{name.replace("validate_", "").replace("benchmark_", "")}",
        track: crate::validation::scenarios::registry::{track},
        tier: crate::validation::scenarios::registry::{tier},
        provenance_crate: "{provenance_crate}",
        provenance_date: "2026-05-20",
        description: "{description}",
    }},
    run: |v, _ctx| run_as_scenario(v),
}};
'''
    return scenario_block

def process_bin_file(bin_path, name, features, is_benchmark):
    """Process a single binary file into an experiment module."""
    content = bin_path.read_text()
    
    track = classify_track(name)
    tier = classify_tier(features)
    description = make_description(name, extract_first_doc_line(content))
    # Escape quotes in description for Rust string literal
    description = description.replace('"', '\\"')
    
    prefix = "bench" if is_benchmark else "exp"
    short_name = name.replace("validate_", "").replace("benchmark_", "")
    mod_name = f"{prefix}_{short_name}"
    
    # Transform the file
    transformed = transform_file(content, name, is_benchmark)
    
    # Append scenario registration
    scenario = generate_scenario_fn(mod_name, name, track, tier, description, is_benchmark)
    transformed = transformed.rstrip() + "\n" + scenario
    
    out_path = EXP_DIR / f"{mod_name}.rs"
    out_path.write_text(transformed)
    
    return mod_name, short_name

def main():
    feature_map = get_feature_map()
    
    # Create experiments directory
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    
    validate_mods = []
    benchmark_mods = []
    
    # Process all bin files
    bin_files = sorted(BIN_DIR.glob("*.rs"))
    skipped = []
    errors = []
    
    for bin_path in bin_files:
        stem = bin_path.stem
        
        if stem in SKIP_BINS or stem in SPECIAL_BINS:
            skipped.append(stem)
            continue
        
        is_benchmark = stem.startswith("benchmark_")
        is_validate = stem.startswith("validate_")
        
        if not is_benchmark and not is_validate:
            # Special binaries (wetspring_gonzales_guidestone, etc.)
            skipped.append(stem)
            continue
        
        features = feature_map.get(stem, [])
        
        try:
            mod_name, short_name = process_bin_file(bin_path, stem, features, is_benchmark)
            if is_benchmark:
                benchmark_mods.append((mod_name, short_name, stem))
            else:
                validate_mods.append((mod_name, short_name, stem))
        except Exception as e:
            errors.append((stem, str(e)))
    
    # Build mod name → features mapping for cfg gating
    all_mods = []  # (mod_name, orig_name, features, is_benchmark)
    for mod_name, short_name, orig_name in validate_mods:
        feats = feature_map.get(orig_name, [])
        all_mods.append((mod_name, orig_name, feats, False))
    for mod_name, short_name, orig_name in benchmark_mods:
        feats = feature_map.get(orig_name, [])
        all_mods.append((mod_name, orig_name, feats, True))
    all_mods.sort(key=lambda x: x[0])
    
    def cfg_for_features(feats):
        """Return the #[cfg(...)] attribute for the given features, or empty string."""
        if not feats:
            return ""
        cfgs = set()
        if "gpu" in feats or "sovereign-dispatch" in feats:
            cfgs.add('"gpu"')
        if "npu" in feats:
            cfgs.add('"npu"')
        if "nautilus" in feats:
            cfgs.add('"nautilus"')
        if "facade" in feats:
            cfgs.add('"facade"')
        if "vault" in feats:
            cfgs.add('"vault"')
        if not cfgs:
            return ""
        if len(cfgs) == 1:
            return f'#[cfg(feature = {cfgs.pop()})]\n'
        return '#[cfg(all({}))]'.format(", ".join(f"feature = {c}" for c in sorted(cfgs))) + "\n"
    
    # Generate mod.rs
    mod_content = "// SPDX-License-Identifier: AGPL-3.0-or-later\n"
    mod_content += "//! Migrated experiment modules from prokaryotic binaries.\n"
    mod_content += "//!\n"
    mod_content += f"//! {len(validate_mods)} validation + {len(benchmark_mods)} benchmark experiments.\n\n"
    
    for mod_name, orig_name, feats, is_bench in all_mods:
        cfg = cfg_for_features(feats)
        mod_content += f"{cfg}pub mod {mod_name};\n"
    
    # Add register_all function
    mod_content += "\n/// Register all migrated experiment scenarios into the registry.\n"
    mod_content += "pub fn register_all(r: &mut super::scenarios::ScenarioRegistry) {\n"
    for mod_name, orig_name, feats, is_bench in all_mods:
        cfg_inline = cfg_for_features(feats).replace("\n", " ").strip()
        if cfg_inline:
            mod_content += f"    {cfg_inline}\n"
        mod_content += f"    r.register({mod_name}::SCENARIO);\n"
    mod_content += "}\n"
    
    (EXP_DIR / "mod.rs").write_text(mod_content)
    
    print(f"Migrated: {len(validate_mods)} validate, {len(benchmark_mods)} benchmark")
    print(f"Skipped:  {len(skipped)} ({', '.join(sorted(skipped)[:5])}...)")
    if errors:
        print(f"Errors:   {len(errors)}")
        for name, err in errors:
            print(f"  {name}: {err}")
    print(f"\nWrote {len(validate_mods) + len(benchmark_mods)} modules to {EXP_DIR}/")

if __name__ == "__main__":
    main()
