"""
reproduce_all.py
================================================================================
Causal Tracing and Circuit Discovery in Small Open-Weight Transformers:
Isolating Factual Retrieval and Suppression Mechanisms

Author: Vedansh Kumar (Independent Researcher)
Target: gpt2-small (124M Parameters, 12 Layers, 12 Heads/Layer, d_model=768)
Requirements: pip install transformer_lens jaxtyping einops pandas tabulate matplotlib seaborn
================================================================================
"""

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
from transformer_lens import HookedTransformer

def main():
    # -------------------------------------------------------------------------
    # 0. Environment Setup
    # -------------------------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[+] Initializing Pipeline on: {device.upper()}")
    
    model = HookedTransformer.from_pretrained(
        "gpt2-small",
        center_unembed=True,
        center_writing_weights=True,
        fold_ln=True,
        device=device
    )
    model.set_use_attn_result(True)

    # -------------------------------------------------------------------------
    # 1. Dataset Generation: Isomorphic Counterfactual Fact Pairs
    # -------------------------------------------------------------------------
    print("\n[+] Phase 1: Building Isomorphic Counterfactual Benchmark...")
    factual_entities = [
        ("France", " Paris"), ("Germany", " Berlin"), ("Italy", " Rome"),
        ("Spain", " Madrid"), ("Japan", " Tokyo"), ("China", " Beijing"),
        ("Russia", " Moscow"), ("Greece", " Athens"), ("Egypt", " Cairo"),
        ("Poland", " Warsaw"), ("Ireland", " Dublin"), ("Norway", " Oslo"),
        ("Sweden", " Stockholm")
    ]
    template = "The capital of Germany is Berlin. The capital of France is Paris. The capital of {} is"

    candidate_pairs = []
    for i, (clean_subj, clean_target) in enumerate(factual_entities):
        if clean_subj in ["Germany", "France"]:
            continue
        for j, (corr_subj, corr_target) in enumerate(factual_entities):
            if i == j or corr_subj in ["Germany", "France"]:
                continue
            
            clean_p = template.format(clean_subj)
            corr_p = template.format(corr_subj)
            
            tok_c_ans = model.to_tokens(clean_target, prepend_bos=False)
            tok_d_ans = model.to_tokens(corr_target, prepend_bos=False)
            if tok_c_ans.shape[-1] != 1 or tok_d_ans.shape[-1] != 1:
                continue
                
            toks_c = model.to_tokens(clean_p)
            toks_d = model.to_tokens(corr_p)
            if toks_c.shape[-1] != toks_d.shape[-1]:
                continue
                
            candidate_pairs.append({
                "clean_prompt": clean_p,
                "corr_prompt": corr_p,
                "clean_ans": clean_target,
                "corr_ans": corr_target,
                "clean_tok_id": tok_c_ans.item(),
                "corr_tok_id": tok_d_ans.item(),
                "seq_len": toks_c.shape[-1]
            })

    df_raw = pd.DataFrame(candidate_pairs).drop_duplicates(subset=["clean_prompt", "corr_prompt"])
    
    verified = []
    for _, row in df_raw.iterrows():
        with torch.no_grad():
            logits = model(row["clean_prompt"])
            pred = logits[0, -1, :].argmax().item()
            ld = (logits[0, -1, row["clean_tok_id"]] - logits[0, -1, row["corr_tok_id"]]).item()
            if pred == row["clean_tok_id"] and ld > 1.5:
                verified.append(row)
                
    df_dataset = pd.DataFrame(verified).sample(n=50, random_state=42).reset_index(drop=True)
    
    clean_prompts = df_dataset["clean_prompt"].tolist()
    corr_prompts = df_dataset["corr_prompt"].tolist()
    clean_tok_ids = torch.tensor(df_dataset["clean_tok_id"].tolist(), device=device)
    corr_tok_ids = torch.tensor(df_dataset["corr_tok_id"].tolist(), device=device)
    seq_len = df_dataset["seq_len"].iloc[0]

    with torch.no_grad():
        clean_logits, clean_cache = model.run_with_cache(clean_prompts)
        corr_logits, corr_cache = model.run_with_cache(corr_prompts)

    def compute_ld(logits):
        fl = logits[:, -1, :]
        c = fl.gather(dim=-1, index=clean_tok_ids.unsqueeze(-1)).squeeze(-1)
        d = fl.gather(dim=-1, index=corr_tok_ids.unsqueeze(-1)).squeeze(-1)
        return (c - d).mean().item()

    clean_base_ld = compute_ld(clean_logits)
    corr_base_ld = compute_ld(corr_logits)
    print(f"    Benchmark: N={len(df_dataset)} | Clean Acc: 100.0% | Clean LD: +{clean_base_ld:.4f}")

    # -------------------------------------------------------------------------
    # 2. Phase 2: Direct Logit Attribution & Logit Lens (Figure 1)
    # -------------------------------------------------------------------------
    print("\n[+] Phase 2: Computing Direct Logit Attribution & Logit Lens...")
    unembed_diff = (model.W_U[:, clean_tok_ids] - model.W_U[:, corr_tok_ids]) # [d_model, N]
    accum_resid = clean_cache.accumulated_resid(incl_mid=False)[:, :, -1, :]  # [layers+1, N, d_model]

    logit_lens = []
    for l_idx in range(accum_resid.shape[0]):
        normed = model.ln_final(accum_resid[l_idx])
        l_logits = model.unembed(normed)
        c_val = l_logits.gather(dim=-1, index=clean_tok_ids.unsqueeze(-1)).squeeze(-1)
        d_val = l_logits.gather(dim=-1, index=corr_tok_ids.unsqueeze(-1)).squeeze(-1)
        logit_lens.append((c_val - d_val).mean().item())

    attn_dla, mlp_dla = [], []
    for l in range(model.cfg.n_layers):
        a_out = clean_cache[f"blocks.{l}.hook_attn_out"][:, -1, :]
        m_out = clean_cache[f"blocks.{l}.hook_mlp_out"][:, -1, :]
        attn_dla.append((a_out * unembed_diff.T).sum(dim=-1).mean().item())
        mlp_dla.append((m_out * unembed_diff.T).sum(dim=-1).mean().item())

    head_dla_mat = np.zeros((model.cfg.n_layers, model.cfg.n_heads))
    for l in range(model.cfg.n_layers):
        h_res = clean_cache[f"blocks.{l}.attn.hook_result"][:, -1, :, :]
        for h in range(model.cfg.n_heads):
            head_dla_mat[l, h] = (h_res[:, h, :] * unembed_diff.T).sum(dim=-1).mean().item()

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=300)
    
    axes[0].plot(range(len(logit_lens)), logit_lens, marker="o", color="#1f77b4", lw=2)
    axes[0].axhline(0, color="gray", ls="--", alpha=0.6)
    axes[0].set_title(r"A. Logit Lens Trajectory ($x_l[-1]$)", fontweight="bold")
    axes[0].set_xlabel("Layer Index (0=Embed, 12=Final)")
    axes[0].set_ylabel(r"Logit Difference ($\Delta \mathcal{L}$)")

    x = np.arange(model.cfg.n_layers)
    axes[1].bar(x - 0.19, attn_dla, width=0.38, label="Attention", color="#2ca02c")
    axes[1].bar(x + 0.19, mlp_dla, width=0.38, label="MLP", color="#ff7f0e")
    axes[1].set_title("B. Direct Logit Attribution by Layer", fontweight="bold")
    axes[1].set_xlabel("Layer Index")
    axes[1].legend(frameon=True, facecolor="white")

    sns.heatmap(head_dla_mat, cmap="coolwarm", center=0, ax=axes[2], cbar_kws={'label': r'DLA Contribution'})
    axes[2].set_title(r"C. Head-Level DLA Heatmap ($12 \times 12$)", fontweight="bold")
    axes[2].set_xlabel("Head Index")
    axes[2].set_ylabel("Layer Index")

    plt.tight_layout()
    plt.savefig("figure1_dla_and_logit_lens.pdf", bbox_inches="tight")
    plt.savefig("figure1_dla_and_logit_lens.png", bbox_inches="tight")
    plt.close()
    print("    Saved Figure 1 (PDF & PNG).")

    # -------------------------------------------------------------------------
    # 3. Phase 3: Activation Patching Sweep (Figure 2)
    # -------------------------------------------------------------------------
    print("\n[+] Phase 3: Sweeping Activation Patching (Causal Tracing)...")
    gap = clean_base_ld - corr_base_ld
    resid_patch = np.zeros((model.cfg.n_layers, seq_len))
    mlp_patch = np.zeros((model.cfg.n_layers, seq_len))
    head_patch_last = np.zeros((model.cfg.n_layers, model.cfg.n_heads))
    head_patch_subj = np.zeros((model.cfg.n_layers, model.cfg.n_heads))
    
    subj_pos, last_pos = 18, 19 # Validated prompt token indices

    for l in range(model.cfg.n_layers):
        r_hook = f"blocks.{l}.hook_resid_post"
        m_hook = f"blocks.{l}.hook_mlp_out"
        h_hook = f"blocks.{l}.attn.hook_result"
        
        for p in range(seq_len):
            def p_resid(act, hook, pos=p, name=r_hook):
                act[:, pos, :] = clean_cache[name][:, pos, :]
                return act
            with torch.no_grad():
                out = model.run_with_hooks(corr_prompts, fwd_hooks=[(r_hook, p_resid)])
            resid_patch[l, p] = (compute_ld(out) - corr_base_ld) / gap

            def p_mlp(act, hook, pos=p, name=m_hook):
                act[:, pos, :] = clean_cache[name][:, pos, :]
                return act
            with torch.no_grad():
                out_m = model.run_with_hooks(corr_prompts, fwd_hooks=[(m_hook, p_mlp)])
            mlp_patch[l, p] = (compute_ld(out_m) - corr_base_ld) / gap

        for h in range(model.cfg.n_heads):
            def p_head_l(act, hook, head_i=h, name=h_hook):
                act[:, -1, head_i, :] = clean_cache[name][:, -1, head_i, :]
                return act
            with torch.no_grad():
                out_hl = model.run_with_hooks(corr_prompts, fwd_hooks=[(h_hook, p_head_l)])
            head_patch_last[l, h] = (compute_ld(out_hl) - corr_base_ld) / gap

            def p_head_s(act, hook, head_i=h, name=h_hook):
                act[:, subj_pos, head_i, :] = clean_cache[name][:, subj_pos, head_i, :]
                return act
            with torch.no_grad():
                out_hs = model.run_with_hooks(corr_prompts, fwd_hooks=[(h_hook, p_head_s)])
            head_patch_subj[l, h] = (compute_ld(out_hs) - corr_base_ld) / gap

    sample_toks = [f"{i}: {repr(model.to_string(t))}" for i, t in enumerate(model.to_tokens(clean_prompts[0])[0])]
    fig, axes = plt.subplots(2, 2, figsize=(18, 12), dpi=300)
    
    sns.heatmap(resid_patch, cmap="RdBu_r", center=0, vmin=-0.2, vmax=1.0, xticklabels=sample_toks, yticklabels=range(12), ax=axes[0, 0])
    axes[0, 0].set_title(r"A. Residual Stream Causal Tracing ($x_l[p]$)", fontweight="bold")
    axes[0, 0].tick_params(axis='x', rotation=90)
    
    sns.heatmap(mlp_patch, cmap="RdBu_r", center=0, vmin=-0.2, vmax=0.6, xticklabels=sample_toks, yticklabels=range(12), ax=axes[0, 1])
    axes[0, 1].set_title(r"B. MLP Activation Patching ($\text{MLP}_l[p]$)", fontweight="bold")
    axes[0, 1].tick_params(axis='x', rotation=90)

    sns.heatmap(head_patch_last, cmap="RdBu_r", center=0, vmin=-0.1, vmax=0.6, yticklabels=range(12), ax=axes[1, 0])
    axes[1, 0].set_title(r"C. Head Output Patching at Final Position ($p=-1$)", fontweight="bold")

    sns.heatmap(head_patch_subj, cmap="RdBu_r", center=0, vmin=-0.1, vmax=0.4, yticklabels=range(12), ax=axes[1, 1])
    axes[1, 1].set_title(r"D. Head Output Patching at Subject Position ($p=\text{Subj}$)", fontweight="bold")

    plt.tight_layout()
    plt.savefig("figure2_activation_patching.pdf", bbox_inches="tight")
    plt.savefig("figure2_activation_patching.png", bbox_inches="tight")
    plt.close()
    print("    Saved Figure 2 (PDF & PNG).")

    # -------------------------------------------------------------------------
    # 4. Phase 4: Knockout Mean-Ablation Battery (Figure 3 & Table 1)
    # -------------------------------------------------------------------------
    print("\n[+] Phase 4: Executing Knockout Ablation Battery...")
    mean_head_acts = {}
    for l in range(model.cfg.n_layers):
        mean_head_acts[l] = clean_cache[f"blocks.{l}.attn.hook_result"].mean(dim=0)

    def run_ablation(target_heads):
        if not target_heads:
            final_l = clean_logits[:, -1, :]
            c = final_l.gather(dim=-1, index=clean_tok_ids.unsqueeze(-1)).squeeze(-1)
            d = final_l.gather(dim=-1, index=corr_tok_ids.unsqueeze(-1)).squeeze(-1)
            return (c - d).mean().item(), (final_l.argmax(dim=-1) == clean_tok_ids).float().mean().item() * 100

        layer_heads = {}
        for l, h in target_heads:
            layer_heads.setdefault(l, []).append(h)

        hooks = []
        for l, heads in layer_heads.items():
            def h_fn(act, hook, l_i=l, h_set=heads):
                for h in h_set:
                    act[:, :, h, :] = mean_head_acts[l_i][:, h, :].unsqueeze(0)
                return act
            hooks.append((f"blocks.{l}.attn.hook_result", h_fn))

        with torch.no_grad():
            out = model.run_with_hooks(clean_prompts, fwd_hooks=hooks)
        fl = out[:, -1, :]
        c = fl.gather(dim=-1, index=clean_tok_ids.unsqueeze(-1)).squeeze(-1)
        d = fl.gather(dim=-1, index=corr_tok_ids.unsqueeze(-1)).squeeze(-1)
        return (c - d).mean().item(), (fl.argmax(dim=-1) == clean_tok_ids).float().mean().item() * 100

    experiments = [
        ("Clean Baseline", []),
        ("Ablate L9H8 (Top Writer #1)", [(9, 8)]),
        ("Ablate L10H0 (Top Writer #2)", [(10, 0)]),
        ("Ablate L8H11 (Top Writer #3)", [(8, 11)]),
        ("Ablate Top 2 Writers (L9H8+L10H0)", [(9, 8), (10, 0)]),
        ("Ablate Core Circuit (L9H8+L10H0+L8H11)", [(9, 8), (10, 0), (8, 11)]),
        ("Ablate L10H7 (Top Suppressor)", [(10, 7)]),
        ("Ablate All Suppressors (L10H7+L11H10)", [(10, 7), (11, 10)]),
        ("Control 1: L0H0", [(0, 0)]),
        ("Control 2: L4H4", [(4, 4)]),
        ("Control 3: Random Triple (L1H2+L3H5+L6H1)", [(1, 2), (3, 5), (6, 1)])
    ]

    records = []
    for label, heads in experiments:
        ld, acc = run_ablation(heads)
        drop = ((clean_base_ld - ld) / clean_base_ld) * 100
        records.append({
            "Condition": label,
            "Logit Diff": ld,
            "Drop (%)": drop,
            "Acc (%)": acc,
            "Heads": len(heads)
        })

    df_res = pd.DataFrame(records)
    print("\n" + tabulate(df_res, headers="keys", tablefmt="psql", floatfmt=".2f"))

    palette = ["#1f77b4", "#d62728", "#d62728", "#d62728", "#8b0000", "#500000", "#2ca02c", "#1e7b1e", "#7f7f7f", "#7f7f7f", "#7f7f7f"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=300)
    y_pos = np.arange(len(df_res))

    ax1.barh(y_pos, df_res["Logit Diff"], color=palette, edgecolor="black", lw=0.6)
    ax1.axvline(clean_base_ld, color="blue", ls="--", alpha=0.6)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(df_res["Condition"])
    ax1.invert_yaxis()
    ax1.set_xlabel(r"Clean vs. Corrupted Logit Diff ($\Delta \mathcal{L}$)")
    ax1.set_title("A. Impact of Head Mean-Ablation on Logit Difference", fontweight="bold")

    ax2.barh(y_pos, df_res["Acc (%)"], color=palette, edgecolor="black", lw=0.6)
    ax2.axvline(100.0, color="blue", ls="--", alpha=0.6)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([])
    ax2.invert_yaxis()
    ax2.set_xlim(0, 105)
    ax2.set_xlabel("Top-1 Prediction Accuracy (%)")
    ax2.set_title("B. Task Accuracy Under Ablation", fontweight="bold")

    plt.tight_layout()
    plt.savefig("figure3_ablation_experiments.pdf", bbox_inches="tight")
    plt.savefig("figure3_ablation_experiments.png", bbox_inches="tight")
    plt.close()
    print("    Saved Figure 3 (PDF & PNG).")

    print("\n[✔] ALL PHASES COMPLETE: Figures and Table reproduced successfully.")

if __name__ == "__main__":
    main()
