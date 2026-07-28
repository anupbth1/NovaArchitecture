#!/usr/bin/env python3
"""
NovaRCV - Convert Existing LLM to RCV Architecture
=====================================================
This script converts weights from standard Transformers (Llama, Qwen, GPT-2)
into the NovaRCV architecture.

IMPORTANT LIMITATION: This is a STRUCTURAL conversion, not a capability transfer.
RCV architecture is fundamentally different from standard Transformers.
Converting weights gives you a BETTER initialization than random, but the model
still needs training to learn the corrective iteration behavior.

Process:
1. Load existing model (HuggingFace)
2. Map weights: embedding → embedding, MLP → BrainCell, output → head
3. Initialize RCV-specific components (SlotDebate, reasoner) with scaled random
4. Save as RCV checkpoint

Usage:
    # Convert GPT-2 small
    python nova/scripts/convert_existing_llm.py --source gpt2 --output checkpoints/rcv_from_gpt2.pt
    
    # Convert Llama 1B
    python nova/scripts/convert_existing_llm.py --source meta-llama/Llama-3.2-1B --output checkpoints/rcv_from_llama.pt
    
    # Convert and train immediately
    python nova/scripts/convert_existing_llm.py --source gpt2 --train --steps 1000
"""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import torch.nn as nn

from nova.rcv.nova_brain import NovaRCV
from nova.rcv.config import RCVConfig


def convert_huggingface_model(
    source_model_name: str,
    output_path: str = "checkpoints/rcv_converted.pt",
    device: str = "cpu",
):
    """
    Convert a HuggingFace Transformer model to NovaRCV.
    
    This maps:
    - Transformer embeddings → RCV embeddings
    - Transformer LM head → RCV head  
    - Transformer MLP layers → RCV BrainCell (averaged if multiple)
    """
    print("=" * 70)
    print(f"Converting {source_model_name} to NovaRCV")
    print("=" * 70)
    
    try:
        from transformers import AutoModelForCausalLM, AutoConfig
    except ImportError:
        print("❌ transformers not installed. Install with: pip install transformers")
        return None
    
    # Load source model
    print(f"\n[1/4] Loading {source_model_name}...")
    source = AutoModelForCausalLM.from_pretrained(
        source_model_name, 
        torch_dtype=torch.float32,
        device_map=device,
    )
    source.eval()
    
    # Get source config
    src_cfg = source.config
    src_params = sum(p.numel() for p in source.parameters())
    print(f"  Source params: {src_params:,} ({src_params/1e9:.2f}B)")
    
    # Determine dimensions
    d_model = getattr(src_cfg, 'hidden_size', getattr(src_cfg, 'n_embd', 768))
    vocab_size = getattr(src_cfg, 'vocab_size', getattr(src_cfg, 'n_vocab', 50257))
    num_layers = getattr(src_cfg, 'num_hidden_layers', getattr(src_cfg, 'n_layer', 12))
    expansion = getattr(src_cfg, 'intermediate_size', d_model * 4) // d_model
    
    print(f"  d_model={d_model}, vocab={vocab_size}, layers={num_layers}, expansion={expansion}")
    
    # Create RCV model with matching dimensions
    print(f"\n[2/4] Creating NovaRCV model...")
    rcv_config = RCVConfig(
        d_model=d_model,
        vocab_size=vocab_size,
        expansion=max(2, expansion),
        num_slots=64,
        max_iterations=30,
        seq_len=2048,
    )
    rcv_model = NovaRCV(rcv_config)
    rcv_params = sum(p.numel() for p in rcv_model.parameters())
    print(f"  RCV params: {rcv_params:,} ({rcv_params/1e9:.2f}B)")
    print(f"  Param ratio: {rcv_params/src_params:.2f}x (RCV is smaller)")
    
    # ----- WEIGHT MAPPING -----
    print(f"\n[3/4] Mapping weights...")
    mapped = 0
    skipped = 0
    
    rcv_sd = rcv_model.state_dict()
    source_sd = source.state_dict()
    
    # 1. Embedding mapping
    # Source: transformer.wte.weight or model.embed_tokens.weight
    # RCV: embed.weight
    embed_keys = [k for k in source_sd.keys() if 'embed' in k.lower() and 'weight' in k and 'norm' not in k.lower()]
    embed_keys += [k for k in source_sd.keys() if 'wte' in k.lower() and 'weight' in k]
    
    if embed_keys:
        src_embed = source_sd[embed_keys[0]]
        # Handle size mismatch
        if src_embed.shape[0] <= rcv_sd['embed.weight'].shape[0]:
            rcv_sd['embed.weight'][:src_embed.shape[0]] = src_embed
            print(f"  ✅ Embedding: {src_embed.shape} → {rcv_sd['embed.weight'].shape}")
            mapped += 1
        else:
            print(f"  ⚠️ Embedding size mismatch: {src_embed.shape} vs {rcv_sd['embed.weight'].shape}. Truncating.")
            rcv_sd['embed.weight'] = src_embed[:rcv_sd['embed.weight'].shape[0]]
            mapped += 1
    
    # 2. Output head mapping
    # Source: lm_head.weight or transformer.wte (tied)
    head_keys = [k for k in source_sd.keys() if 'lm_head' in k.lower() and 'weight' in k]
    if not head_keys and embed_keys:
        # Tied embeddings - use same as embed
        head_keys = embed_keys
    
    if head_keys:
        src_head = source_sd[head_keys[0]]
        if src_head.shape[0] <= rcv_sd['head.weight'].shape[0]:
            rcv_sd['head.weight'][:src_head.shape[0]] = src_head
            print(f"  ✅ Head: {src_head.shape} → {rcv_sd['head.weight'].shape}")
            mapped += 1
        else:
            rcv_sd['head.weight'] = src_head[:rcv_sd['head.weight'].shape[0]]
            mapped += 1
    
    # 3. BrainCell mapping (average all source MLP layers)
    # Source: model.layers[i].mlp.{gate_proj,up_proj,down_proj} (Llama)
    # or transformer.h[i].mlp.{c_fc,c_proj} (GPT-2)
    # RCV: reasoner.brain.{fc1,fc2}
    
    mlp_fc1_keys = [k for k in source_sd.keys() if any(x in k.lower() for x in ['mlp.c_fc', 'mlp.gate_proj', 'mlp.up_proj', 'mlp.dense_h_to_4h'])]
    mlp_fc2_keys = [k for k in source_sd.keys() if any(x in k.lower() for x in ['mlp.c_proj', 'mlp.down_proj', 'mlp.dense_4h_to_h'])]
    
    # Filter to weight keys only
    mlp_fc1_weights = [k for k in mlp_fc1_keys if k.endswith('.weight')]
    mlp_fc2_weights = [k for k in mlp_fc2_keys if k.endswith('.weight')]
    
    if mlp_fc1_weights and mlp_fc2_weights:
        # Average all layer weights
        avg_fc1 = sum(source_sd[k] for k in mlp_fc1_weights) / len(mlp_fc1_weights)
        avg_fc2 = sum(source_sd[k] for k in mlp_fc2_weights) / len(mlp_fc2_weights)
        
        # Handle projection size mismatch (Llama uses gate+up)
        target_fc1 = rcv_sd['reasoner.brain.fc1.weight']
        target_fc2 = rcv_sd['reasoner.brain.fc2.weight']
        
        if avg_fc1.shape == target_fc1.shape:
            rcv_sd['reasoner.brain.fc1.weight'] = avg_fc1
            rcv_sd['reasoner.brain.fc2.weight'] = avg_fc2
            print(f"  ✅ BrainCell fc1: {avg_fc1.shape}, fc2: {avg_fc2.shape}")
            mapped += 2
        else:
            # Handle dimension mismatch
            # Average over layers but truncate/pad to match RCV
            d1 = min(avg_fc1.shape[0], target_fc1.shape[0])
            d2 = min(avg_fc1.shape[1], target_fc1.shape[1])
            target_fc1[:d1, :d2] = avg_fc1[:d1, :d2]
            
            d1 = min(avg_fc2.shape[0], target_fc2.shape[0])
            d2 = min(avg_fc2.shape[1], target_fc2.shape[1])
            target_fc2[:d1, :d2] = avg_fc2[:d1, :d2]
            print(f"  ⚠️ BrainCell resized: {avg_fc1.shape} → {target_fc1.shape}")
            mapped += 2
        
        # Average biases
        fc1_bias_keys = [k for k in mlp_fc1_keys if k.endswith('.bias')]
        fc2_bias_keys = [k for k in mlp_fc2_keys if k.endswith('.bias')]
        
        if fc1_bias_keys and 'reasoner.brain.fc1.bias' in rcv_sd:
            avg_bias = sum(source_sd[k] for k in fc1_bias_keys) / len(fc1_bias_keys)
            target = rcv_sd['reasoner.brain.fc1.bias']
            target[:min(len(avg_bias), len(target))] = avg_bias[:min(len(avg_bias), len(target))]
        
        if fc2_bias_keys and 'reasoner.brain.fc2.bias' in rcv_sd:
            avg_bias = sum(source_sd[k] for k in fc2_bias_keys) / len(fc2_bias_keys)
            target = rcv_sd['reasoner.brain.fc2.bias']
            target[:min(len(avg_bias), len(target))] = avg_bias[:min(len(avg_bias), len(target))]
    else:
        print(f"  ⚠️ Could not find MLP layers in source model")
        print(f"     Found keys example: {[k for k in source_sd.keys()][:5]}")
    
    # 4. LayerNorm mapping
    norm_keys = [k for k in source_sd.keys() if ('norm' in k.lower() or 'ln' in k.lower()) and 'weight' in k]
    if norm_keys and 'reasoner.brain.norm1.weight' in rcv_sd:
        # Use the first norm layers
        first_norms = [k for k in norm_keys if 'h.0' in k or 'layers.0' in k or '0.' in k.split('.')[0]]
        if first_norms:
            for nk in first_norms:
                # Map to appropriate RCV norm
                for rcv_key in ['reasoner.brain.norm1.weight', 'reasoner.brain.norm2.weight',
                               'reasoner.iter_norm.weight', 'ln_out.weight']:
                    if rcv_key in rcv_sd and rcv_sd[rcv_key].shape == source_sd[nk].shape:
                        rcv_sd[rcv_key] = source_sd[nk].clone()
                        print(f"  ✅ Norm: {nk} → {rcv_key}")
                        mapped += 1
                        break
    
    print(f"\n[4/4] Applying mapped weights...")
    rcv_model.load_state_dict(rcv_sd, strict=False)
    
    # Save
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    torch.save({
        'source_model': source_model_name,
        'model_state_dict': rcv_model.state_dict(),
        'rcv_config': rcv_config,
    }, output_path)
    
    print(f"\n{'=' * 70}")
    print(f"✅ Conversion complete!")
    print(f"  Source: {source_model_name} ({src_params:,} params)")
    print(f"  RCV:   {output_path} ({rcv_params:,} params)")
    print(f"  Mapped: {mapped} tensors | Skipped: {skipped}")
    print(f"{'=' * 70}")
    print(f"\n⚠️  IMPORTANT: This model needs TRAINING to learn corrective iteration behavior.")
    print(f"   Run: python nova/scripts/train_new_model.py --resume {output_path} --mode custom")
    print()
    
    return rcv_model


# ============================================================
# COMMAND LINE
# ============================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Convert existing LLM to NovaRCV")
    parser.add_argument('--source', default='gpt2',
                        help='HuggingFace model name (e.g., gpt2, meta-llama/Llama-3.2-1B)')
    parser.add_argument('--output', default='checkpoints/rcv_converted.pt',
                        help='Output path for RCV checkpoint')
    parser.add_argument('--device', default='cpu', help='Device')
    parser.add_argument('--train', action='store_true', 
                        help='Train after conversion')
    parser.add_argument('--steps', type=int, default=500,
                        help='Training steps if --train is set')
    
    args = parser.parse_args()
    
    model = convert_huggingface_model(
        source_model_name=args.source,
        output_path=args.output,
        device=args.device,
    )
    
    if args.train and model is not None:
        print("\nStarting post-conversion training...")
        from train_new_model import train
        train(
            config_name="tiny",  # Will be overridden by config from model
            device=args.device,
            steps=args.steps,
            resume=args.output,
        )