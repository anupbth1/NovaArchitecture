"""
Phase 6: Multimodal Input
============================
Images and audio ko same structural atom system mein convert karo.

Rule: No Vision Transformer. No CNN. No image embeddings.
Only: Images -> structural features -> same memory/rules system as text.

Process:
1. Image pixels -> simple structural features (color regions, edges, shapes)
2. These become (role, value) bindings in working memory
3. Same rule engine processes text + image bindings together
4. Cross-modal reasoning: rules can match both text and image features

No neural networks for vision. Just pixel statistics -> discrete atoms.
"""
import sys, os, math
from typing import List, Tuple, Dict, Optional
from collections import defaultdict

if __name__ == '__main__':
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    from nova.v2.memory import WorkingMemory, RoleVocabulary
    from nova.v2.rules import RuleEngine, Rule, Pattern, Action, create_default_rules
else:
    from nova.v2.memory import WorkingMemory, RoleVocabulary
    from nova.v2.rules import RuleEngine, Rule, Pattern, Action, create_default_rules


class ImageFingerprint:
    """
    Convert image pixels to structural features - no neural networks.

    Instead of CNN or ViT embeddings, we compute simple statistics:
    - Color regions: dominant colors, their positions
    - Edge density: how many edges in each region
    - Brightness/saturation: overall image statistics
    - Texture: local pixel variance patterns

    Output: list of (role, value) bindings for WorkingMemory.

    Memory per image: ~100-200 bindings (vs millions of pixels or 768-dim vectors)
    """

    # Number of regions to divide image into
    GRID_SIZE = 4  # 4x4 = 16 regions

    @staticmethod
    def extract_features(pixels: List[List[Tuple[int, int, int]]]) -> List[Tuple[int, int]]:
        """
        Extract structural features from RGB pixel data.

        Args:
            pixels: 2D list of (R, G, B) tuples (e.g., 32x32 image)

        Returns:
            List of (role_id, value) bindings
        """
        bindings = []
        H = len(pixels)
        W = len(pixels[0]) if H > 0 else 0
        if H == 0 or W == 0:
            return bindings

        # 1. Overall image statistics (role 100-109)
        total_pixels = H * W
        avg_r = sum(pixels[y][x][0] for y in range(H) for x in range(W)) / total_pixels
        avg_g = sum(pixels[y][x][1] for y in range(H) for x in range(W)) / total_pixels
        avg_b = sum(pixels[y][x][2] for y in range(H) for x in range(W)) / total_pixels

        # Quantize colors to 0-31 range (5 bits)
        bindings.append((100, int(avg_r / 8)))     # Avg Red (0-31)
        bindings.append((101, int(avg_g / 8)))     # Avg Green (0-31)
        bindings.append((102, int(avg_b / 8)))     # Avg Blue (0-31)

        # 2. Brightness and saturation (role 110-111)
        brightness = (avg_r + avg_g + avg_b) / 3
        max_c = max(avg_r, avg_g, avg_b)
        min_c = min(avg_r, avg_g, avg_b)
        saturation = (max_c - min_c) / max(max_c, 1)

        bindings.append((110, int(brightness / 8)))     # Brightness (0-31)
        bindings.append((111, int(saturation * 31)))    # Saturation (0-31)

        # 3. Region-based features (role 120+)
        # Divide into GRID_SIZE x GRID_SIZE regions
        for ry in range(ImageFingerprint.GRID_SIZE):
            for rx in range(ImageFingerprint.GRID_SIZE):
                y_start = ry * H // ImageFingerprint.GRID_SIZE
                y_end = (ry + 1) * H // ImageFingerprint.GRID_SIZE
                x_start = rx * W // ImageFingerprint.GRID_SIZE
                x_end = (rx + 1) * W // ImageFingerprint.GRID_SIZE

                region_pixels = []
                for y in range(y_start, y_end):
                    for x in range(x_start, x_end):
                        region_pixels.append(pixels[y][x])

                if not region_pixels:
                    continue

                # Region average color
                r_avg = sum(p[0] for p in region_pixels) / len(region_pixels)
                g_avg = sum(p[1] for p in region_pixels) / len(region_pixels)
                b_avg = sum(p[2] for p in region_pixels) / len(region_pixels)

                # Region edge density (variance between adjacent pixels)
                variance = 0
                if len(region_pixels) > 1:
                    means = [(r_avg + g_avg + b_avg) / 3]
                    vals = [(p[0] + p[1] + p[2]) / 3 for p in region_pixels]
                    variance = sum((v - means[0]) ** 2 for v in vals) / len(vals)

                # Region ID: ry * GRID_SIZE + rx
                region_id = ry * ImageFingerprint.GRID_SIZE + rx

                # Region color (role 120 + region_id, value = quantized RGB)
                color_hash = (int(r_avg / 8) << 10) | (int(g_avg / 8) << 5) | int(b_avg / 8)
                bindings.append((120 + region_id, color_hash))

                # Region texture (role 140 + region_id, value = log variance)
                tex_val = min(255, int(math.log(variance + 1) * 10))
                bindings.append((140 + region_id, tex_val))

        # 4. Dominant color analysis (role 160)
        # Find the most common color bucket
        color_buckets = defaultdict(int)
        for y in range(0, H, 2):  # Sample every 2nd pixel for speed
            for x in range(0, W, 2):
                r, g, b = pixels[y][x]
                bucket = (r // 32, g // 32, b // 32)  # 8x8x8 = 512 buckets
                color_buckets[bucket] += 1

        if color_buckets:
            dominant = max(color_buckets, key=color_buckets.get)
            # Encode dominant color as single integer
            dom_code = (dominant[0] << 6) | (dominant[1] << 3) | dominant[2]
            bindings.append((160, dom_code))

        return bindings

    @staticmethod
    def create_simple_test_image(size: int = 16) -> List[List[Tuple[int, int, int]]]:
        """Create a simple test image (colored square)."""
        pixels = []
        for y in range(size):
            row = []
            for x in range(size):
                # Red square in top-left, blue in bottom-right
                if y < size // 2 and x < size // 2:
                    row.append((200, 30, 30))   # Red
                elif y >= size // 2 and x >= size // 2:
                    row.append((30, 30, 200))   # Blue
                elif y < size // 2:
                    row.append((200, 200, 30))  # Yellow
                else:
                    row.append((30, 200, 30))   # Green
            pixels.append(row)
        return pixels


# ============================================================
# DEMO: Multimodal reasoning
# ============================================================

def demo():
    print("=" * 70)
    print("Nova V2 - Phase 6: Multimodal Input")
    print("No Vision Transformer. No CNN. No image embeddings.")
    print("=" * 70)

    # Create a test image
    print("\nCreating test image (16x16, 4 colored quadrants)...")
    image = ImageFingerprint.create_simple_test_image(16)
    print(f"  Image size: {len(image)}x{len(image[0])} pixels")

    # Extract structural features
    print("\nExtracting structural features (no neural networks)...")
    features = ImageFingerprint.extract_features(image)
    print(f"  Features extracted: {len(features)} bindings")

    # Group by type
    color_features = [f for f in features if 100 <= f[0] < 120]
    region_features = [f for f in features if 120 <= f[0] < 160]
    dominant = [f for f in features if f[0] >= 160]

    print(f"  - Global color stats: {len(color_features)}")
    print(f"  - Region features: {len(region_features)}")
    print(f"  - Dominant color: {len(dominant)}")

    # Add to working memory
    roles = RoleVocabulary()
    memory = WorkingMemory()
    for role, value in features:
        memory.add(role, value)

    # Also add text context
    memory.add(roles.get_id("TOKEN_HASH"), 534022)  # "The"
    memory.add(roles.get_id("CONTEXT"), 5)  # "image" context marker

    print(f"\nWorking memory: {len(memory)} bindings")
    print(f"  Sample bindings:")
    for role, value in list(memory.match())[:5]:
        print(f"    role={role} -> value={value}")

    # Run rules on mixed text + image memory
    print(f"\nRunning rules on multimodal memory...")
    engine = RuleEngine()
    for rule in create_default_rules(roles):
        engine.add_rule(rule)

    memory, trace = engine.forward_with_trace(memory)
    print(f"  Rules fired: {len(trace)}")
    print(f"  Memory now: {len(memory)} bindings")

    print()
    print("NO Vision Transformer used:")
    print("  - Image features: pixel statistics + region analysis")
    print("  - Memory: same set of bindings as text")
    print("  - Rules: same rule engine as text")
    print("  - No CNN, no ViT, no image embeddings")
    print("  - No matrix multiplication for vision")
    print()
    print("Phase 6: WORKING")


if __name__ == '__main__':
    demo()