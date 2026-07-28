"""
Phase 5: Text Generator
=========================
Hash lookups -> actual text. No softmax. No embedding matrix.

Includes save/load for trained models.
"""
from collections import defaultdict
from typing import List, Tuple, Dict, Optional
import sys, os, random, json, pickle

if __name__ == '__main__':
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    from nova.v2.fingerprint import djb2_hash, StructuralFingerprint, FingerprintParser
    from nova.v2.learning import LearningEngine
else:
    from nova.v2.fingerprint import djb2_hash, StructuralFingerprint, FingerprintParser
    from nova.v2.learning import LearningEngine


class ReverseHashTable:
    """Maps hash -> token(s). Replaces softmax vocabulary."""

    def __init__(self):
        self.table: Dict[int, List[Tuple[str, int]]] = defaultdict(list)

    def add(self, token: str, freq: int = 1):
        h = djb2_hash(token) % StructuralFingerprint.HASH_SPACE
        for i, (t, f) in enumerate(self.table[h]):
            if t == token:
                self.table[h][i] = (token, f + freq)
                return
        self.table[h].append((token, freq))

    def add_corpus(self, tokens: List[str]):
        for t in tokens:
            self.add(t)

    def lookup(self, hash_val: int) -> List[Tuple[str, float]]:
        candidates = self.table.get(hash_val, [])
        if not candidates:
            return []
        total = sum(f for _, f in candidates)
        scored = [(t, f / total) for t, f in candidates]
        scored.sort(key=lambda x: -x[1])
        return scored

    def resolve(self, hash_val: int) -> Optional[str]:
        candidates = self.table.get(hash_val, [])
        if not candidates:
            return None
        candidates.sort(key=lambda x: -x[1])
        return candidates[0][0]

    @property
    def size(self) -> int:
        return len(self.table)


class TextGenerator:
    """Generate text from learned patterns. Supports save/load."""

    def __init__(self, learner: 'LearningEngine', reverse_table: ReverseHashTable,
                 parser: FingerprintParser):
        self.learner = learner
        self.reverse = reverse_table
        self.parser = parser
        self.vocab = self._build_vocab()
    
    def _build_vocab(self) -> Dict[str, int]:
        """Build token->freq mapping from reverse table."""
        vocab = {}
        for hash_val, candidates in self.reverse.table.items():
            for token, freq in candidates:
                vocab[token] = vocab.get(token, 0) + freq
        return vocab

    def generate(self, prompt: str, max_tokens: int = 50,
                 temperature: float = 0.7) -> str:
        """Generate text from prompt with anti-repetition."""
        tokens = prompt.split()
        generated = list(tokens)
        repeat_count = 0
        last_token = None

        for _ in range(max_tokens):
            ctx = generated[-5:] if len(generated) >= 5 else generated
            pred_hash = self.learner._predict(None, ctx)
            
            if pred_hash == 0 or pred_hash is None:
                break
                
            next_token = self.reverse.resolve(pred_hash)
            if next_token is None:
                break

            # Anti-repetition
            if next_token == last_token:
                repeat_count += 1
                if repeat_count >= 3:
                    break
            else:
                repeat_count = 0
            last_token = next_token
            generated.append(next_token)

            # Stop on punctuation
            if next_token in {'.', '!', '?'} and len(generated) > len(tokens) + 5:
                break

        return ' '.join(generated)

    def query(self, prompt: str, max_tokens: int = 30,
              temperature: float = 0.7) -> str:
        """Single query interface."""
        return self.generate(prompt, max_tokens, temperature)

    def save(self, path: str):
        """Save entire model to directory."""
        os.makedirs(path, exist_ok=True)
        
        # Save reverse table as JSON
        rev_data = {str(k): v for k, v in self.reverse.table.items()}
        with open(os.path.join(path, 'reverse_table.json'), 'w') as f:
            json.dump(rev_data, f)
        
        # Save context map as JSON
        ctx_data = {','.join(str(x) for x in k): v 
                    for k, v in self.learner._context_map.items()}
        with open(os.path.join(path, 'context_map.json'), 'w') as f:
            json.dump(ctx_data, f)
        
        # Save metadata
        meta = {
            'step_count': self.learner.step_count,
            'vocab_size': len(self.vocab),
            'patterns': len(self.learner._context_map),
        }
        with open(os.path.join(path, 'meta.json'), 'w') as f:
            json.dump(meta, f)
        
        # Save vocab
        with open(os.path.join(path, 'vocab.json'), 'w') as f:
            json.dump(self.vocab, f)
        
        print(f"  Model saved to '{path}' ({len(self.learner._context_map)} patterns)")
        return path

    @staticmethod
    def load(path: str) -> 'TextGenerator':
        """Load trained model from directory."""
        from nova.v2.memory import RoleVocabulary
        from nova.v2.rules import RuleEngine, create_default_rules
        from nova.v2.learning import LearningEngine
        from nova.v2.fingerprint import FingerprintParser
        
        roles = RoleVocabulary()
        parser = FingerprintParser()
        engine = RuleEngine()
        for rule in create_default_rules(roles):
            engine.add_rule(rule)
        
        learner = LearningEngine(engine, roles)
        reverse = ReverseHashTable()
        
        # Load reverse table
        rev_path = os.path.join(path, 'reverse_table.json')
        if os.path.exists(rev_path):
            with open(rev_path) as f:
                rev_data = json.load(f)
            reverse.table = {int(k): v for k, v in rev_data.items()}
        
        # Load context map
        ctx_path = os.path.join(path, 'context_map.json')
        if os.path.exists(ctx_path):
            with open(ctx_path) as f:
                ctx_data = json.load(f)
            learner._context_map = {}
            for k_str, v in ctx_data.items():
                key = tuple(int(x) for x in k_str.split(','))
                learner._context_map[key] = v
        
        # Load meta
        meta_path = os.path.join(path, 'meta.json')
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            learner.step_count = meta.get('step_count', 0)
        
        gen = TextGenerator(learner, reverse, parser)
        print(f"  Model loaded from '{path}' ({len(learner._context_map)} patterns)")
        return gen


def demo():
    print("=" * 70)
    print("Nova V2 - Phase 5: Text Generator with Save/Load")
    print("=" * 70)

    from nova.v2.memory import RoleVocabulary
    from nova.v2.rules import RuleEngine, create_default_rules

    roles = RoleVocabulary()
    parser = FingerprintParser()
    engine = RuleEngine()
    for rule in create_default_rules(roles):
        engine.add_rule(rule)

    learner = LearningEngine(engine, roles)
    reverse = ReverseHashTable()
    sentences = ["The cat sat on the mat .", "A dog ran quickly home ."]

    all_tokens = []
    for sent in sentences:
        for t in sent.split():
            all_tokens.append(t)
            reverse.add(t)
    parser.fit(all_tokens)

    for sent in sentences:
        tokens = sent.split()
        for i in range(1, len(tokens)):
            learner.train_step(tokens[:i], tokens[i], parser)

    gen = TextGenerator(learner, reverse, parser)
    gen.save("v2_demo_model")
    
    loaded = TextGenerator.load("v2_demo_model")
    print(f"  Query 'The': {loaded.query('The', max_tokens=10)}")
    print(f"  Query 'A':   {loaded.query('A', max_tokens=10)}")
    
    import shutil; shutil.rmtree("v2_demo_model", ignore_errors=True)


if __name__ == '__main__':
    demo()