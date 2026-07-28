from dataclasses import dataclass
from typing import List

import spacy

from nova.representation.semantic import (
    SemanticUnit,
    SemanticType,
)
from nova.representation.graph import ConceptGraph

nlp = spacy.load("en_core_web_sm")


class SemanticParser:

    def __init__(self):
        self.nlp = nlp

    def parse(self, text: str) -> ConceptGraph:

        doc = self.nlp(text)

        graph = ConceptGraph()

        token_map = {}

        # ---------- Create Semantic Units ----------

        for token in doc:

            if token.dep_ == "ROOT":
                stype = SemanticType.ACTION

            elif token.pos_ == "VERB":
                stype = SemanticType.ACTION

            elif token.dep_ in ("nsubj", "nsubjpass"):
                stype = SemanticType.ENTITY

            elif token.dep_ in ("dobj", "obj"):
                stype = SemanticType.OBJECT

            elif token.ent_type_ == "GPE":
                stype = SemanticType.LOCATION

            elif token.ent_type_ == "DATE":
                stype = SemanticType.TIME

            elif token.like_num:
                stype = SemanticType.NUMBER

            else:
                stype = SemanticType.UNKNOWN

            unit = SemanticUnit.create(
                semantic_type=stype,
                value=token.text,
                confidence=1.0,
                pos=token.pos_,
                dep=token.dep_,
            )

            graph.add_node(unit)

            token_map[token.i] = unit

        # ---------- Dependency Graph ----------

        for token in doc:

            if token.head.i == token.i:
                continue

            graph.connect(
                token_map[token.head.i],
                token_map[token.i],
                relation=token.dep_,
            )

        return graph