from uuid import uuid4

from .hypothesis import Hypothesis


class HypothesisGenerator:

    def generate(

        self,

        query,

        activated_crystals,

    ):

        hypotheses = []

        for crystal in activated_crystals[:10]:

            hypotheses.append(

                Hypothesis(

                    id=str(uuid4()),

                    statement=f"{query} -> {crystal.name}",

                    confidence=crystal.activation,

                )

            )

        return hypotheses