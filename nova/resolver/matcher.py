class ContextMatcher:

    def similarity(

        self,

        word,

        candidate,

    ):

        score=0

        if word.lower()==candidate.lower():

            score+=1

        if word.lower() in candidate.lower():

            score+=0.5

        return score