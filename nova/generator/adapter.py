from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM

import torch


class Generator:

    def __init__(

        self,

        model_name="Qwen/Qwen2.5-1.5B-Instruct",

        device="cuda",

    ):

        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForCausalLM.from_pretrained(

            model_name,

            torch_dtype=torch.float16,

        ).to(device)

    def generate(

        self,

        prompt,

        max_new_tokens=256,

    ):

        ids = self.tokenizer(

            prompt,

            return_tensors="pt"

        ).to(self.device)

        output = self.model.generate(

            **ids,

            max_new_tokens=max_new_tokens,

            temperature=0.7,

            do_sample=True,

        )

        return self.tokenizer.decode(

            output[0],

            skip_special_tokens=True,

        )