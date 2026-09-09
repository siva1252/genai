from transformers import pipeline

model_name = "YOUR_MODEL_ID"

generator = pipeline(
    "text-generation",
    model=model_name
)

prompt = "Artificial Intelligence is"

result = generator(
    prompt,
    max_new_tokens=50
)

print(result[0]["generated_text"])


#--------without pipeline-----------------------------


from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "YOUR_MODEL_ID"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(model_name)

prompt = "Artificial Intelligence is"

inputs = tokenizer(prompt, return_tensors="pt")

outputs = model.generate(
    **inputs,
    max_new_tokens=50
)

result = tokenizer.decode(
    outputs[0],
    skip_special_tokens=True
)

print(result)


#comparison models 

import time
from transformers import pipeline

models = [
    "MODEL_A",
    "MODEL_B",
    "MODEL_C"
]

prompt = "Explain machine learning to a beginner."

for model_name in models:

    generator = pipeline(
        "text-generation",
        model=model_name
    )

    start = time.time()

    result = generator(
        prompt,
        max_new_tokens=100
    )

    end = time.time()

    print("\nMODEL:", model_name)
    print("OUTPUT:", result[0]["generated_text"])
    print("LATENCY:", end - start, "seconds")

