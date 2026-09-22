import os
import time
import pandas as pd
from openai import OpenAI
from dotenv import  load_dotenv 

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------
load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

MODEL = "nex-agi/nex-n2.5-pro:free"

PROMPT = """
Write a short story in exactly 100 words about a programmer
who discovers an unusual bug in a production system.
"""

RUNS = 5

# --------------------------------------------------
# HELPER FUNCTION
# --------------------------------------------------

def generate(prompt, temperature, top_p=1.0):
    response = client.responses.create(
        model=MODEL,
        input=prompt,
        temperature=temperature,
        top_p=top_p
    )

    return response.output_text


# --------------------------------------------------
# EXPERIMENT 1: TEMPERATURE
# --------------------------------------------------

results = []

temperatures = [0, 0.3, 0.7, 1.0, 1.5]

for temperature in temperatures:

    print(f"\n===== Temperature = {temperature} =====")

    for run in range(1, RUNS + 1):

        print(f"Run {run}/5")

        try:
            output = generate(
                PROMPT,
                temperature=temperature,
                top_p=1.0
            )

            results.append({
                "experiment": "temperature",
                "setting": temperature,
                "run": run,
                "output": output
            })

            print(output)
            print("-" * 60)

        except Exception as e:
            print("ERROR:", e)

        time.sleep(0.5)


# --------------------------------------------------
# EXPERIMENT 2: TOP-P
# Temperature is fixed at 0.7
# --------------------------------------------------

top_p_values = [0.5, 0.7, 0.9, 0.95, 1.0]

for top_p in top_p_values:

    print(f"\n===== Top-p = {top_p} =====")

    for run in range(1, RUNS + 1):

        print(f"Run {run}/5")

        try:
            output = generate(
                PROMPT,
                temperature=0.7,
                top_p=top_p
            )

            results.append({
                "experiment": "top_p",
                "setting": top_p,
                "run": run,
                "output": output
            })

            print(output)
            print("-" * 60)

        except Exception as e:
            print("ERROR:", e)

        time.sleep(0.5)


# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------

df = pd.DataFrame(results)

df.to_csv(
    "sampling_experiment.csv",
    index=False,
    encoding="utf-8"
)

print("\n===================================")
print("Experiment completed!")
print("Saved to: sampling_experiment.csv")
print("Total successful outputs:", len(df))
print("===================================")