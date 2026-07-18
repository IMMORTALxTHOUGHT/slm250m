#!/usr/bin/env python3
"""Generate a local instruct-tuning dataset (OFFLINE, no HF/GitHub needed).

Produces data/sft_data.jsonl where each line is:
    {"messages": [{"role":"user","content":...}, {"role":"assistant","content":...}]}

Usage:
    python scripts/make_sft_data.py            # -> data/sft_data.jsonl
    python scripts/make_sft_data.py --out my.jsonl --n 300
"""
from __future__ import annotations

import argparse
import json
import os


# --- curated, accurate pairs (the high-quality core) ---------------------- #
CORE = [
    ("What is photosynthesis?", "Photosynthesis is the process by which green plants, algae, and some bacteria convert sunlight, water, and carbon dioxide into glucose and oxygen."),
    ("Explain gravity in one sentence.", "Gravity is the force that attracts objects with mass toward one another, giving weight to physical bodies and keeping planets in orbit."),
    ("What is the capital of France?", "The capital of France is Paris."),
    ("Who wrote Romeo and Juliet?", "Romeo and Juliet was written by William Shakespeare."),
    ("What is the speed of light?", "The speed of light in a vacuum is about 299,792 kilometers per second."),
    ("Define machine learning.", "Machine learning is a branch of artificial intelligence where systems learn patterns from data to make predictions or decisions without being explicitly programmed for each task."),
    ("What is the largest planet in our solar system?", "Jupiter is the largest planet in our solar system."),
    ("How many continents are there?", "There are seven continents: Africa, Antarctica, Asia, Australia, Europe, North America, and South America."),
    ("What is DNA?", "DNA, or deoxyribonucleic acid, is the molecule that carries the genetic instructions used in the growth, development, and reproduction of all known living organisms."),
    ("What is the boiling point of water?", "At standard atmospheric pressure, water boils at 100 degrees Celsius (212 degrees Fahrenheit)."),
    ("Explain what an algorithm is.", "An algorithm is a step-by-step set of instructions for solving a problem or completing a task, like a recipe for a computer."),
    ("What causes the seasons?", "Seasons are caused by the tilt of Earth's axis as it orbits the Sun, which changes how directly sunlight hits each hemisphere during the year."),
    ("What is the capital of Japan?", "The capital of Japan is Tokyo."),
    ("Who was the first person to walk on the Moon?", "Neil Armstrong was the first person to walk on the Moon, in 1969."),
    ("What is climate change?", "Climate change refers to long-term shifts in temperatures and weather patterns, most notably the ongoing rise in global average temperature driven largely by human emissions of greenhouse gases."),
    ("What is the square root of 64?", "The square root of 64 is 8."),
    ("What is protein?", "Protein is a macronutrient made of amino acids that builds and repairs tissues and supports many functions in the body."),
    ("What is the difference between a virus and bacteria?", "Bacteria are single-celled living organisms that can reproduce on their own, while viruses are smaller infectious particles that need a host cell to replicate."),
    ("What is the capital of Australia?", "The capital of Australia is Canberra."),
    ("Explain what an API is.", "An API, or application programming interface, is a set of rules that lets different software programs communicate with each other."),
    ("What is the smallest prime number?", "The smallest prime number is 2."),
    ("What is the purpose of the respiratory system?", "The respiratory system brings oxygen into the body and removes carbon dioxide, supporting cellular energy production."),
    ("Who painted the Mona Lisa?", "The Mona Lisa was painted by Leonardo da Vinci."),
    ("What is the freezing point of water?", "Water freezes at 0 degrees Celsius (32 degrees Fahrenheit) at standard pressure."),
    ("What is an ecosystem?", "An ecosystem is a community of living organisms interacting with each other and their physical environment."),
    ("What is the capital of Germany?", "The capital of Germany is Berlin."),
    ("Explain what a black hole is.", "A black hole is a region of space where gravity is so strong that nothing, not even light, can escape once it passes the event horizon."),
    ("What is the main gas in Earth's atmosphere?", "The main gas in Earth's atmosphere is nitrogen, which makes up about 78 percent."),
    ("What is the capital of Italy?", "The capital of Italy is Rome."),
    ("What is electricity?", "Electricity is the flow of electric charge, typically through wires, that powers devices and transmits energy."),
    ("What is the difference between weather and climate?", "Weather is the short-term state of the atmosphere, while climate is the long-term average of weather patterns in a region."),
    ("What is the capital of Canada?", "The capital of Canada is Ottawa."),
    ("What is a neuron?", "A neuron is a nerve cell that transmits electrical and chemical signals in the brain and nervous system."),
    ("What is the capital of India?", "The capital of India is New Delhi."),
    ("Explain what a database is.", "A database is an organized collection of data that a computer program can store, search, and update efficiently."),
    ("What is the largest ocean?", "The Pacific Ocean is the largest ocean on Earth."),
    ("What is the capital of Brazil?", "The capital of Brazil is Brasília."),
    ("What is the function of the heart?", "The heart is a muscle that pumps blood through the body, delivering oxygen and nutrients to tissues."),
    ("What is the capital of Russia?", "The capital of Russia is Moscow."),
    ("What is a verb?", "A verb is a word that describes an action, state, or occurrence, such as 'run', 'be', or 'happen'."),
    ("What is the capital of Egypt?", "The capital of Egypt is Cairo."),
    ("What is the tallest mountain on Earth?", "Mount Everest is the tallest mountain above sea level, at about 8,849 meters."),
    ("What is the capital of Spain?", "The capital of Spain is Madrid."),
    ("Explain what the Internet is.", "The Internet is a global network of connected computers that communicate using standard protocols to share information."),
]


def build(cur=CORE):
    pairs = list(cur)
    # a few template expansions for variety
    translations = [
        ("Hello", "Bonjour", "French"), ("Thank you", "Merci", "French"),
        ("Goodbye", "Au revoir", "French"), ("Hello", "Hola", "Spanish"),
        ("Thank you", "Gracias", "Spanish"), ("Goodbye", "Adiós", "Spanish"),
        ("Hello", "Hallo", "German"), ("Thank you", "Danke", "German"),
        ("Goodbye", "Auf Wiedersehen", "German"),
    ]
    for eng, foreign, lang in translations:
        pairs.append((f"Translate '{eng}' into {lang}.",
                      f"'{eng}' in {lang} is '{foreign}'."))

    lists = [
        ("Name three primary colors.", "The three primary colors are red, blue, and yellow."),
        ("List three renewable energy sources.", "Three renewable energy sources are solar, wind, and hydroelectric power."),
        ("Name three planets in our solar system.", "Three planets are Mercury, Venus, and Earth."),
        ("List three mammals.", "Three mammals are dogs, whales, and humans."),
        ("Name three programming languages.", "Three programming languages are Python, Java, and C++."),
    ]
    pairs += lists

    howto = [
        ("How do you boil an egg?", "Place eggs in a pot of water, bring it to a boil, then simmer for about 9 to 12 minutes depending on the desired firmness, and cool in cold water."),
        ("How do I save a file?", "Open the File menu or press Ctrl+S, choose a location, name the file, and click Save."),
        ("How do I tie a knot?", "Cross the two ends, loop one around the other, pull it through the loop, and tighten."),
    ]
    pairs += howto

    math_q = [
        ("What is 15 plus 27?", "15 plus 27 is 42."),
        ("What is 100 divided by 4?", "100 divided by 4 is 25."),
        ("What is 7 times 8?", "7 times 8 is 56."),
        ("What is 90 minus 38?", "90 minus 38 is 52."),
    ]
    pairs += math_q

    rewritten = [(f"Rewrite this to be clearer: '{p}'", r) for p, r in CORE[:5]]
    pairs += rewritten

    examples = []
    for instr, resp in pairs:
        examples.append({"messages": [
            {"role": "user", "content": instr},
            {"role": "assistant", "content": resp},
        ]})
    return examples


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--out", default=os.path.join(here, "..", "data", "sft_data.jsonl"))
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    ex = build()
    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for e in ex:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"[make_sft_data] wrote {len(ex)} examples -> {out}")


if __name__ == "__main__":
    main()
