#!/usr/bin/env python3

import re
import random
from collections import defaultdict


HIRAGANA = ["あaかkaさsaたtaなnaはhaまmaやyaらraわwa",
            "いiきkiしshiちchiにniひhiみmiりri",
            "うuくkuすsuつtsuぬnuふfuむmuゆyuるruんn",
            "えeけkeせseてteねneへheめmeれre",
            "おoこkoそsoとtoのnoほhoもmoよyoろroをo"]

KATAKANA = ["アaカkaサsaタtaナnaハhaマmaヤyaラraワwa",
            "イiキkiシshiチchiニniヒhiミmiリri",
            "ウuクkuスsuツtsuヌnuフfuムmuユyuルruンn",
            "エeケkeセseテteネneヘheメmeレre",
            "オoコkoソsoトtoノnoホhoモmoヨyoロroヲo"]


def generate_pair(text):

    if type(text) == list:
        text = ''.join(text)

    # text : str
    result = re.findall(r'(.)([a-z]{1,3})', text)
    pairs = {k: v for k, v in result}
    # pairs : dict

    return pairs


def main():
    hiragana_pair = generate_pair(HIRAGANA)
    # hiragana : dict

    wrong_dict = defaultdict(lambda: 0)

    while True:
        k, v = random.choice(list(hiragana_pair.items()))
        ans = input(f"{k}: ")

        if ans == 'q':
            break

        if ans == v:
            print("Correct")
        else:
            print(f"Wrong, {k}: {v}")
            wrong_dict[k] += 1

    print("\n=== Wrong ===")
    for k, v in wrong_dict.items():
        print(f"{k}: {hiragana_pair[k]} | x{v}")


if __name__ == "__main__":
    main()
