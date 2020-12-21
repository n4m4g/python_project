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
    pairs = generate_pair(KATAKANA)
    # hiragana : dict

    wrong_dict = defaultdict(lambda: 0)

    while True:
        ks = list(pairs.keys())
        random.shuffle(ks)
        vs = [pairs[k] for k in ks]

        for k, v in zip(ks, vs):
            ans = input(f"{k}: ")

            if ans == 'q':
                break

            if ans == v:
                print("Correct")
            else:
                print(f"Wrong, {k}: {v}")
                wrong_dict[k] += 1

        if ans == 'q':
            break

    print("\n=== Wrong ===")
    for k, v in sorted(wrong_dict.items(),
                       key=lambda item: item[1],
                       reverse=True):
        print(f"{k}: {pairs[k]} | x{v}")


if __name__ == "__main__":
    main()
