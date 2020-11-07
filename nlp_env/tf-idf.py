from itertools import chain

def get_tf(method='log'):



if __name__ == "__main__":
    docs = [
        "it is a good day, I like to stay here",
        "I am happy to be here",
        "I am bob",
        "it is sunny today",
        "I have a party today",
        "it is a dog and that is a cat",
        "there are dog and cat on the tree",
        "I study hard this morning",
        "today is a good day",
        "tomorrow will be a good day",
        "I like coffee, I like book and I like apple",
        "I do not like it",
        "I am kitty, I like bob",
        "I do not care who like bob, but I like kitty",
        "It is coffee time, bring your cup",
    ]
    
    docs = [line.replace(',', '').split(' ') for line in docs]
    vocab = set(chain(*docs))
    
    v2i, i2v = {}, {}
    for idx, v in enumerate(vocab):
        v2i[v] = idx
        i2v[idx] = v
    
    tf = get_tf()

