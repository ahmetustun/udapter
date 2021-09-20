import sys
import random

if len(sys.argv) < 2:
    print("please specify file and number of sentences to keep")
    exit(1)

conll_file = sys.argv[1]
number_to_keep = int(sys.argv[2])

sentences = []
with open(conll_file, 'r', encoding='UTF-8') as f:
    conll_sentence = []
    for line in f:
        if len(line.strip()) == 0:
            if len(conll_sentence) > 0:
                sentence = []
                for conll_token in conll_sentence:
                    if not conll_token or conll_token.startswith('#'):
                        continue
                    sentence.append(conll_token)
                sentences.append(sentence)
            conll_sentence = []
        else:
            conll_sentence.append(line)

total = len(sentences)
lines_to_keep = random.sample(range(0, total), number_to_keep)

fout = open(conll_file + '_' + str(number_to_keep), 'w')
for l in lines_to_keep:
    for tokens in sentences[l]:
        fout.write(tokens)
    fout.write('\n')
fout.close()
