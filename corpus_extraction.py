import jieba
set_char = set({})
set_all = set({})
with open("./corpus/1000sentences.txt", mode="r",encoding="UTF-8") as f1:
    for i in f1.readlines():
        if i.strip().isdigit():
            continue
        words = set(jieba.cut(i.strip(), cut_all=False))
        set_all |= words
with open("./corpus/1000sentences.txt", mode="r",encoding="UTF-8") as f1:
    words= f1.readlines()
    for char in "".join(words).strip():
        set_char.add(char)

set_char_unicom= set({})
set_new = set({})
with open("./corpus/unicom.txt", mode="r",encoding="UTF-8") as f2:
    for i in f2.readlines():
        if i.strip().isdigit():
            continue
        words = set(jieba.cut(i.strip(), cut_all=False))
        set_new |= words
with open("./corpus/unicom.txt", mode="r",encoding="UTF-8") as f2:
    words = f2.readlines()
    for char in "".join(words).strip():
        set_char_unicom.add(char)
#list(set_new.difference(set_new & set_all))
"""sentences = []
i = 0
sentence = ""
for word in list(set_new.difference(set_new & set_all)):
    sentence += word
    i+=1
    if i>=12:
        sentences.append(sentence)
        i=0
        sentence=""
print(sentences)
with open("./corpus/new_corpus.txt",mode="w",encoding="UTF-8") as f3:
    f3.writelines([s+'\n' for s in sentences])
"""

set_char_shell = set({})
set_shell = set({})
with open('./corpus/transcript.txt', mode="r", encoding="UTF-8") as f3:
    for i in f3.readlines():
        words = i.split()[1:]
        for char in "".join(words).strip():
            set_char_shell.add(char)
#print(len(list(set_shell.difference(set_new & set_all))))


sentences = []
sentence = ""
count = 0
for char in list(set_char_shell.difference(set_char_unicom | set_char)):
    sentence+=char
    count+=1
    if count>=20:
        sentences.append(sentence)
        count=0
        sentence=""
with open("./corpus/new_corpus.txt",mode="a",encoding="UTF-8") as f3:
    f3.writelines([s+'\n' for s in sentences])
