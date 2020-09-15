import collections
import re
import functools
from multiprocessing import Pool, cpu_count
from code.Tools.FileOp import ls
from code.Tools.Time import Timer
import sys


def loadFile(path, verbose: bool = False):
    text = []
    with open(path, 'r') as f:
        text.extend(f.read().splitlines())

    text = [x.replace('*', '') for x in text]
    text = [re.sub('[^ A-Za-z0-9]', '', x) for x in text]
    text = [x for x in text if x != '']
    raw_text = []
    for x in text:
        raw_text.extend(x.split(' '))
    raw_text = [str.lower(x) for x in raw_text if x != '']
    text = raw_text
    return text, len(text)


def loadAllFile(path, process_cnt, verbose: bool = False):
    file_list = ls(path, ".txt")
    file_contents = []
    total_len = 0
    # for file in file_list:
    #     content, length = loadFile(file)
    #     file_contents.append(content)
    #     total_len += length
    with Pool(process_cnt) as p:
        for content, length in p.map(loadFile, file_list):
            # content, length = loadFile(file)
            file_contents.append(content)
            total_len += length
    return file_contents, total_len


def flatten_contents(contents, verbose: bool = False):
    return [word for f in contents for word in f]


def get_word_set_from_content(content, verbose: bool = False):
    return set(content)


def build_dict_from_wordset(wordset, verbose: bool = False):
    # wordset = [word for word in set(wordset)]
    wordset = list(wordset)
    wordset.sort()
    return {word: i for i, word in enumerate(wordset)}, wordset


def get_word_cnt_from_content(content, verbose: bool = False):
    return collections.Counter(content)


def get_word_freq_from_cnt(cnt, total_cnt, verbose: bool = False):
    return {word: wcnt / total_cnt for word, wcnt in cnt.items()}


def drop_word_by_frequency(frequency, frequency_hold, placeholder, word):
    return word if frequency[word] > frequency_hold else placeholder


def drop_in_content_by_frequency(frequency, frequency_hold: float, placeholder,
                                 content):
    return list(
        map(
            functools.partial(drop_word_by_frequency, frequency,
                              frequency_hold, placeholder), content))
    # with Pool(4) as p:
    #     return p.map(
    #         functools.partial(drop_word_by_frequency, frequency,
    #                           frequency_hold, placeholder), content)


def drop_in_contents_by_frequency(frequency, frequency_hold: float,
                                  placeholder, contents):
    # with Pool(4) as p:
    #     return p.map(
    #         functools.partial(drop_in_content_by_frequency, frequency,
    #                           frequency_hold, placeholder), contents)
    return list(
        map(
            functools.partial(drop_in_content_by_frequency, frequency,
                              frequency_hold, placeholder), contents))


def transform_content(dic, content):
    # return map(lambda x: dic[x], content)
    return [dic[x] for x in content]


def transform_contents(dic, process_cnt, contents):
    # return map(functools.partial(transform_content, dic), contents)
    # print("transform %d file" % (len(contents)))
    with Pool(process_cnt) as p:
        contents = p.map(functools.partial(transform_content, dic), contents)
    # print("transform_contents %d file" % (len(contents)))
    return contents


def loadDataSet(path,
                frequency_hold,
                placeholder,
                process_cnt=cpu_count(),
                verbose: bool = False):
    with Timer() as fileLoadTimer:
        contents, length = loadAllFile(path, process_cnt, verbose)

    flattened_contents = flatten_contents(contents, verbose)
    word_cnt = get_word_cnt_from_content(flattened_contents, verbose)
    word_freq = get_word_freq_from_cnt(word_cnt, length, verbose)
    contents = drop_in_contents_by_frequency(word_freq, frequency_hold,
                                             placeholder, contents)
    # print("Loaded %d files" % (len(contents)))

    flattened_contents = flatten_contents(contents, verbose)
    wordset = get_word_set_from_content(flattened_contents, verbose)
    wordtoid, idtoword = build_dict_from_wordset(wordset, verbose)
    id_contents = transform_contents(wordtoid, process_cnt, contents)
    flattened_id_contents = flatten_contents(id_contents, verbose)
    word_cnt = get_word_cnt_from_content(flattened_id_contents, verbose)
    word_freq = get_word_freq_from_cnt(word_cnt, length, verbose)

    return id_contents, length, wordtoid, idtoword, word_freq, len(wordset)
    # id_content = flatten_contents(id_contents, verbose)
    # id_word_cnt = get_word_cnt_from_content(id_content, verbose)
    # id_word_frequency = get_word_freq_from_cnt(id_word_cnt, length, verbose)


if __name__ == "__main__":
    with Timer() as timer:
        loadDataSet(sys.argv[1], float(sys.argv[2]), sys.argv[3])
    print(timer.elapsed)
