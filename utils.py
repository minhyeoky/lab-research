import json
import re


def load_config(file):
    with open(file) as f:
        data = json.load(f)
    return data


def load_vocab(path):
    vocab = {}
    with open(path, mode='r', encoding='utf8') as f:
        for idx, line in enumerate(f):
            line = line.strip('\n')
            vocab[line] = idx
    return vocab


def _prep_txt(txt):
    """
    (철자전사)/(발음전사)로 표기된 경우 발음전사 사용

    b/ : 숨소리 -> 제거
    l/ : 웃음소리 -> 데이터 제거
    o/ : 다음사람 -> 데이터 제거
    n/ : 잡음 -> 제거
    u/ : 전혀 알아들을 수 없음 -> 데이터 제거

    + : 반복 발성 -> 모두 사용, '+'만 제거
    * : 발음이 부정확 -> 제거
    +, * 가 같이 있는 경우, 기호 제거 후 사용
    """
    pattern = '[(]{1}[a-z가-힣0-9 ]+[)]{1}[\/]{1}[(]{1}[a-z가-힣0-9 ]+[)]{1}'
    # (철자전사)/(발음전사)
    delete_list = ['l/', 'o/']
    remove_list = ['b/', 'n/', '+', '*', '*+', '+*', '/', '\n', '.', '?', '!', ',']
    for each in delete_list:
        if each in txt:
            return False
    regex = re.compile(pattern)
    for each in regex.findall(txt):
        rep = each.split('/')
        rep = rep[-1][1:-1]
        txt = txt.replace(each, rep)
    for each in remove_list:
        txt = txt.replace(each, '')
    return ' '.join(txt.strip().split())  # white space 정리
