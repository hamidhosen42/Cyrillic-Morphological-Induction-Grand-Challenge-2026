"""Rule-based SEV->KAM (and POM->KAM) mapping learned from train pair diffs."""
import re
ACC = '́'


def r_yat(s):  # unstressed ѣ -> и
    out = list(s)
    for i, ch in enumerate(out):
        if ch == 'ѣ' and not (i + 1 < len(out) and out[i + 1] == ACC): out[i] = 'и'
    return ''.join(out)


def r_sk(s):
    return s.replace('сьц', 'ськ')


def r_velar(s):
    # SEV palatalized velars after ь: ьц->ьк, ьз->ьг, ьс->ьх ; then after к/г/х: и->ы (stress mark kept), ѣ->и
    s = s.replace('ьц', 'ьк').replace('ьз', 'ьг').replace('ьс', 'ьх')
    return s


def r_post_velar(s):
    # after к/г/х (incl. from ськ): и -> ы, ѣ -> и
    out = list(s)
    for i in range(1, len(out)):
        if out[i - 1] in 'кгх':
            if out[i] == 'и': out[i] = 'ы'
            elif out[i] == 'ѣ': out[i] = 'и'
    return ''.join(out)


STAGES = [('yat', r_yat), ('sk', r_sk), ('velar', r_velar), ('postvelar', r_post_velar)]


def to_kam(s, upto=None):
    for name, fn in STAGES:
        s = fn(s)
        if name == upto: break
    return s
