import re
VOW = set('аеиоуыэюяёѣѫ')

def stem_vowels_rule(lemma, pos, dialect):
    """Number of vowels in the stem of the target form, derived from the Russian lemma."""
    l = lemma
    if pos == 'N':
        stem = l[:-1] if l[-1] in 'аяоеёьй' else l
    elif pos == 'V':
        if l.endswith('ться') or l.endswith('тись') or l.endswith('чься'): stem = l[:-4]
        elif l.endswith('ть') or l.endswith('ти') or l.endswith('чь'): stem = l[:-2]
        else: stem = l
    else:
        if l.endswith('ся'): l = l[:-2]
        stem = l[:-2] if l[-2:] in ('ый', 'ий', 'ой') else l
    nv = sum(1 for c in stem if c in VOW)
    if dialect == 'SEV':
        nv -= len(re.findall('оро|ере|оло', stem))
    return nv
