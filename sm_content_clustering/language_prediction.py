import fasttext


FT = None
def get_lang(s, model_path, index=0, score=False):
    global FT
    if FT is None:
        FT = fasttext.load_model(model_path)
    s = s.strip().replace('\n', ' ')
    if not s:
        return 'no_text'
    p = FT.predict(s, k=index+1)
    if not score:
        return p[0][index].replace('__label__', '')
    else:
        return p[1][index]


def is_lang_in_top(s, lang, index=0, score=False):
    p = FT.predict(s, k=index+1)
    l = '__label__' + lang.lower()
    for la, sc in zip(p[0], p[1]):
        if la == l:
            if not score:
                return 1
            else:
                return sc
    if not score:
        return 0
    else:
        return 0.0
