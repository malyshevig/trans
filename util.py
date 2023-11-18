
def readfile(fname: str):
    with open(fname, "rt", encoding="utf-8" ) as fd:
        sb = ""
        for s in fd:
            sb = sb + s

    return sb