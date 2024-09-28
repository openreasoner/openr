def str2bool(x: str):
    if x == "False":
        return False
    elif x == "True":
        return True
    else:
        raise ValueError(
            'you should either input "True" or "False" but not {}'.format(x)
        )


def list_of_ints(arg):
    return list(map(int, arg.split(",")))
