def is_relprop_possible(model):
    try:
        model.relprop(cam=None, alpha=1.0)
    except AttributeError:
        return False
    except Exception:
        return True
    return True