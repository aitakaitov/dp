def count_rec(l):
    """
    Recursively counts the elements of a list
    :param l:
    :return:
    """
    count = 0
    # Iterate over the list
    for elem in l:
        # Check if type of element is list
        if type(elem) == list:
            # Again call this function to get the size of this element
            count += count_rec(elem)
        else:
            count += 1
    return count
