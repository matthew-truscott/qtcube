def check_extension(abspath, ext):
    """checks for an extension and returns True if the path matches
    
    Arguments:
        abspath {str/Path} -- absolute path
        ext {str} -- extension string (without .)

    Currently doesn't do much but might be useful to add certain assertion tests here
    """
    return (ext == abspath.split("/")[-1])