def pytest_addoption(parser):
    """ add a command line option """
    # parser.addoption("--quiet", action="store_true", default=False,
    #                  help="Shhh! SHHHHHHH!!")

def pytest_configure(config):
    """ called after command line options have been parsed
        and all plugins and initial conftest files been loaded.
    """

    import logging
    from astropy import log as logger

    if config.getoption('verbose') == 2:
        logger.setLevel(logging.DEBUG)

    elif config.getoption('verbose') == 1:
        logger.setLevel(logging.INFO)

    elif config.getoption('quiet'):
        logger.setLevel(logging.WARN)

    else:
        logger.setLevel(logging.INFO+1)
