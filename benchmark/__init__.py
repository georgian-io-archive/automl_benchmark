if __name__ == '__main__':
    # this needs to be here because other libs import mp
    import multiprocessing as mp
    try:
        mp.set_start_method('forkserver')
    except RuntimeError:
        print('Failed to set forkserver')
