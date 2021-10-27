def main():

    import mindaffectBCI.online_bci
    config = mindaffectBCI.online_bci.load_config("ssvep_bci")
    mindaffectBCI.online_bci.run(**config)

    mindaffectBCI.online_bci.shutdown()

if __name__ == "__main__":
        main()