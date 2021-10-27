def main():
    %load_ext autoreload
    %autoreload 2
    import mindaffectBCI.online_bci
    #--------------------------- HUB ------------------------------
    # start the utopia-hub process
    hub_process = mindaffectBCI.online_bci.startHubProcess()
    # start the ganglion acquisition process
    # Using brainflow for the acquisition driver.
    #  so change the board_id and other args to use other boards
    acq_args =dict(board_id=1, serial_port='com5') # connect to the ganglion
    acq_process = mindaffectBCI.online_bci.startacquisitionProcess('brainflow', acq_args)

    # start the decoder process, wih default args for noise-tagging
    decoder_args = dict(
            stopband=((45,65),(3,25,'bandpass')),  # frequency filter parameters
            out_fs=100,  # sample rate after pre-processing
            evtlabs=("re","fe"),  # use rising-edge and falling-edge as brain response triggers
            tau_ms=450, # use 450ms as the brain response duration
            calplots=True, # make the end-of-calibration model plots
            predplots=False # don't make plots during prediction
        )
    decoder_process = mindaffectBCI.online_bci.startDecoderProcess('decoder', decoder_args)

    # check all is running?
    print("Hub running {}".format(hub_process.poll() is None))
    print("Acquisition running {}".format(acq_process.is_alive()))
    print("Decoder running {}".format(decoder_process.is_alive()))
    print("Everything running? {}".format(mindaffectBCI.online_bci.check_is_running(hub_process,acq_process,decoder_process)))

     symbols= [["I'm happy","I'm sad"], ["I want to play","I want to sleep"]]

     # run the presentation, with our matrix and default parameters for a noise tag
    from mindaffectBCI.examples.presentation import selectionMatrix
    selectionMatrix.run(symbols=symbols, stimfile="mgold_65_6532_psk_60hz.png")

if __name__ == "__main__":
    main()cd..