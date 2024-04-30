import numpy as np
import mne
import matplotlib.pyplot as plt
from scipy.stats import sem

# Event Codes
# 51-53 and 54-56 = 100 filler sentences (no mistakes marked)
# 11-15, 111-115, 21-25, 121-125, 31-35, 131-135, 41-45, 141-145 = 200 subject-verb sentences
# 201-204, 205-208, 211-214, 215-128 = 100 stimulus sentences
# ^400 sentences total

def plot_epoch(filepath, electrode_of_interest, event_of_interest, tmin, tmax):
    raw_data = mne.io.read_raw_curry(filepath, preload=True, verbose = False)
    raw_data = raw_data.filter(l_freq=None, h_freq=35, picks=electrode_of_interest,
                               verbose=False)
    events = mne.events_from_annotations(raw_data)
    epochs = mne.Epochs(raw_data, events=events[0], event_id=events[1],
                        tmin = tmin, tmax = tmax, baseline=(-0.2, 0),
                        verbose=False)
    
    e1 = epochs[event_of_interest]
    e1.apply_baseline((-0.2,0), verbose = False)
    e1.plot_image(picks=electrode_of_interest)

    del raw_data

def plot_events(filepath):
    raw_data = mne.io.read_raw_curry(filepath, preload=False, verbose = False)
    events = mne.events_from_annotations(raw_data, verbose=False)
    
    figure = mne.viz.plot_events(events=events[0], event_id=events[1], sfreq=raw_data.info["sfreq"])
    
    del raw_data

def plot_raw_eeg(filepath, electrode_of_interest = "data", time_window = 10):
    raw_data = mne.io.read_raw_curry(filepath, preload=False, verbose=False)
    raw_data.plot(picks=electrode_of_interest, scalings = "auto", duration = time_window,
                  lowpass=35)
    plt.show()

    del raw_data

def get_erp(filepath, event_of_interest, electrode_of_interest, tmin, tmax):
    # Import and preprocess data (35Hz low-pass filter)
    raw_data = mne.io.read_raw_curry(filepath, preload=True, verbose=False)
    raw_data = raw_data.filter(l_freq=None, h_freq=35, verbose=False)
    
    # Pull events from annotations
    events = mne.events_from_annotations(raw_data, verbose = False)
    event_dict = events[1]
    events = events[0]
    
    event_index = event_dict[event_of_interest]

    # Pull all timestamps for event_of_interest
    timestamps = []
    for i in range(events.shape[0]):
        if events[i,2] == event_index:
            timestamps.append(events[i,0])

    # Pull epochs from 0.5 seconds before event to 1s after event
    num_vals = int((tmax-tmin)*1000)        # seconds * sampling frequency
    epochs = np.empty((0,num_vals))         # create empty array that's the correct length

    for j in timestamps:
        epoch = raw_data.get_data(picks=electrode_of_interest,
                                  start=int(j+(tmin*1000)),
                                  stop=int(j+(tmax*1000)))
        
        # set baseline as mean of 200ms before event
        baseline_val = np.mean(epoch[0, int(0.3*1000):int(0.5*1000)])
        epoch = epoch-baseline_val

        # Convert to microVolts
        epoch = epoch*1e6

        #append value to final array of epochs
        epochs = np.append(epochs, epoch, axis = 0)

    erp_average = np.mean(epochs, axis=0)
    erp_sem = sem(epochs, axis=0)

    del raw_data
    
    return epochs, erp_average, erp_sem

def plot_erp(filepath, event_of_interest, electrode_of_interest, tmin, tmax):
    # Import and preprocess data (35Hz low-pass filter)
    raw_data = mne.io.read_raw_curry(filepath, preload=True, verbose=False)
    raw_data = raw_data.filter(l_freq=None, h_freq=35, verbose=False)
    
    # Pull events from annotations
    events = mne.events_from_annotations(raw_data, verbose = False)
    event_dict = events[1]
    events = events[0]
    
    event_index = event_dict[event_of_interest]

    # Pull all timestamps for event_of_interest
    timestamps = []
    for i in range(events.shape[0]):
        if events[i,2] == event_index:
            timestamps.append(events[i,0])

    # Pull epochs from 0.5 seconds before event to 1s after event
    num_vals = int((tmax-tmin)*1000)        # seconds * sampling frequency
    epochs = np.empty((0,num_vals))         # create empty array that's the correct length

    for j in timestamps:
        epoch = raw_data.get_data(picks=electrode_of_interest,
                                  start=int(j+(tmin*1000)),
                                  stop=int(j+(tmax*1000)))
        
        # set baseline as mean of 200ms before event
        baseline_val = np.mean(epoch[0, int(0.3*1000):int(0.5*1000)])
        epoch = epoch-baseline_val

        # Convert to microVolts
        epoch = epoch*1e6

        #append value to final array of epochs
        epochs = np.append(epochs, epoch, axis = 0)
    
    # Pull average ERP and standard error of the mean
    sampling_frequency = 1000
    num_samples = int((tmax-tmin)*sampling_frequency)

    xvals = np.linspace(tmin, tmax, num_samples)
    erp_average = np.mean(epochs, axis=0)
    erp_sem = sem(epochs, axis=0)
    minus_sem = erp_average-erp_sem
    plus_sem = erp_average+erp_sem

    # Plot Results
    plt.figure()

    plt.plot(xvals, erp_average, color='k')
    plt.fill_between(xvals, minus_sem, plus_sem, color='gray', alpha=0.3)
    plt.axvline(0, ymin=0, ymax=1, linestyle='--', color="k")
    plt.axhline(0, xmin=0, xmax=1, linestyle='--', color='k')
    plt.grid(True)
    plt.xlabel("Time (s)")
    plt.ylabel('Voltage ($\\mu$V)')
    plt.ylim([-10, 10])
    plt.show()

    del raw_data

def get_audio_length(filepath, event_start, event_stop, sampling_frequency = 1000):
    raw_data = mne.io.read_raw_curry(filepath, preload=False, verbose=False)

    events = mne.events_from_annotations(raw_data, verbose=False)
    events_dict = events[1]
    events = events[0]

    e1_idx = events_dict[event_start]
    e2_idx = events_dict[event_stop]

    e1_timestamps = []
    e2_timestamps = []

    for i in range(len(events)):
        if events[i,2] == e1_idx:
            e1_timestamps.append(events[i,0])
        elif events[i,2] == e2_idx:
            e2_timestamps.append(events[i,0])
    
    time_diff = np.array(e2_timestamps)-np.array(e1_timestamps)
    mean_diff = np.mean(time_diff)
    sd_diff = np.std(time_diff)
    sd_diff /= 1000
    mean_diff /= sampling_frequency

    del raw_data

    return mean_diff, sd_diff



def main() -> None:

    # Datafile locations by subject
    s1 = "Data/SS01/SS01-SR-02082016.dat"
    s2 = "Data/SS02/SS02-LV-03082016.dat"
    s3 = "Data/SS03/SS03-MO-03082016.dat"
    s4 = "Data/SS04/SS04-KT-04082016.dat"
    s5 = "Data/SS05/SS05-BJ-05082016.dat"

    # Pick subject to plot
    subject = s4

    # Set tmin, tmax, and sampling frequency
    tmin = -0.5
    tmax = 1.0
    sampling_frequency = 1000

    # Pull average ERP and standard error of the mean
    # for congruent australian (203) and incongruent australisn (207) events
    congruent_epochs, congruent_mean, congruent_sem = get_erp(subject, "203", "Cz",
                                                tmin, tmax)
    mistake_mean, mistake_mean, mistake_sem = get_erp(subject, "207", "Cz",
                                            tmin, tmax)

    congruent_minus_sem = congruent_mean-congruent_sem
    congruent_plus_sem = congruent_mean+congruent_sem

    mistake_minus_sem = mistake_mean-mistake_sem
    mistake_plus_sem = mistake_mean+mistake_sem

    num_samples = int((tmax-tmin)*sampling_frequency)
    xvals = np.linspace(tmin, tmax, num_samples)

    # Plot and save individual results
    plt.figure(figsize=(8,5))

    plt.plot(xvals, congruent_mean, color='blue', label = "Congruent")
    plt.fill_between(xvals, congruent_minus_sem, congruent_plus_sem, color='blue', alpha=0.2)
    plt.plot(xvals, mistake_mean, color = 'red', label= "Mistake")
    plt.fill_between(xvals, mistake_minus_sem, mistake_plus_sem, color='red', alpha = 0.2)
    plt.axvline(0, ymin=0, ymax=1, linestyle='--', color="k")
    plt.axhline(0, xmin=0, xmax=1, linestyle='--', color='k')
    plt.legend()
    plt.grid(True)
    plt.xlabel("Time (s)")
    plt.ylabel('Voltage ($\\mu$V)')
    plt.title(f"Subject {subject[8]} Results")
    plt.xlim([-0.5, 1.0])
    plt.savefig(f"SPHSC525_2024_Dragos/Results/Subject_{subject[8]}_Results.png")
    plt.show()


    # Pull all epochs for all subjects
    s1_congruent_epochs, _, _ = get_erp(s1, "203", "Cz", -0.5, 1.0)
    s1_mistake_epochs, _, _ = get_erp(s1, "207", "Cz", -0.5, 1.0)

    s2_congruent_epochs, _, _ = get_erp(s2, "203", "Cz", -0.5, 1.0)
    s2_mistake_epochs, _, _ = get_erp(s2, "207", "Cz", -0.5, 1.0)

    s3_congruent_epochs, _, _ = get_erp(s3, "203", "Cz", -0.5, 1.0)
    s3_mistake_epochs, _, _ = get_erp(s3, "207", "Cz", -0.5, 1.0)

    s4_congruent_epochs, _, _ = get_erp(s4, "203", "Cz", -0.5, 1.0)
    s4_mistake_epochs, _, _ = get_erp(s4, "207", "Cz", -0.5, 1.0)

    s5_congruent_epochs, _, _ = get_erp(s5, "203", "Cz", -0.5, 1.0)
    s5_mistake_epochs, _, _ = get_erp(s5, "207", "Cz", -0.5, 1.0)

    # Average across all subjects and calculate SEM
    congrunet_epochs = np.concatenate((s1_congruent_epochs,
                                    s2_congruent_epochs,
                                    s3_congruent_epochs,
                                    s4_congruent_epochs,
                                    s5_congruent_epochs))

    mistake_epochs = np.concatenate((s1_mistake_epochs,
                                    s2_mistake_epochs,
                                    s3_mistake_epochs,
                                    s4_mistake_epochs,
                                    s5_mistake_epochs))

    congruent_mean = np.mean(congrunet_epochs, axis=0)
    congruent_sem = sem(congrunet_epochs, axis=0)
    congruent_plus_sem = congruent_mean+congruent_sem
    congruent_minus_sem = congruent_mean-congruent_sem

    mistake_mean = np.mean(mistake_epochs, axis = 0)
    mistake_sem = sem(mistake_epochs, axis=0)
    mistake_plus_sem = mistake_mean+mistake_sem
    mistake_minus_sem = mistake_mean-mistake_sem

    # Plot and save population results
    plt.figure(figsize=(8,5))

    plt.plot(xvals, congruent_mean, color='blue', label = "Congruent")
    plt.fill_between(xvals, congruent_minus_sem, congruent_plus_sem, color='blue', alpha=0.2)
    plt.plot(xvals, mistake_mean, color = 'red', label= "Mistake")
    plt.fill_between(xvals, mistake_minus_sem, mistake_plus_sem, color='red', alpha = 0.2)
    plt.axvline(0, ymin=0, ymax=1, linestyle='--', color="k")
    plt.axhline(0, xmin=0, xmax=1, linestyle='--', color='k')
    plt.legend()
    plt.grid(True)
    plt.xlabel("Time (s)")
    plt.ylabel('Voltage ($\\mu$V)')
    plt.title(f"Population Results")
    plt.xlim([-0.5, 1.0])
    plt.savefig(f"SPHSC525_2024_Dragos/Results/Population_Results.png")
    plt.show()


if __name__ == "__main__":
    main()