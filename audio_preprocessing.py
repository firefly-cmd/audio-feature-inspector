import numpy as np
import librosa


# Create a function to calculate amplitude envelope
def calculate_amplitude_envelope(signal, frame_size, hop_lenght):
    return np.array(
        [max(signal[i : i + frame_size]) for i in range(0, len(signal), hop_lenght)]
    )


# Compute the band energy ratio for each frame
def calculate_band_energy_ratio(
    y, sr, frame_length, hop_length, split_freq=1000, n_fft=None
):
    """
    Compute the Band Energy Ratio (BER) for multiple frames of the audio signal using a split frequency.

    Parameters:
    - y: The audio signal.
    - sr: Sampling rate.
    - frame_length: Number of samples per frame.
    - hop_length: Number of samples between successive frames.
    - split_freq: Frequency at which to split the energy calculation.
    - n_fft: Number of Fourier components used in STFT. If None, it will be set to frame_length.

    Returns:
    - BER values for each frame.
    """
    if n_fft is None:
        n_fft = frame_length

    # Compute the STFT of the signal
    D_frames = np.abs(
        librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=frame_length)
    )

    # Frequency vector
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    # Compute BER for each frame
    ber_values = []
    for D in D_frames.T:
        below_split_energy = np.sum(D[freqs < split_freq]) ** 2
        above_split_energy = np.sum(D[freqs >= split_freq]) ** 2
        ber = below_split_energy / (
            above_split_energy + 1e-10
        )  # Avoid division by zero
        ber_values.append(ber)

    return np.array(ber_values)


# Compute log amplitude spectogram
def calculate_spectrogram(y, n_fft, hop_length, window):
    """
    Calculate the magnitude spectrogram in decibels of an audio signal.

    Parameters:
    - y (np.ndarray): The audio time series.
    - sr (int): The sampling rate of the audio.
    - n_fft (int): The number of samples for each Fourier Transform frame.
                    It affects the frequency resolution of the spectrogram.
    - hop_length (int): The number of samples between successive frames.
                        It affects the time resolution of the spectrogram.
    - window (str): The windowing function to apply to each frame before Fourier Transform.
                    It helps reduce spectral leakage.

    Returns:
    - D (np.ndarray): The magnitude spectrogram in decibels.
    """

    # Compute the STFT
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window=window)

    # Convert to magnitude and then to decibels
    D = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    return D


# Compute root mean square energy
def calculate_rms_energy(y, frame_length, hop_length):
    """
    Calculate the Root Mean Square (RMS) energy of an audio signal.

    Parameters:
    - y (np.ndarray): The audio time series.
    - frame_length (int): The number of samples for each frame.
                        It affects the frequency resolution of the RMS energy calculation.
    - hop_length (int): The number of samples between successive frames.
                        It affects the time resolution of the RMS energy calculation.

    Returns:
    - rms_energy (np.ndarray): The RMS energy for each frame.
    """

    return librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]


# Compute zero crossing rate
def calculate_zero_crossing_rate(y, frame_length, hop_length):
    """
    Calculate the zero crossing rate of an audio signal.

    Parameters:
    - y (np.ndarray): The audio time series.
    - frame_length (int): The number of samples for each frame.
                        It affects the frequency resolution of the ZCR calculation.
    - hop_length (int): The number of samples between successive frames.
                        It affects the time resolution of the ZCR calculation.

    Returns:
    - zcr (np.ndarray): The zero crossing rate for each frame.
    """

    return librosa.feature.zero_crossing_rate(
        y=y, frame_length=frame_length, hop_length=hop_length
    )[0]


# Compute spectral centroid
def calculate_spectral_centroid(y, sr, frame_length, hop_length):
    """
    Calculate the spectral centroid of an audio signal.

    Parameters:
    - y (np.ndarray): The audio time series.
    - sr (int): The sampling rate of the audio.
    - frame_length (int): The number of samples for each frame.
                        It affects the frequency resolution of the spectral centroid calculation.
    - hop_length (int): The number of samples between successive frames.
                        It affects the time resolution of the spectral centroid calculation.

    Returns:
    - spectral_centroid (np.ndarray): The spectral centroid for each frame.
    """

    return librosa.feature.spectral_centroid(
        y=y, sr=sr, n_fft=frame_length, hop_length=hop_length
    )[0]


# Compute spectral bandwidth
def calculate_spectral_bandwidth(y, sr, frame_length, hop_length):
    """
    Calculate the spectral bandwidth of an audio signal.

    Parameters:
    - y (np.ndarray): The audio time series.
    - sr (int): The sampling rate of the audio.
    - frame_length (int): The number of samples for each frame.
                        It affects the frequency resolution of the spectral bandwidth calculation.
    - hop_length (int): The number of samples between successive frames.
                        It affects the time resolution of the spectral bandwidth calculation.

    Returns:
    - spectral_bandwidth (np.ndarray): The spectral bandwidth for each frame.
    """

    return librosa.feature.spectral_bandwidth(
        y=y, sr=sr, n_fft=frame_length, hop_length=hop_length
    )[0]


def calculate_mel_spectrogram(y, sr, frame_length, hop_length, n_mels):
    """
    Calculate the mel spectrogram of an audio signal.

    Parameters:
    - y (np.ndarray): The audio time series.
    - sr (int): The sampling rate of the audio.
    - frame_length (int): The number of samples for each frame.
                        It affects the frequency resolution of the mel spectrogram calculation.
    - hop_length (int): The number of samples between successive frames.
                        It affects the time resolution of the mel spectrogram calculation.
    - n_mels (int): The number of Mel bands to generate.

    Returns:
    - mel_spectrogram (np.ndarray): The mel spectrogram for each frame.
    """

    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=frame_length, hop_length=hop_length, n_mels=n_mels
    )

    return librosa.power_to_db(S, ref=np.max)


def calculate_mfcc(y, sr, n_mfcc, n_fft, hop_length):
    """
    Calculate the Mel-frequency Cepstral Coefficients (MFCCs) for a given audio signal.

    Parameters:
    - y (numpy.ndarray): Audio time-series.
    - sr (int): Sampling rate of the audio.
    - n_mfcc (int): Number of MFCCs to return.
    - n_fft (int): Length of the FFT window.
    - hop_length (int): Number of samples between successive frames.

    Returns:
    - numpy.ndarray: MFCC sequence.
    """
    return librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length
    )


def calculate_first_delta_mfcc(y, sr, n_mfcc, n_fft, hop_length, n_mels):
    """
    Calculates the first delta (derivative) of the Mel-frequency cepstral coefficients (MFCCs) of a given audio signal.

    Parameters:
    - y (np.array): The audio time series.
    - sr (int): The sample rate of the audio time series.
    - n_mfcc (int): Number of MFCCs to compute.
    - n_fft (int): FFT window size.
    - hop_length (int): Number samples between successive frames.
    - n_mels (int): Number of Mel bands.

    Returns:
    - np.array: First delta MFCCs.
    """

    mfccs = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    delta_mfccs = librosa.feature.delta(mfccs, order=1)

    return delta_mfccs


def calculate_second_delta_mfcc(y, sr, n_mfcc, n_fft, hop_length, n_mels):
    """
    Calculates the second delta (acceleration) of the Mel-frequency cepstral coefficients (MFCCs) of a given audio signal.

    Parameters:
    - y (np.array): Audio time series.
    - sr (int): The sample rate of the audio time series.
    - n_mfcc (int): Number of MFCCs to return.
    - n_fft (int): Length of the FFT window.
    - hop_length (int): Number samples between successive frames.
    - n_mels (int): Number of Mel bands to use.

    Returns:
    - np.array: Second delta MFCCs.
    """

    # Calculate the MFCCs
    mfccs = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )

    # Calculate the second delta (acceleration) of the MFCCs
    second_delta_mfccs = librosa.feature.delta(mfccs, order=2)

    return second_delta_mfccs


def calculate_cqt(y, sr, hop_length, n_bins, bins_per_octave):
    """
    Calculates the Constant-Q transform (CQT) of a given audio signal.

    Parameters:
    - y (np.array): Audio time series.
    - sr (int): The sample rate of the audio time series.
    - hop_length (int): Number of samples between successive frames.
    - n_bins (int): Number of frequency bins.
    - bins_per_octave (int): Number of bins per octave.

    Returns:
    - np.array: CQT matrix.
    """

    # Calculate the CQT
    cqt_result = librosa.cqt(
        y=y,
        sr=sr,
        hop_length=hop_length,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave,
    )

    # Convert amplitude to dB scale for better visualization
    cqt_db = librosa.amplitude_to_db(np.abs(cqt_result), ref=np.max)

    return cqt_db


def calculate_gtcc(y, sr, hop_length, n_fft, n_mels, n_coeffs):
    """
    Calculates the Gammatone Cepstral Coefficients (GTCC) of a given audio signal.

    Parameters:
    - y (np.array): Audio time series.
    - sr (int): The sample rate of the audio time series.
    - hop_length (int): Number of samples between successive frames.
    - n_fft (int): FFT window size.
    - n_mels (int): Number of Mel bands.
    - n_coeffs (int): Number of cepstral coefficients.

    Returns:
    - np.array: GTCC matrix.
    """

    # Compute the Mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, hop_length=hop_length, n_fft=n_fft, n_mels=n_mels
    )

    # Convert to gammatone spectrogram
    gamma_spec = np.power(mel_spec, 0.5)

    # Convert the power spectrogram to dB scale
    gamma_spec_db = librosa.power_to_db(gamma_spec)

    # Compute the GTCCs
    gtcc = librosa.feature.mfcc(S=gamma_spec_db, n_mfcc=n_coeffs)

    return gtcc


def calculate_chroma(y, sr, hop_length, n_fft, n_chroma):
    """
    Calculates the chroma feature of a given audio signal.

    Parameters:
    - y (np.array): Audio time series.
    - sr (int): The sample rate of the audio time series.
    - hop_length (int): Number of samples between successive frames.
    - n_fft (int): Number of samples in the FFT window.
    - n_chroma (int): Number of chroma bins to produce.

    Returns:
    - np.array: Chroma matrix.
    """

    # Calculate the chromagram
    chromagram = librosa.feature.chroma_stft(
        y=y, sr=sr, hop_length=hop_length, n_fft=n_fft, n_chroma=n_chroma
    )

    return chromagram


def calculate_spectral_rolloff(y, sr, frame_length, hop_length, roll_percent):
    """
    Calculates the spectral rolloff for a given audio signal.

    Parameters:
    - y (np.array): Audio time series.
    - sr (int): The sample rate of the audio time series.
    - frame_length (int): Window size for the STFT.
    - hop_length (int): Number of samples between successive frames.
    - roll_percent (float): The rolloff percentage.

    Returns:
    - np.array: Spectral rolloff for each frame.
    """

    # Calculate the spectral rolloff
    rolloff = librosa.feature.spectral_rolloff(
        y=y, sr=sr, n_fft=frame_length, hop_length=hop_length, roll_percent=roll_percent
    )

    return rolloff


# Calculate gammatone cepstral coefficients
# Calculate chroma features
# Calculate spectral rolloff
