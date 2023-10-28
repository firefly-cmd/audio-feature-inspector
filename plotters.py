import plotly.graph_objects as go
import numpy as np
import librosa


# Plot the amplitude envelope results
def plot_peak_envelope(signal, sr, envelope, hop_length):
    """
    Plot the audio signal with its peak envelope.

    Parameters:
    - signal: The audio signal.
    - sr: Sampling rate.
    - envelope: Computed peak envelope of the signal.
    - hop_length: Number of samples to jump between frames.

    Returns:
    - Plotly figure.
    """
    # Create time vectors for signal and envelope
    time_signal = np.linspace(0, len(signal) / sr, num=len(signal))
    time_envelope = np.arange(0, len(signal), hop_length) / sr

    # Create the figure
    fig = go.Figure()

    # Add traces for signal and envelope
    fig.add_trace(go.Scatter(x=time_signal, y=signal, mode="lines", name="Signal"))
    fig.add_trace(
        go.Scatter(
            x=time_envelope,
            y=envelope,
            mode="lines",
            name="Peak Envelope",
            line=dict(color="red", width=1.5),
        )
    )

    # Update layout for better visualization
    fig.update_layout(
        title="Peak Envelope of the Signal",
        xaxis_title="Time (seconds)",
        yaxis_title="Amplitude",
    )

    return fig


def plot_band_energy_ratio(ber_values, sr, hop_length):
    """
    Plot the Band Energy Ratio (BER) for multiple frames of the audio signal.

    Parameters:
    - ber_values: The computed Band Energy Ratio values.
    - sr: Sampling rate.
    - hop_length: Number of samples between successive frames.

    Returns:
    - Plotly figure.
    """

    # Time values for each frame center
    times = np.arange(len(ber_values)) * hop_length / sr

    # Plot
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=times, y=ber_values, mode="lines", name="Band Energy Ratio")
    )
    fig.update_layout(
        title="Band Energy Ratio (BER)",
        xaxis_title="Time (seconds)",
        yaxis_title="BER Value",
    )

    return fig


def plot_spectrogram(D, sr, hop_length):
    """
    Plot the magnitude spectrogram in decibels using plotly.

    Parameters:
    - D (np.ndarray): The magnitude spectrogram in decibels.
    - sr (int): The sampling rate of the audio.
    - hop_length (int): The number of samples between successive frames.
                        It is used to determine the time axis for plotting.

    Returns:
    - fig (plotly.graph_objects.Figure): The plotly figure object containing the spectrogram plot.
    """

    # Create time and frequency axes for plotting
    times = librosa.times_like(D, sr=sr, hop_length=hop_length)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2 * D.shape[0])

    # Create the figure using plotly
    fig = go.Figure(
        data=go.Heatmap(
            x=times, y=freqs, z=D, colorscale="Viridis", zmin=np.min(D), zmax=np.max(D)
        )
    )

    # Configure the layout
    fig.update_layout(
        title="Spectrogram",
        xaxis_title="Time (s)",
        yaxis_title="Frequency (Hz)",
        yaxis_type="log",
    )

    return fig


def plot_rms_energy(rms_energy, sr, hop_length):
    """
    Plot the RMS energy using plotly.

    Parameters:
    - rms_energy (np.ndarray): The RMS energy for each frame.
    - sr (int): The sampling rate of the audio.
    - hop_length (int): The number of samples between successive frames.
                        It is used to determine the time axis for plotting.

    Returns:
    - fig (plotly.graph_objects.Figure): The plotly figure object containing the RMS energy plot.
    """

    # Create time axis for plotting
    times = librosa.times_like(rms_energy, sr=sr, hop_length=hop_length)

    # Create the figure using plotly
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=times, y=rms_energy, mode="lines", name="RMS Energy"))

    # Configure the layout
    fig.update_layout(
        title="RMS Energy Over Time", xaxis_title="Time (s)", yaxis_title="Energy"
    )

    return fig


def plot_zero_crossing_rate(zcr, sr, hop_length):
    """
    Plot the zero crossing rate using plotly.

    Parameters:
    - zcr (np.ndarray): The zero crossing rate for each frame.
    - sr (int): The sampling rate of the audio.
    - hop_length (int): The number of samples between successive frames.
                        It is used to determine the time axis for plotting.

    Returns:
    - fig (plotly.graph_objects.Figure): The plotly figure object containing the ZCR plot.
    """

    # Create time axis for plotting
    times = librosa.times_like(zcr, sr=sr, hop_length=hop_length)

    # Create the figure using plotly
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=times, y=zcr, mode="lines", name="Zero Crossing Rate"))

    # Configure the layout
    fig.update_layout(
        title="Zero Crossing Rate Over Time",
        xaxis_title="Time (s)",
        yaxis_title="Zero Crossings",
    )

    return fig


def plot_spectral_centroid(spectral_centroid, sr, hop_length):
    """
    Plot the spectral centroid using plotly.

    Parameters:
    - spectral_centroid (np.ndarray): The spectral centroid for each frame.
    - sr (int): The sampling rate of the audio.
    - hop_length (int): The number of samples between successive frames.
                        It is used to determine the time axis for plotting.

    Returns:
    - fig (plotly.graph_objects.Figure): The plotly figure object containing the spectral centroid plot.
    """

    # Create time axis for plotting
    times = librosa.times_like(spectral_centroid, sr=sr, hop_length=hop_length)

    # Create the figure using plotly
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(x=times, y=spectral_centroid, mode="lines", name="Spectral Centroid")
    )

    # Configure the layout
    fig.update_layout(
        title="Spectral Centroid Over Time",
        xaxis_title="Time (s)",
        yaxis_title="Frequency (Hz)",
    )

    return fig


def plot_spectral_bandwidth(spectral_bandwidth, sr, hop_length):
    """
    Plot the spectral bandwidth using plotly.

    Parameters:
    - spectral_bandwidth (np.ndarray): The spectral bandwidth for each frame.
    - sr (int): The sampling rate of the audio.
    - hop_length (int): The number of samples between successive frames.
                        It is used to determine the time axis for plotting.

    Returns:
    - fig (plotly.graph_objects.Figure): The plotly figure object containing the spectral bandwidth plot.
    """

    # Create time axis for plotting
    times = librosa.times_like(spectral_bandwidth, sr=sr, hop_length=hop_length)

    # Create the figure using plotly
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=times, y=spectral_bandwidth, mode="lines", name="Spectral Bandwidth"
        )
    )

    # Configure the layout
    fig.update_layout(
        title="Spectral Bandwidth Over Time",
        xaxis_title="Time (s)",
        yaxis_title="Bandwidth (Hz)",
    )

    return fig


def plot_mel_spectrogram(mel_spectrogram, sr, hop_length, n_mels):
    """
    Plot the mel spectrogram using plotly.

    Parameters:
    - mel_spectrogram (np.ndarray): The mel spectrogram for each frame.
    - sr (int): The sampling rate of the audio.
    - hop_length (int): The number of samples between successive frames.
                        It is used to determine the time axis for plotting.

    Returns:
    - fig (plotly.graph_objects.Figure): The plotly figure object containing the mel spectrogram heatmap.
    """

    # Create time axis for plotting
    times = librosa.times_like(mel_spectrogram, sr=sr, hop_length=hop_length)

    # Create the figure using plotly
    fig = go.Figure(
        data=go.Heatmap(
            z=mel_spectrogram, x=times, colorscale="Viridis", colorbar={"title": "dB"}
        )
    )

    # Configure the layout
    fig.update_layout(
        title="Mel Spectrogram",
        xaxis_title="Time (s)",
        yaxis_title="Mel Frequency Bands",
        yaxis_nticks=n_mels,
    )

    return fig


def plot_mfcc(mfcc):
    """
    Plot the Mel-frequency Cepstral Coefficients (MFCCs).

    Parameters:
    - mfcc (numpy.ndarray): MFCC sequence.
    - hop_length (int): Number of samples between successive frames.

    Returns:
    - plotly.graph_objs._figure.Figure: Plotly figure object displaying the MFCCs.
    """
    fig = go.Figure(data=go.Heatmap(z=mfcc, colorscale="Viridis", zmin=-50, zmax=50))
    fig.update_layout(
        title="Mel-frequency Cepstral Coefficients (MFCCs)",
        xaxis_title="Time (frames)",
        yaxis_title="MFCC Coefficients",
    )
    return fig


def plot_first_delta_mfcc(delta_mfccs):
    """
    Plots the first delta (derivative) of the Mel-frequency cepstral coefficients (MFCCs) of a given audio signal using plotly.graph_objects.

    Parameters:
    - delta_mfccs (np.array): First delta MFCCs.
    - sr (int): The sample rate of the audio time series.
    - hop_length (int): Number samples between successive frames.

    Returns:
    - plotly.graph_objects.Figure: Figure object displaying the first delta MFCCs.
    """

    # Create heatmap using plotly.graph_objects
    fig = go.Figure(
        data=go.Heatmap(
            z=delta_mfccs,
            colorscale="Viridis",
            colorbar=dict(title="Amplitude"),
        )
    )

    # Update the layout
    fig.update_layout(
        title="First Delta MFCCs",
        xaxis_title="Time (frames)",
        yaxis_title="MFCC Coefficients",
    )

    return fig


def plot_second_delta_mfcc(delta2_mfccs):
    """
    Plots the second delta (acceleration) of the Mel-frequency cepstral coefficients (MFCCs) of a given audio signal using plotly.graph_objects.

    Parameters:
    - delta2_mfccs (np.array): Second delta MFCCs.
    - sr (int): The sample rate of the audio time series.
    - hop_length (int): Number samples between successive frames.

    Returns:
    - plotly.graph_objects.Figure: Figure object displaying the second delta MFCCs.
    """

    # Create heatmap using plotly.graph_objects
    fig = go.Figure(
        data=go.Heatmap(
            z=delta2_mfccs,
            colorscale="Viridis",
            colorbar=dict(title="Amplitude"),
        )
    )

    # Update the layout
    fig.update_layout(
        title="Second Delta MFCCs",
        xaxis_title="Time (frames)",
        yaxis_title="MFCC Coefficients",
    )

    return fig


def plot_cqt(cqt_db, sr, hop_length, bins_per_octave):
    """
    Plots the CQT matrix using plotly.

    Parameters:
    - cqt_db (np.array): CQT matrix in dB scale.
    - sr (int): The sample rate of the audio time series.
    - hop_length (int): Number of samples between successive frames.
    """

    time_axis = np.linspace(0, len(cqt_db[0]) * hop_length / sr, num=len(cqt_db[0]))
    fmin = 32.70  # Starting frequency (C1)
    freq_axis = librosa.cqt_frequencies(
        n_bins=cqt_db.shape[0], fmin=fmin, bins_per_octave=bins_per_octave
    )

    fig = go.Figure(
        data=go.Heatmap(
            z=cqt_db,
            x=time_axis,
            y=freq_axis,
            colorscale="Viridis",
            colorbar=dict(title="dB"),
        )
    )

    fig.update_layout(
        title="Constant-Q Transform (CQT)",
        xaxis_title="Time (s)",
        yaxis_title="Frequency (Hz)",
        yaxis_type="log",
    )

    return fig


def plot_gtcc(gtcc, hop_length, sr):
    """
    Plots the Gammatone Cepstral Coefficients (GTCC) using the plotly library.

    Parameters:
    - gtcc (np.array): GTCC matrix.
    - hop_length (int): Number of samples between successive frames.
    - sr (int): Sample rate of the audio signal.

    Returns:
    - plotly.graph_objects.Figure: A Plotly figure representing the GTCC.
    """

    # Get time and coefficient axes
    time_axis = np.arange(gtcc.shape[1]) * hop_length / sr
    coeff_axis = np.arange(gtcc.shape[0])

    # Create the figure
    fig = go.Figure(
        data=go.Heatmap(z=gtcc, x=time_axis, y=coeff_axis, colorscale="Viridis")
    )

    # Update the layout
    fig.update_layout(
        title="Gammatone Cepstral Coefficients (GTCC)",
        xaxis_title="Time (seconds)",
        yaxis_title="Cepstral Coefficients",
        yaxis=dict(autorange="reversed"),  # Flip the y-axis for better visualization
    )

    return fig


def plot_chroma(chroma, hop_length, sr):
    """
    Plots the chroma feature using Plotly.

    Parameters:
    - chroma (np.array): Chroma matrix.
    - hop_length (int): Number of samples between successive frames.
    - sr (int): Sample rate of the original signal.

    Returns:
    - plotly.graph_objects.Figure: Figure object ready to be shown or saved.
    """

    # Generate time axis
    times = librosa.times_like(chroma, hop_length=hop_length, sr=sr)

    fig = go.Figure()

    fig.add_trace(
        go.Heatmap(
            x=times,
            y=[
                str(i) for i in range(chroma.shape[0])
            ],  # Representing the 12 chroma bins
            z=chroma,
            colorscale="Viridis",
            colorbar=dict(title="Amplitude"),
        )
    )

    fig.update_layout(
        title="Chroma Feature",
        xaxis_title="Time (s)",
        yaxis_title="Chroma Bin",
        yaxis=dict(autorange="reversed"),  # Reverse y-axis for proper visualization
    )

    return fig


def plot_spectral_rolloff(rolloff, sr, hop_length):
    """
    Plots the spectral rolloff of an audio signal.

    Parameters:
    - rolloff (np.array): Spectral rolloff values.
    - sr (int): The sample rate of the audio time series.
    - hop_length (int): Number of samples between successive frames.

    Returns:
    - plotly.graph_objects.Figure: A plotly figure containing the spectral rolloff plot.
    """

    # Calculate the time values for each frame
    times = np.arange(rolloff.shape[1]) * hop_length / sr

    # Create the plot
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=times, y=rolloff[0], mode="lines", name="Spectral Rolloff")
    )

    fig.update_layout(
        title="Spectral Rolloff", xaxis_title="Time (s)", yaxis_title="Frequency (Hz)"
    )

    return fig
