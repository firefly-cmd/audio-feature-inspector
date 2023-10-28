import streamlit as st

from audio_preprocessing import (
    calculate_amplitude_envelope,
    calculate_band_energy_ratio,
    calculate_spectrogram,
    calculate_rms_energy,
    calculate_zero_crossing_rate,
    calculate_spectral_centroid,
    calculate_spectral_bandwidth,
    calculate_mel_spectrogram,
    calculate_mfcc,
    calculate_first_delta_mfcc,
    calculate_second_delta_mfcc,
    calculate_cqt,
    calculate_gtcc,
    calculate_chroma,
    calculate_spectral_rolloff,
)

from plotters import (
    plot_peak_envelope,
    plot_band_energy_ratio,
    plot_spectrogram,
    plot_rms_energy,
    plot_zero_crossing_rate,
    plot_spectral_centroid,
    plot_spectral_bandwidth,
    plot_mel_spectrogram,
    plot_mfcc,
    plot_first_delta_mfcc,
    plot_second_delta_mfcc,
    plot_cqt,
    plot_gtcc,
    plot_chroma,
    plot_spectral_rolloff,
)


class AudioAnalyzer:
    def __init__(self, signal, sr, params) -> None:
        self.signal = signal
        self.sr = sr
        self.params = params

    def amplitude_envelope(self):
        # Compute amplitude envelope
        amp_env = calculate_amplitude_envelope(
            self.signal, self.params["frame_length"], self.params["hop_length"]
        )

        # Plot the amplitude envelope with the original signal
        fig = plot_peak_envelope(
            self.signal, self.sr, amp_env, self.params["hop_length"]
        )

        # Display the plot on streamlit
        st.plotly_chart(fig, use_container_width=True)

    def band_energy_ratio(self):
        # Compute band energy ratio #TODO Add n_fft parameter if necessary
        ber_values = calculate_band_energy_ratio(
            self.signal,
            self.sr,
            self.params["frame_length"],
            self.params["hop_length"],
            self.params["split_frequency"],
        )

        # Plot the band energy ratio
        fig = plot_band_energy_ratio(ber_values, self.sr, self.params["hop_length"])

        # Display the plot on streamlit
        st.plotly_chart(fig, use_container_width=True)

    def spectogram(self):
        # Calculate the spectogram
        spec = calculate_spectrogram(
            self.signal,
            self.params["n_fft"],
            self.params["hop_length"],
            window=self.params["window_type"],
        )

        # Generate the plot
        fig = plot_spectrogram(spec, self.sr, self.params["hop_length"])

        # Display the plot in Streamlit
        st.plotly_chart(fig, use_container_width=True)

    def rms_energy(self):
        # Calculate rms energy
        rms = calculate_rms_energy(
            self.signal, self.params["frame_length"], self.params["hop_length"]
        )

        # Generate the plot
        fig = plot_rms_energy(rms, self.sr, self.params["hop_length"])

        # Display the plot in Streamlit
        st.plotly_chart(fig, use_container_width=True)

    def zero_crossing_rate(self):
        # Calculate zero crossing rate
        zcr = calculate_zero_crossing_rate(
            self.signal, self.params["frame_length"], self.params["hop_length"]
        )

        # Generate the plot
        fig = plot_zero_crossing_rate(zcr, self.sr, self.params["hop_length"])

        # Display the plot in Streamlit
        st.plotly_chart(fig, use_container_width=True)

    def spectral_centroid(self):
        # Calculate spectral centroid
        spectral_centroid = calculate_spectral_centroid(
            self.signal, self.sr, self.params["frame_length"], self.params["hop_length"]
        )

        # Generate the plot
        fig = plot_spectral_centroid(
            spectral_centroid, self.sr, self.params["hop_length"]
        )

        # Display the plot in Streamlit
        st.plotly_chart(fig, use_container_width=True)

    def spectral_bandwidth(self):
        # Calculate spectral bandwidth
        spectral_bandwidth = calculate_spectral_bandwidth(
            self.signal, self.sr, self.params["frame_length"], self.params["hop_length"]
        )

        # Generate the plot
        fig = plot_spectral_bandwidth(
            spectral_bandwidth, self.sr, self.params["hop_length"]
        )

        # Display the plot in Streamlit
        st.plotly_chart(fig, use_container_width=True)

    def mel_spectogram(self):
        # Calculate mel spectogram
        mel_spectrogram = calculate_mel_spectrogram(
            self.signal,
            self.sr,
            self.params["frame_length"],
            self.params["hop_length"],
            self.params["n_mels"],
        )

        # Generate the plot
        fig = plot_mel_spectrogram(
            mel_spectrogram, self.sr, self.params["hop_length"], self.params["n_mels"]
        )

        # Display the plot in Streamlit
        st.plotly_chart(fig, use_container_width=True)

    def mfcc(self):
        # Calculate mfcc
        mfcc = calculate_mfcc(
            self.signal,
            self.sr,
            self.params["n_mfcc"],
            self.params["n_fft"],
            self.params["hop_length"],
        )
        # Generate plot
        fig = plot_mfcc(mfcc)

        # Display the plot in Streamlit
        st.plotly_chart(fig, use_container_width=True)

    def mfcc_first_delta(self):
        # Calculate mfcc first delta
        delta_mfccs = calculate_first_delta_mfcc(
            self.signal,
            self.sr,
            self.params["n_mfcc"],
            self.params["n_fft"],
            self.params["hop_length"],
            self.params["n_mels"],
        )

        # Generate plot
        fig = plot_first_delta_mfcc(delta_mfccs)

        # Display the plot in Streamlit
        st.plotly_chart(fig, use_container_width=True)

    def mfcc_second_delta(self):
        # Calculate mfcc first delta
        delta_mfccs = calculate_second_delta_mfcc(
            self.signal,
            self.sr,
            self.params["n_mfcc"],
            self.params["n_fft"],
            self.params["hop_length"],
            self.params["n_mels"],
        )

        # Generate plot
        fig = plot_second_delta_mfcc(delta_mfccs)

        # Display the plot in Streamlit
        st.plotly_chart(fig, use_container_width=True)

    # TODO Add an fmin parameter if necessary
    def cqt(self):
        # Calculate cqt
        cqt_db = calculate_cqt(
            self.signal,
            self.sr,
            self.params["hop_length"],
            self.params["n_bins"],
            self.params["bins_per_octave"],
        )

        # Generate plot
        fig = plot_cqt(
            cqt_db, self.sr, self.params["hop_length"], self.params["bins_per_octave"]
        )

        # Display the plot in Streamlit
        st.plotly_chart(fig, use_container_width=True)

    def gtcc(self):
        # Calculate gtcc
        gtcc = calculate_gtcc(
            self.signal,
            self.sr,
            self.params["hop_length"],
            self.params["n_fft"],
            self.params["n_mels"],
            self.params["n_mfcc"],
        )

        # Generate plot
        fig = plot_gtcc(gtcc, self.params["hop_length"], self.sr)

        # Display the plot in Streamlit
        st.plotly_chart(fig, use_container_width=True)

    def chroma(self):
        # Calculate chroma
        chroma = calculate_chroma(
            self.signal,
            self.sr,
            self.params["hop_length"],
            self.params["n_fft"],
            self.params["n_chroma"],
        )

        # Generate plot
        fig = plot_chroma(chroma, self.params["hop_length"], self.sr)

        # Display the plot in Streamlit
        st.plotly_chart(fig, use_container_width=True)

    def spectral_rolloff(self):
        # Calculate spectral rolloff
        rolloff = calculate_spectral_rolloff(
            self.signal,
            self.sr,
            self.params["frame_length"],
            self.params["hop_length"],
            self.params["roll_percent"],
        )

        # Generate plot
        fig = plot_spectral_rolloff(rolloff, self.sr, self.params["hop_length"])

        # Display the plot in Streamlit
        st.plotly_chart(fig, use_container_width=True)
