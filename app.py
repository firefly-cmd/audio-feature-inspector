import streamlit as st
import librosa
from audio_analyzer import AudioAnalyzer
import base64


# Load the image as base64
def get_image_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


def audio_explorer(file_uploader_key, plot_button, params):
    # Create a audio file loader
    # Use the file uploader widget to allow user to upload an audio file
    uploaded_audio_file = st.file_uploader(
        "Choose an audio file",
        type=["wav"],  # TODO Add support for other filetypes as well
        key=file_uploader_key,
    )

    if uploaded_audio_file is not None:
        st.write("Listen to the audio")
        # Create a widget to playback the sound loaded
        st.audio(
            uploaded_audio_file, format="audio/wav"
        )  # adjust the format according to your needs

    # If a file is uploaded, display it using the audio widget
    if uploaded_audio_file is not None and plot_button:
        # Load file in a usable format
        audio_data, sr = librosa.load(
            uploaded_audio_file, sr=None
        )  # TODO Make this sampling rate constant if necessary

        # Initialize the audio analyzer object
        audio_analyzer = AudioAnalyzer(signal=audio_data, sr=sr, params=params)

        # Create plots based on user selection
        if "Amplitude Envelope" in selected_features:
            audio_analyzer.amplitude_envelope()

        if "Band Energy Ratio" in selected_features:
            audio_analyzer.band_energy_ratio()

        if "Spectogram" in selected_features:
            audio_analyzer.spectogram()

        if "RMS Energy" in selected_features:
            audio_analyzer.rms_energy()

        if "Zero Crossing Rate" in selected_features:
            audio_analyzer.zero_crossing_rate()

        if "Spectral Centroid" in selected_features:
            audio_analyzer.spectral_centroid()

        if "Spectral Bandwidth" in selected_features:
            audio_analyzer.spectral_bandwidth()

        if "Mel Spectogram" in selected_features:
            audio_analyzer.mel_spectogram()

        if "MFCC" in selected_features:
            audio_analyzer.mfcc()

        if "MFCC First Delta" in selected_features:
            audio_analyzer.mfcc_first_delta()

        if "MFCC Second Delta" in selected_features:
            audio_analyzer.mfcc_second_delta()

        if "CQT" in selected_features:
            audio_analyzer.cqt()

        if "Gammatone Cepstral Coefficients" in selected_features:
            audio_analyzer.gtcc()

        if "Chroma" in selected_features:
            audio_analyzer.chroma()

        if "Spectral Rolloff" in selected_features:
            audio_analyzer.spectral_rolloff()


if __name__ == "__main__":
    # Set the app layout configs
    st.set_page_config(
        page_title="Audio Data Inspector",
        page_icon="🎵",
        layout="wide",
    )

    st.title("WELCOME TO AUDIO INSPECTOR")
    st.markdown(
        """
                ### In order to draw plots, first load a sound file below. Then, select which features needs to be plotted from the sidebar and adjust necessary parameters. Click on the Plot data button from the sidebar and the selected feature plots would be represented.
        """
    )

    # A button to trigger the plotting
    plot_button = st.button("Plot Data", use_container_width=True)

    # Create tabs for the app
    individual_sound_explorer_tab, compare_two_tracks_tab = st.tabs(
        ["INDIVIDUAL SOUND EXPLORER", "COMPARE 2 TRACKS"]
    )

    # Initialize the parameters that will be used in this project
    PARAM_CONFIGS = {
        "frame_length": {
            "label": "Frame Length",
            "min_value": 512,
            "max_value": 8192,
            "value": 2048,
            "step": 512,
        },
        "hop_length": {
            "label": "Hop Length",
            "min_value": 256,
            "max_value": 4096,
            "value": 512,
            "step": 256,
        },
        "split_frequency": {
            "label": "Split Frequency (Hz)",
            "min_value": 500,
            "max_value": 5000,
            "value": 1000,
            "step": 100,
        },
        "n_fft": {
            "label": "n_fft",
            "min_value": 256,
            "max_value": 8192,
            "value": 2048,
            "step": 256,
        },
        "n_mels": {
            "label": "Number of Mel bands",
            "min_value": 10,
            "max_value": 128,
            "value": 64,
            "step": 1,
        },
        "n_mfcc": {
            "label": "Number of MFCCs",
            "min_value": 5,
            "max_value": 40,
            "value": 13,
            "step": 1,
        },
        "bins_per_octave": {
            "label": "Bins per Octave",
            "min_value": 1,
            "max_value": 48,
            "value": 12,
            "step": 1,
        },
        "n_bins": {
            "label": "Number of CQT Bins",
            "min_value": 1,
            "max_value": 300,
            "value": 84,
            "step": 1,
        },
        "n_chroma": {
            "label": "Number of Chroma Bins",
            "min_value": 1,
            "max_value": 24,
            "value": 12,
            "step": 1,
        },
        "roll_percent": {
            "label": "Spectral Rolloff Percentage",
            "min_value": 0.1,
            "max_value": 0.99,
            "value": 0.85,
            "step": 0.01,
        },
    }

    with st.sidebar:
        # Select features to plot
        features = [
            "Amplitude Envelope",
            "Band Energy Ratio",
            "Spectogram",
            "RMS Energy",
            "Zero Crossing Rate",
            "Spectral Centroid",
            "Spectral Bandwidth",
            "Mel Spectogram",
            "MFCC",
            "MFCC First Delta",
            "MFCC Second Delta",
            "CQT",
            "Gammatone Cepstral Coefficients",
            "Chroma",
            "Spectral Rolloff",
        ]
        selected_features = st.multiselect("SELECT FEATURES TO PLOT", features)

        st.title("ADJUST PARAMETERS")

        params = {}
        for param, config in PARAM_CONFIGS.items():
            params[param] = st.slider(
                label=config["label"],
                min_value=config["min_value"],
                max_value=config["max_value"],
                value=config["value"],
                step=config["step"],
            )

        params["window_type"] = st.sidebar.selectbox(
            "Window Function", ["hann", "hamming", "blackman", "bartlett"], index=0
        )

    with individual_sound_explorer_tab:
        audio_explorer(
            file_uploader_key="Single Track", plot_button=plot_button, params=params
        )

    with compare_two_tracks_tab:
        # Divide the tab into 2 columns
        col1, col2 = st.columns(2)

        with col1:
            audio_explorer(
                file_uploader_key="First Track", plot_button=plot_button, params=params
            )

        with col2:
            audio_explorer(
                file_uploader_key="Second Track", plot_button=plot_button, params=params
            )
