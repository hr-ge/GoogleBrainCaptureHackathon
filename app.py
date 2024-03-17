# app.py
import streamlit as st
import pandas as pd
import sys
import os
sys.path.append(os.getcwd() + '/')
from src.data.utils.eeg import get_raw
from src.data.processing import load_data_dict, get_data, normalize_and_add_scaling_channel
from src.data.conf.eeg_annotations import braincapture_annotations
from src.data.conf.eeg_channel_picks import hackathon
from src.data.conf.eeg_channel_order import standard_19_channel
from src.data.conf.eeg_annotations import braincapture_annotations, tuh_eeg_artefact_annotations
from sklearn import datasets
from braindecode.classifier import EEGClassifier
import mne

max_length = lambda raw : int(raw.n_times / raw.info['sfreq']) 
DURATION = 60
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'

class viz_raw():
    def __init__(self, path):
        self.raw = mne.io.read_raw(path, preload=True)


    def plot_raw(self):
        return self.raw.plot()
    
    
    def epoch_viz(self):
        events, event_id = mne.events_from_annotations(self.raw)
        epochs = mne.Epochs(self.raw, events, event_id, tmin=-0.3, tmax=0.7, preload=True)
        return epochs.plot()


def load_model(device='cpu'):
    """Loading ShallowFBCSPNet model
    Args:
        device (str): The device to be used.
    Returns:
        ShallowFBCSPNet (braindecode.classifier.EEGClassifier): The model
    """

    # Initialize the model
    model = EEGClassifier(
    'ShallowFBCSPNet',
    module__final_conv_length='auto',
    module__n_times=1537,
    module__n_chans=20,
    module__n_outputs=5,
    device = "cpu",
)

    # Initialize it with the parameters of the best checkpoint:
    model.initialize()
    model.load_params('checkpoint/params.pt')

    return model

def make_dir(dir):
    try:
        os.makedirs(dir)
    except FileExistsError:
        pass

def get_file_paths(edf_file_buffers):
    """
    input: edf_file_buffers: list of files uploaded by user

    output: paths: paths to the files
    """
    paths = []
    # make tempoary directory to store the files
    temp_dir = tempfile.mkdtemp()
    print(temp_dir)
    for edf_file_buffer in edf_file_buffers:
        folder_name = os.path.join(temp_dir, edf_file_buffer.name[:4])
        make_dir(folder_name)
        # make tempoary file
        path = os.path.join(folder_name , edf_file_buffer.name)
        # write bytesIO object to file
        with open(path, 'wb') as f:
            f.write(edf_file_buffer.getvalue())

        paths.append(path)

    return temp_dir + '/', paths

def get_raw_data(path):
    raw = get_raw(path)
    raw_norm = normalize_and_add_scaling_channel(raw)
    return raw_norm

def get_annotations(raw_norm, model):
    info = mne.create_info(ch_names=20, sfreq=256, ch_types='eeg')
    epochs = mne.EpochsArray(raw_norm, info=info)
    probs = np.array(model.predict_proba(epochs))  

    annotations = probs.argmax(axis=-1)
    return annotations

def plot_dscnn_annotations(annotations):
    label_dict = {'Eye blinking': 0, 'Eye movement left-right': 1, 'Eyes closed': 2, 'Eyes opened': 3, 'Jaw clenching': 4}
    color_dict = {
        0: np.array([255, 255, 0]),  # yellow
        1: np.array([255, 0, 0]),  # red
        2: np.array([50, 205, 50]),  # lime
        3: np.array([255, 160, 0]),  # orange
        4: np.array([100, 149, 237]),  # cornflowerblue
    }
    names = list(label_dict.keys())
    colors = ['yellow', 'red', 'lime', 'orange', 'cornflowerblue']
    
    annotations_RGB = np.array([color_dict[label] for label in annotations])
    x = np.linspace(0, len(annotations), len(annotations))
    colorsx = [colors[int(item)] for item in annotations]

    fig, ax = plt.subplots(2, 1, figsize=(30, 8))
    ax[0].scatter(x, annotations, c=colorsx, s=100)
    ax[0].set_yticks([0, 1, 2, 3, 4])
    ax[0].set_yticklabels(names, fontsize=24)
    ax[0].set_xlim(0, len(annotations))

    ax[1].imshow(np.expand_dims(annotations_RGB, axis=0), interpolation='nearest', aspect='auto')
    formatter = matplotlib.ticker.FuncFormatter(lambda pos, _: str(datetime.timedelta(seconds=int(pos * 2))))
    
    for i in range(2):
        ax[i].set_xticks(np.linspace(0, len(annotations), 10, dtype=float))
        ax[i].xaxis.set_major_formatter(formatter)
        ax[i].tick_params(axis='x', labelsize=20)

    patches = [mpatches.Patch(color=color, label=name) for color, name in zip(colors, names)]
    ax[0].legend(handles=patches, bbox_to_anchor=(1.05, 1.0), loc='upper left', fontsize=24)
    ax[1].set_yticklabels([])
    plt.suptitle('Predicted Artifacts Labels', size=32)
    plt.show()


st.title("EEG Data Visualization and Annotation")

# File uploader
st.subheader("Upload your EEG data files")
uploaded_files = st.file_uploader("Choose a file...", accept_multiple_files=True)

if uploaded_files:
    temp_dir, paths = get_file_paths(uploaded_files)
    st.success(f"Files uploaded to {temp_dir}")
    
    # Select a file to visualize
    file_to_visualize = st.selectbox("Select a file to visualize", options=paths)

    # Visualization section
    if st.button("Visualize Raw Data"):
        viz = viz_raw(file_to_visualize)
        st.pyplot(viz.plot_raw())
    
    if st.button("Visualize Epochs"):
        viz = viz_raw(file_to_visualize)
        st.pyplot(viz.epoch_viz())

    # Load and display model annotations
    model_loaded = load_model()
    if st.button("Display Model Annotations"):
        raw_norm = get_raw_data(file_to_visualize)
        annotations = get_annotations(raw_norm, model_loaded)
        plot_dscnn_annotations(annotations)
    