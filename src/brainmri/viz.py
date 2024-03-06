import plotly.express as px

from brainmri.core import intensity_to_name


def volume_histogram(volumes, intensities):

    fig = px.bar(
        x=[intensity_to_name[str(int(i))] for i in intensities],
        y=volumes,
        title="Volume per Intensity",
    )
    fig.update_layout(
        xaxis_title="Intensity",
        yaxis_title="Volume (ml)",
        showlegend=False,
    )
    return fig
