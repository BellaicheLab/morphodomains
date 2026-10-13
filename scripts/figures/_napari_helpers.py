"""Shared presentation helpers for napari scene builders."""


def reformat_time_legends(viewer):
    """Rewrite time-legend layer text from 'HHhMM' to 'HH:MM hAPF'.

    Call at the end of a scene builder after all tensor-map layers
    have been added. No-op on viewers without time-legend layers.
    """
    for layer in viewer.layers:
        if layer.name == "time legend":
            features = layer.features.copy()
            features["beginning_interval"] = [
                v.replace("h", ":") for v in features["beginning_interval"]
            ]
            features["ending_interval"] = [
                v.replace("h", ":") + " hAPF" for v in features["ending_interval"]
            ]
            layer.features = features
        elif layer.name == "median time legend":
            features = layer.features.copy()
            features["median_time"] = [
                v.replace("h", ":") + " hAPF" for v in features["median_time"]
            ]
            layer.features = features
