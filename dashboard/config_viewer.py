"""Config display widgets for the AFLF dashboard."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from dashboard.data_loader import load_available_configs, load_config_content


def render_config_viewer(default_path: str | None = None) -> None:
    st.subheader("Config Viewer")

    config_paths = load_available_configs()
    if not config_paths:
        st.info("No config files found")
        return

    path_strings = [str(path) for path in config_paths]
    default_index = 0
    if default_path and default_path in path_strings:
        default_index = path_strings.index(default_path)

    selected = st.selectbox("Select config", options=path_strings, index=default_index)
    payload = load_config_content(selected)

    if payload is None:
        st.info("No data found")
        return

    c1, c2 = st.columns([1, 2])
    c1.caption("Config file")
    c1.write(Path(selected).name)

    c2.caption("YAML content")
    st.json(payload, expanded=False)
