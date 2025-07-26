import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import time

from simulation import setup_solver, animate_simulation, plot_kinetic_energy
from utils.ui_state import (
    init_state,
    start_running,
    stop_running,
    reset_state,
)



def maybe_rerun():
    """Call Streamlit's rerun function under any API name."""
    if hasattr(st, "experimental_rerun"):
        st.experimental_rerun()
    else:
        st.rerun()


def run_live_demo(ui):
    """Run the live demo without triggering Streamlit reruns."""
    solver = setup_solver(
        ui["n_particles"],
        ui["bounding_box_radius"],
        ui["sim_time"],
        ui["substeps"],
    )
    iterator = solver.run_simulation_iter(ui["sim_time"], int(ui["substeps"]))

    for _runtime, data in iterator:
        if st.session_state.get("fig_ax_scatter") is None:
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.set_aspect("equal")
            ax.set_facecolor("darkgray")
            circle = plt.Circle(
                (0, 0),
                ui["bounding_box_radius"],
                edgecolor="black",
                facecolor="white",
                fill=True,
            )
            ax.add_patch(circle)
            scatter = ax.scatter([], [], cmap="gist_rainbow", edgecolors="white", linewidth=0)
            st.session_state["fig_ax_scatter"] = (fig, ax, scatter)
        fig, ax, scatter = st.session_state["fig_ax_scatter"]
        fig.canvas.draw()
        px_per_scale = (
            ax.get_window_extent().width / (2 * ui["bounding_box_radius"] + 2) * 72.0 / fig.dpi
        )
        particles = np.vstack(data)
        scatter.set_offsets(particles[:, :2])
        scatter.set_array(np.linalg.norm(particles[:, 6:8], axis=1))
        scatter.set_sizes((px_per_scale * 2 * particles[:, 7]) ** 2)
        st.session_state["canvas_slot"].pyplot(fig)
        time.sleep(ui["speed"] / 60)

    reset_state()



def build_sidebar():
    """Return a dictionary with sidebar widget values."""
    st.sidebar.header("Simulation Parameters")
    ui = {
        "n_particles": st.sidebar.slider(
            "Number of particles", min_value=1, max_value=50, value=10
        ),
        "bounding_box_radius": st.sidebar.slider(
            "Bounding box radius", min_value=10, max_value=100, value=50
        ),
        "sim_time": st.sidebar.number_input(
            "Simulation time (s)", min_value=1.0, max_value=20.0, value=5.0
        ),
        "substeps": st.sidebar.number_input(
            "Substeps", min_value=10.0, max_value=2000.0, value=800.0
        ),
        "frame_skip": st.sidebar.number_input(
            "Display every n-th frame", min_value=1, max_value=50, value=1, step=1
        ),
        "speed": st.sidebar.slider(
            "Speed multiplier", min_value=1.0, max_value=2.0, value=1.0, step=0.1
        ),
        "run": st.sidebar.button("Run Simulation"),
        "live": st.sidebar.button("Live Demo"),
    }
    with st.sidebar:
        st.button("Play", on_click=start_running)
        st.button("Pause", on_click=stop_running)
        st.button("Reset", on_click=reset_state)
    return ui


def run_loop_step(ui):
    """Execute a single update-draw iteration."""
    if "iter" not in st.session_state:
        st.session_state["iter"] = st.session_state["solver"].run_simulation_iter(
            ui["sim_time"], int(ui["substeps"])
        )

    try:
        _, data = next(st.session_state["iter"])
    except StopIteration:
        stop_running()
        return False

    if st.session_state.get("fig_ax_scatter") is None:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_aspect("equal")
        ax.set_facecolor("darkgray")
        circle = plt.Circle(
            (0, 0),
            ui["bounding_box_radius"],
            edgecolor="black",
            facecolor="white",
            fill=True,
        )
        ax.add_patch(circle)
        scatter = ax.scatter([], [], cmap="gist_rainbow", edgecolors="white", linewidth=0)
        st.session_state["fig_ax_scatter"] = (fig, ax, scatter)
    fig, ax, scatter = st.session_state["fig_ax_scatter"]
    fig.canvas.draw()
    px_per_scale = (
        ax.get_window_extent().width / (2 * ui["bounding_box_radius"] + 2) * 72.0 / fig.dpi
    )
    particles = np.vstack(data)
    scatter.set_offsets(particles[:, :2])
    scatter.set_array(np.linalg.norm(particles[:, 6:8], axis=1))
    scatter.set_sizes((px_per_scale * 2 * particles[:, 7]) ** 2)
    st.session_state["canvas_slot"].pyplot(fig)
    return True


def main():
    st.title("Particle Simulation")
    init_state()
    ui = build_sidebar()

    if ui["run"]:
        st.session_state["solver"] = setup_solver(
            ui["n_particles"], ui["bounding_box_radius"], ui["sim_time"], ui["substeps"]
        )
        results = st.session_state["solver"].run_simulation(
            ui["sim_time"], ui["substeps"]
        )
        animation, kinetic = animate_simulation(
            results, ui["bounding_box_radius"], ui["sim_time"], ui["substeps"]
        )
        kinetic_fig = plot_kinetic_energy(kinetic)

        st.subheader("Animation")
        html_anim = animation.to_html5_video()
        st.components.v1.html(html_anim, height=500)
        html_anim = animation.to_jshtml()
        st.components.v1.html(html_anim, height=400)

        st.subheader("Kinetic Energy")
        st.pyplot(kinetic_fig)

    if ui["live"]:
        run_live_demo(ui)

    while st.session_state.get("running", False):
        if not run_loop_step(ui):
            break

        time.sleep(ui["speed"] / 60)
        maybe_rerun()


if __name__ == "__main__":
    main()
