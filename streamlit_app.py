import streamlit as st
import matplotlib.pyplot as plt
from simulation import setup_solver, animate_simulation, plot_kinetic_energy

st.title("Particle Simulation")

st.sidebar.header("Simulation Parameters")

n_particles = st.sidebar.slider("Number of particles", min_value=1, max_value=50, value=10)
bounding_box_radius = st.sidebar.slider("Bounding box radius", min_value=10, max_value=100, value=50)
sim_time = st.sidebar.number_input("Simulation time (s)", min_value=1.0, max_value=20.0, value=5.0)
substeps = st.sidebar.number_input("Substeps", min_value=10.0, max_value=2000.0, value=800.0)
frame_skip = st.sidebar.number_input(
    "Display every n-th frame", min_value=1, max_value=50, value=1, step=1
)

run = st.sidebar.button("Run Simulation")
live = st.sidebar.button("Live Demo")

if run:
    solver = setup_solver(n_particles, bounding_box_radius, sim_time, substeps)
    results = solver.run_simulation(sim_time, substeps)
    animation, kinetic = animate_simulation(results, bounding_box_radius, sim_time, substeps)
    kinetic_fig = plot_kinetic_energy(kinetic)

    st.subheader("Animation")
    html_anim = animation.to_html5_video()
    st.components.v1.html(html_anim, height=500)
    html_anim = animation.to_jshtml()
    st.components.v1.html(html_anim, height=400)

    st.subheader("Kinetic Energy")
    st.pyplot(kinetic_fig)

if live:
    solver = setup_solver(n_particles, bounding_box_radius, sim_time, substeps)
    placeholder = st.empty()
    progress = st.progress(0.0)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_aspect("equal")
    ax.set_facecolor("darkgray")
    circle = plt.Circle((0, 0), bounding_box_radius,
                        edgecolor="black",
                        facecolor="white",
                        fill=True)
    ax.add_patch(circle)
    fig.canvas.draw()
    px_per_scale = (ax.get_window_extent().width /
                    (2 * bounding_box_radius + 2) * 72.0 / fig.dpi)
    scatter = ax.scatter([], [], cmap="gist_rainbow",
                         edgecolors="white", linewidth=0)

    for step, (_, data) in enumerate(
            solver.run_simulation_iter(sim_time, int(substeps))):
        particles = np.vstack(data)
        if step % int(frame_skip) == 0:
            scatter.set_offsets(particles[:, :2])
            scatter.set_array(np.linalg.norm(particles[:, 6:8], axis=1))
            scatter.set_sizes((px_per_scale * 2 * particles[:, 7]) ** 2)
            placeholder.pyplot(fig)
        progress.progress((step + 1) / substeps)
    plt.close(fig)
