import streamlit as st
from simulation import setup_solver, animate_simulation, plot_kinetic_energy

st.title("Particle Simulation")

st.sidebar.header("Simulation Parameters")

n_particles = st.sidebar.slider("Number of particles", min_value=1, max_value=50, value=10)
bounding_box_radius = st.sidebar.slider("Bounding box radius", min_value=10, max_value=100, value=50)
sim_time = st.sidebar.number_input("Simulation time (s)", min_value=1.0, max_value=20.0, value=5.0)
substeps = st.sidebar.number_input("Substeps", min_value=10.0, max_value=2000.0, value=800.0)

run = st.sidebar.button("Run Simulation")

if run:
    solver = setup_solver(n_particles, bounding_box_radius, sim_time, substeps)
    results = solver.run_simulation(sim_time, substeps)
    animation, kinetic = animate_simulation(results, bounding_box_radius, sim_time, substeps)
    kinetic_fig = plot_kinetic_energy(kinetic)

    st.subheader("Animation")
    html_anim = animation.to_html5_video()
    st.components.v1.html(html_anim, height=500)

    st.subheader("Kinetic Energy")
    st.pyplot(kinetic_fig)
