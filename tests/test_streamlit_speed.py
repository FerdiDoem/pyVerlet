import streamlit as st
import simulation
from streamlit_app import run_loop_step
from utils.ui_state import init_state


def test_streamlit_loop_advances_time():
    st.session_state.clear()
    init_state()
    solver = simulation.setup_solver(n_particles=1, bounding_box_radius=10, time=0.2, substeps=2)
    st.session_state['solver'] = solver
    st.session_state['running'] = True
    ui = {
        'sim_time': 0.2,
        'substeps': 2,
        'bounding_box_radius': 10,
        'speed': 1.0,
    }
    st.session_state['iter'] = solver.run_simulation_iter(ui['sim_time'], int(ui['substeps']))
    before = solver.runtime
    run_loop_step(ui)
    after = solver.runtime
    assert after > before
