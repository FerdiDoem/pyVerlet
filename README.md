# pyVerlet
A simple particle simulation

## Streamlit Application

Run the web interface with:

```bash
streamlit run streamlit_app.py
```

### Collision Detection

`Solver` uses a spatial hash grid for neighbour search by default. Pass
`use_grid=False` to fall back to the slower KD-tree implementation.

## How to benchmark

Run the profiler with:

```bash
python profile_runner.py --n_particles 100 --substeps 1000 --cell_size 2
```

This writes `stats.prof` which can be inspected with `snakeviz` or `pstats`.

Run a parameter sweep (prints FPS):

```bash
python sweep.py --particles 50 100 --substeps 200 500 --cell_sizes 1 2
```

