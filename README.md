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
