#!/usr/bin/env python3
"""Regenerate examples/porous/porous_contours.png from the existing
   cells.csv outputs (without re-solving). Use this after editing the
   visualization in run_and_plot.py."""

import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

# Import the plot-side helpers from run_and_plot.
from run_and_plot import (  # noqa: E402
    CURVES, CURVE_BY_TAG, parse_msh_curved_edges, plot_mesh,
    mask_pores_polygonal, mask_pores_circular,
    per_vertex_average, load_cells_csv,
    SMBDG_MESH, AGEDG_MESH, TAU_R, DG_ORDER,
)

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri


def main() -> None:
   output_dir = THIS_DIR / "output"

   v_s, t_s, T_s, _, _ = load_cells_csv(output_dir / "smbdg_porous_cells.csv")
   v_a, t_a, T_a, _, _ = load_cells_csv(output_dir / "agedg_porous_cells.csv")

   amp = max(abs(T_s.min()), abs(T_s.max()), abs(T_a.min()), abs(T_a.max()))
   amp = math.ceil(amp * 10) / 10
   levels = np.linspace(-amp, amp, 21)

   # SMBDG_MESH/AGEDG_MESH in run_and_plot.py are paths relative to the
   # generated YAML in output/. From this directory they're just the
   # bare mesh filenames.
   smb_curved = parse_msh_curved_edges(THIS_DIR / Path(SMBDG_MESH).name)
   age_curved = parse_msh_curved_edges(THIS_DIR / Path(AGEDG_MESH).name)
   print(f"  SMBDG curved edges: {len(smb_curved)}")
   print(f"  AGEDG curved edges: {len(age_curved)}")

   fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

   for ax, verts, tris, Tcell, mesh_curved_edges, pore_mask_kind, label in (
      (axes[0], v_s, t_s, T_s, smb_curved, "polygon",
       f"SMBDG  /  {len(t_s)} elements (dense straight-sided mesh)"),
      (axes[1], v_a, t_a, T_a, age_curved, "circle",
       f"AGEDG  /  {len(t_a)} elements (sparse AGE mesh)"),
   ):
      tri = mtri.Triangulation(verts[:, 0], verts[:, 1], tris)
      v_vals = per_vertex_average(tris, Tcell, len(verts))
      im = ax.tricontourf(tri, v_vals, levels=levels, cmap="coolwarm", extend="both")
      ax.tricontour(tri, v_vals, levels=levels,
                    colors="k", linewidths=0.4, alpha=0.55)
      if pore_mask_kind == "polygon":
         mask_pores_polygonal(ax, verts, mesh_curved_edges)
      else:
         mask_pores_circular(ax)
      plot_mesh(ax, verts, tris,
                {} if pore_mask_kind == "polygon" else mesh_curved_edges,
                mesh_curved_edges,
                color="0.4", linewidth=0.2, alpha=0.5,
                boundary_color="0.0", boundary_linewidth=1.0,
                boundary_alpha=1.0)
      ax.set_aspect("equal")
      ax.set_title(label)
      plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                   ticks=np.linspace(-amp, amp, 5))

   for ax in axes:
      ax.set_xlim(-1, 1)
      ax.set_ylim(-1, 1)
      ax.set_xticks([])
      ax.set_yticks([])

   fig.suptitle(
      f"Porous medium (paper §5.2), τ_R = τ_N = {TAU_R}, DG order {DG_ORDER}, "
      f"periodic ΔT = 1 (top/bottom), specular L/R, diffuse pores",
      y=1.00, fontsize=11)
   plt.tight_layout()
   plot_path = THIS_DIR / "porous_contours.png"
   plt.savefig(plot_path, dpi=160, bbox_inches="tight")
   print(f"Wrote {plot_path}")


if __name__ == "__main__":
   main()
