import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import os, sys, inspect
import femm

def read_ans_with_mesh(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # find solution nodes
    sol_start = next(i for i, line in enumerate(lines) if '[Solution]' in line) + 1
    npts = int(lines[sol_start].split()[0])
    node_lines = lines[sol_start + 1 : sol_start + 1 + npts]

    xs, ys, Ts = [], [], []
    for line in node_lines:
        parts = line.split()
        if len(parts) >= 3:
            x, y, T = map(float, parts[:3])
            xs.append(x)
            ys.append(y)
            Ts.append(T)
    xs, ys, Ts = np.array(xs), np.array(ys), np.array(Ts)

    # find elements (triangles for meshing)
    elem_start = sol_start + 1 + npts
    n_elems = int(lines[elem_start].split()[0])
    elem_lines = lines[elem_start + 1 : elem_start + 1 + n_elems]

    elems = []
    for line in elem_lines:
        parts = line.split()
        if len(parts) >= 3:
            # first three are node indices
            elems.append([int(parts[0]), int(parts[1]), int(parts[2])])
    elems = np.array(elems)

    return xs, ys, Ts, elems


def dessiner_ligne_de_champ(xs, ys, Ts, elems, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
        show = True
    else:
        show = False

    # -------- draw FEMM mesh directly --------
    triang = tri.Triangulation(xs, ys, elems)

    # Lignes de champ (toutes fines et identiques)
    niveaux = 20  # nombre d'isolignes
    cs = ax.tricontour(
        triang, Ts, niveaux,
        colors="k",           # noir
        linestyles=":",       # pointillées
        linewidths=0.9        # épaisseur fine
    )

    # ax.clabel(cs, inline=True, fontsize=8)

    if show:
        plt.show()

    return ax

def dessiner_echauffement(xs, ys, Ts, elems, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
        show = True
    else:
        show = False

    # Création du maillage triangulaire
    triang = tri.Triangulation(xs, ys, elems)

    # Nombre de niveaux de couleur
    niveaux = 50

    # Affichage du champ avec une colormap
    tcf = ax.tricontourf(
        triang, Ts, niveaux,
        cmap="plasma"  # tu peux essayer aussi "viridis", "inferno", "coolwarm", etc.
    )


    # Ajout d’une barre de couleur
    cbar = plt.colorbar(tcf, ax=ax)
    cbar.set_label("Température (°C)", rotation=270, labelpad=15)

    # Mise en forme du graphique
    ax.set_title("Champ de température (solution FEMM)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")

    if show:
        plt.show()

    return ax


def solve_feh_file(feh_path):
    # 1️⃣ Ouvrir FEMM
    femm.openfemm()

    # 2️⃣ Charger le fichier .feh
    femm.opendocument(feh_path)

    # 3️⃣ Lancer la résolution
    femm.hi_analyze()

    # 4️⃣ Charger le résultat pour post-traitement (facultatif)
    femm.hi_loadsolution()

    # 5️⃣ Fermer le document et FEMM
    femm.closefemm()

# file_paths = [
#     r"C:\Users\valen\OneDrive - Moonwatt\Calcul.FEH",
#     r"C:\Users\valen\OneDrive - Moonwatt\Calcul_convections.FEH",
#     r"C:\Users\valen\OneDrive - Moonwatt\Calcul_convections_plus.FEH",
#     r"C:\Users\valen\OneDrive - Moonwatt\Calcul_convections_plus_no_bridge.FEH",
# ]
# for feh_path in file_paths:
#     solve_feh_file(feh_path)


