import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import re
from femm import *


def hi_setbound(x, y, boundName):
    hi_selectsegment(x, y)
    hi_setsegmentprop(boundName, 0, 1, 0, 0, "<None>")
    hi_clearselected()

def hi_addRectangle(x0, y0, x1, y1, blocName="<None>", group=0):
    hi_drawrectangle(x0, y0, x1, y1)
    if blocName :
        hi_addLabel(x0/2+x1/2, y0/2+y1/2, blocName, group)

def hi_addLabel(x, y, blocName="<None>", group=0):
    hi_addblocklabel(x, y)
    hi_selectlabel(x, y)
    hi_setblockprop(blocName, 1, 0, group)
    hi_clearselected()

def solve_feh_file(feh_path):
    femm.openfemm()
    femm.opendocument(feh_path)
    femm.hi_analyze()
    femm.hi_loadsolution()
    femm.closefemm()


def read_femm_problem(file_path):
    """
    Lit un fichier FEMM thermique et extrait les données structurées :
    - paramètres globaux
    - propriétés de bord (Bdry)
    - propriétés de bloc (Block)
    - conducteurs
    - points, segments, trous et labels
    """
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    data = {
        "globals": {},
        "bdrys": [],
        "blocks": [],
        "conductors": [],
        "points": [],
        "segments": [],
        "holes": [],
        "labels": []
    }

    i = 0
    while i < len(lines):
        line = lines[i]

        # --- Paramètres globaux simples [XXX] = value
        if line.startswith('[') and '=' in line and '<' not in line:
            key = re.search(r'\[(.*?)\]', line).group(1)
            value = line.split('=')[1].strip()
            data["globals"][key] = value

        # --- Bloc Bdry ---
        if '<BeginBdry>' in line:
            bdry = {}
            i += 1
            while '<EndBdry>' not in lines[i]:
                if '=' in lines[i]:
                    key, val = [x.strip() for x in lines[i].split('=')]
                    key = re.sub(r'[<>]', '', key)
                    bdry[key] = val.strip('"')
                i += 1
            data["bdrys"].append(bdry)

        # --- Bloc Block ---
        elif '<BeginBlock>' in line:
            block = {}
            i += 1
            while '<EndBlock>' not in lines[i]:
                if '=' in lines[i]:
                    key, val = [x.strip() for x in lines[i].split('=')]
                    key = re.sub(r'[<>]', '', key)
                    block[key] = val.strip('"')
                i += 1
            data["blocks"].append(block)

        # --- Bloc Conductor ---
        elif '<BeginConductor>' in line:
            cond = {}
            j = i + 1
            while j < len(lines) and '<EndConductor>' not in lines[j]:
                if '=' in lines[j]:
                    key, val = [x.strip() for x in lines[j].split('=')]
                    key = re.sub(r'[<>]', '', key)
                    cond[key] = val.strip('"')
                j += 1
            data["conductors"].append(cond)
            i = j-1  # on saute directement à la fin du bloc


        # --- Points ---
        elif line.startswith('[NumPoints]'):
            npts = int(line.split('=')[1])
            for j in range(i + 1, i + 1 + npts):
                parts = lines[j].split()
                if len(parts) >= 2:
                    data["points"].append(tuple(map(float, parts[:5])))
            i += npts

        # --- Segments ---
        elif line.startswith('[NumSegments]'):
            nseg = int(line.split('=')[1])
            for j in range(i + 1, i + 1 + nseg):
                parts = list(map(int, lines[j].split()))
                data["segments"].append(parts)
            i += nseg

        # --- Holes ---
        elif line.startswith('[NumHoles]'):
            nholes = int(line.split('=')[1])
            for j in range(i + 1, i + 1 + nholes):
                parts = list(map(float, lines[j].split()))
                data["holes"].append(parts)
            i += nholes

        # --- Block labels ---
        elif line.startswith('[NumBlockLabels]'):
            nlab = int(line.split('=')[1])
            for j in range(i + 1, i + 1 + nlab):
                parts = lines[j].split()
                data["labels"].append(tuple(map(float, parts)))
            i += nlab

        i += 1

    return data

#

def read_ans_with_mesh(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # trouver la section [Solution]
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

    # lecture des éléments
    elem_start = sol_start + 1 + npts
    n_elems = int(lines[elem_start].split()[0])
    elem_lines = lines[elem_start + 1 : elem_start + 1 + n_elems]

    elems = []
    elem_values = []
    for line in elem_lines:
        parts = line.split()
        if len(parts) >= 4:
            # indices de noeuds
            n1, n2, n3 = map(int, parts[:3])
            elems.append([n1, n2, n3])
            # lecture de la 4e valeur
            elem_values.append(float(parts[3]))
    elems = np.array(elems)
    elem_values = np.array(elem_values)

    return xs, ys, Ts, elems, elem_values

## TTS FEMM v1

import numpy as np
from pathlib import Path

def write_transient_solution(base_anh_file, all_Ts, times, save_path):
    """
    Écrit un fichier FEMM transitoire en modifiant uniquement la section [Solution] :
    - réécrit le nombre de points et les x y T0 T1 ... pour chaque point
    - conserve la section des éléments telle quelle
    - ajoute une ligne finale [n_steps] t0 t1 ...
    """
    # lecture du fichier d'origine
    with open(base_anh_file, "r") as f:
        lines = f.readlines()

    try:
        sol_start = next(i for i, line in enumerate(lines) if '[Solution]' in line) + 1
    except StopIteration:
        raise ValueError("Section [Solution] non trouvée dans le fichier FEMM.")

    xs, ys, Ts0, elems, _ = read_ans_with_mesh(base_anh_file)
    xs, ys = np.array(xs), np.array(ys)
    n_pts = len(xs)
    n_steps = len(times)
    all_Ts = np.array(all_Ts)  # shape = (n_steps, n_pts)

    if all_Ts.shape != (n_steps, n_pts):
        raise ValueError(f"Dimensions incohérentes : all_Ts={all_Ts.shape}, attendu=({n_steps}, {n_pts})")

    # trouver début et fin des éléments dans le fichier original
    n_elems = len(elems)
    elem_start = sol_start + 1 + n_pts
    elem_end = elem_start + 1 + n_elems

    # sauvegarde
    save_path = Path(save_path)
    with open(save_path, "w") as f:
        # écrire tout avant [Solution]
        for i in range(sol_start):
            f.write(lines[i])

        # réécrire le nombre de points
        f.write(f"{n_pts}\n")

        # réécrire les x y T0 T1 ... pour chaque point
        for i in range(n_pts):
            values = [xs[i], ys[i]] + [all_Ts[t, i] for t in range(n_steps)]
            line = " ".join(f"{v:.6f}" for v in values)
            f.write(line + "\n")

        # réécrire la section éléments inchangée
        for i in range(elem_start, elem_end):
            f.write(lines[i])

        # ligne finale avec les temps
        f.write(f"[{n_steps}] " + " ".join(f"{t:.6f}" for t in times) + "\n")

    print(f"✅ Fichier transitoire sauvegardé : {save_path}")


def read_transient_solution(file_path):
    """
    Lit un fichier transitoire généré par write_transient_solution :
    - section [Solution] : nombre de points, x y T0 T1 ... Tn
    - section éléments : inchangée
    - dernière ligne : [n_steps] t0 t1 ... tN

    Retour :
    xs, ys : coordonnées des points (n_pts,)
    elems : indices des éléments (n_elems, 3)
    Ts_time : température par point et par pas de temps (n_pts, n_steps)
    times : temps correspondant aux colonnes de Ts_time (n_steps,)
    """
    with open(file_path, "r") as f:
        lines = [l.strip() for l in f if l.strip()]

    # --- section [Solution] ---
    try:
        sol_start = next(i for i, line in enumerate(lines) if "[Solution]" in line) + 1
    except StopIteration:
        raise ValueError("Section [Solution] non trouvée dans le fichier.")

    # nombre de points
    npts = int(lines[sol_start])

    # lecture des points et températures
    xs, ys, Ts_list = [], [], []
    for line in lines[sol_start + 1 : sol_start + 1 + npts]:
        parts = list(map(float, line.split()))
        xs.append(parts[0])
        ys.append(parts[1])
        Ts_list.append(parts[2:])
    xs = np.array(xs)
    ys = np.array(ys)
    Ts_time = np.array(Ts_list).T  # shape = (npts, n_steps)

    # lecture des éléments
    elem_start = sol_start + 1 + npts
    n_elems = int(lines[elem_start].split()[0])
    elem_lines = lines[elem_start + 1 : elem_start + 1 + n_elems]

    elems = []
    elem_values = []
    for line in elem_lines:
        parts = line.split()
        if len(parts) >= 4:
            # indices de noeuds
            n1, n2, n3 = map(int, parts[:3])
            elems.append([n1, n2, n3])
            # lecture de la 4e valeur
            elem_values.append(float(parts[3]))
    elems = np.array(elems)
    elem_values = np.array(elem_values)

    # lecture de la ligne des temps
    times_line = lines[-1]
    parts = times_line.replace("[", "").replace("]", "").split()
    n_steps = int(parts[0])
    times = np.array(list(map(float, parts[1:])))

    if Ts_time.shape[0] != n_steps:
        print(f"⚠️ Avertissement : {Ts_time.shape[1]} colonnes trouvées, {n_steps} temps annoncés.")

    return xs, ys, Ts_time, elems, elem_values, times


## POST Traitement
def assign_elements_to_material(xs, ys, elems, elem_values, labels):
    """
    Assigne les matériaux aux éléments selon le label le plus proche
    et propage par elem_values.
    """
    # Calcul des centroïdes
    centroids = np.mean(xs[elems], axis=1), np.mean(ys[elems], axis=1)
    centroids = np.stack(centroids, axis=1)

    # Créer un dictionnaire : valeur -> indices des éléments
    value_to_elems = {}
    for idx, val in enumerate(elem_values):
        value_to_elems.setdefault(val, []).append(idx)

    # Liste finale des matériaux
    elem_materials = np.zeros(len(elems), dtype=int)

    # Pour chaque label
    for lab in labels:
        lab_coord = np.array([lab[0], lab[1]])
        mat_id = int(lab[2])

        # Cherche l'élément le plus proche du label
        dists = np.linalg.norm(centroids - lab_coord, axis=1)
        closest_idx = np.argmin(dists)

        # Récupère sa valeur
        val = elem_values[closest_idx]

        # Assigne ce matériau à tous les éléments avec cette valeur
        for e in value_to_elems[val]:
            elem_materials[e] = mat_id

    return elem_materials




## FEMM
def compute_boundary_avg_temperature(femm_data, onNone=False):
    points = np.array(femm_data["points"])
    segments = femm_data["segments"]
    bdrys = [{'BdryName': '<None>'}] + femm_data['bdrys']  # boundary 0

    # Dictionnaires pour accumuler valeurs par boundary
    dict_Ts_bdrys = {i: [] for i in range(len(bdrys))}
    dict_DTs_bdrys = {i: [] for i in range(len(bdrys))}
    dict_dP_bdrys = {i: [] for i in range(len(bdrys))}
    dict_lchems_bdrys = {i: [] for i in range(len(bdrys))}

    # Boucle sur tous les segments
    for i, segment in enumerate(segments):
        # print(f"Segment {i}")
        i1, i2, *_ , bdry = segment[:4]
        if bdry != 0 or onNone:
            pt1 = np.array(points[i1][:2])
            pt2 = np.array(points[i2][:2])
            length = np.linalg.norm(pt2 - pt1)

            # Calcul FEMM via contour
            ho_addcontour(pt1[0], pt1[1])
            ho_addcontour(pt2[0], pt2[1])
            T = ho_lineintegral(3)[0]
            DT = T - float(bdrys[bdry]['Tinf'])
            dP = DT * float(bdrys[bdry]['h'])
            ho_clearcontour()

            # Stockage des valeurs
            dict_Ts_bdrys[int(bdry)].append(T)
            dict_DTs_bdrys[int(bdry)].append(DT)
            dict_dP_bdrys[int(bdry)].append(dP)
            dict_lchems_bdrys[int(bdry)].append(length/1000)

    # Calcul de la moyenne pondérée et mise à jour des bdrys
    for i, bdry_info in enumerate(bdrys):
        lengths = np.array(dict_lchems_bdrys[i])
        if lengths.sum() > 0:
            Tavg = np.sum(np.array(dict_Ts_bdrys[i]) * lengths) / lengths.sum()
            DTavg = np.sum(np.array(dict_DTs_bdrys[i]) * lengths) / lengths.sum()
            dP_avg = np.sum(np.array(dict_dP_bdrys[i]) * lengths)
        else:
            Tavg = 0.0
            DTavg = 0.0
            dP_avg = 0.0

        # Mise à jour du dictionnaire boundary
        bdry_info['Tavg'] = Tavg
        bdry_info['DTavg'] = DTavg
        bdry_info['dP_avg'] = dP_avg
        bdry_info['length'] = lengths.sum()

    return bdrys

def compute_labels_avg_temperature(femm_data):
    block_selected = False
    for i_block, block in enumerate(femm_data['blocks']):
        for label in femm_data['labels']:
            if label[2] == i_block+1:
                x, y = label[:2]
                ho_selectblock(x, y)
                block_selected = True
        if block_selected : block['Tmean'] = ho_blockintegral(0)[0]
        ho_clearblock()
        block_selected = False

    for i_lab, label in enumerate(femm_data['labels']):
        ho_selectblock(*label[:2])
        label += (ho_blockintegral(0)[0], )
        ho_clearblock()
        femm_data['labels'][i_lab] = label

    return femm_data
## Plot functions
from matplotlib.collections import PolyCollection
import matplotlib.tri as tri
import numpy as np

def plot_labels_on_mesh(ax, xs, ys, elems, elem_values, labels, blocks, material_colors, edgecolor=None, linewidth=0.1):
    """
    Trace le maillage coloré par matériau avec couleurs fixées dans material_colors.
    """
    # Construire les coordonnées de chaque élément
    verts = np.array([[[xs[i], ys[i]] for i in elem] for elem in elems])

    # Couleurs par élément
    facecolors = [material_colors.get(m, '#CCCCCC') for m in elem_values]

    # Créer la collection et l'ajouter à l'axe
    collection = PolyCollection(verts, facecolors=facecolors, edgecolors=edgecolor, linewidths=linewidth)
    ax.add_collection(collection)

    # Affichage des labels
    for lab in labels:
        block_index = int(lab[2]) - 1
        block_name = blocks[block_index]['BlockName']
        ax.text(lab[0], lab[1], f"{block_name}", color='black', ha='center', va='center', fontsize=15)

    ax.set_aspect('equal')
    ax.set_xlim(xs.min(), xs.max())
    ax.set_ylim(ys.min(), ys.max())
    ax.set_title("Vérification de l’affectation des matériaux")



def plot_segments_by_boundary(ax, femm_data, colormap='tab20', boundary_colors=None, linewidth=2):
    """
    Trace les segments colorés selon leur boundary et ajoute une légende si bdrys est fourni.
    On peut fournir un dictionnaire boundary_colors pour fixer certaines couleurs.

    Args:
        ax: matplotlib.axes.Axes
        femm_data: dict contenant 'points', 'segments', 'bdrys'
        colormap: str, nom du colormap matplotlib
        boundary_colors: dict, optionnel, mapping boundary_index -> couleur (nom ou RGBA)
    """
    points, segments, bdrys = femm_data["points"], femm_data["segments"], femm_data["bdrys"]
    bdrys = [{'BdryName': '<None>'}] + bdrys
    points = np.array(points)
    boundaries = np.array([seg[3] for seg in segments])
    unique_boundaries = np.unique(boundaries)

    # Créer un mapping boundary -> couleur via colormap
    cmap = plt.get_cmap(colormap)
    colors = {b: cmap(i / len(unique_boundaries)) for i, b in enumerate(unique_boundaries)}
    colors[0] = 'black'
    # Remplacer par les couleurs fixées si dict fourni
    if boundary_colors:
        for b, c in boundary_colors.items():
            colors[b] = c

    # Dictionnaire pour la légende
    legend_added = {}

    # Tracer chaque segment
    for seg in segments:
        p1, p2 = points[seg[0]][:2], points[seg[1]][:2]
        boundary = seg[3]
        label = None
        if bdrys is not None and 0 <= boundary < len(bdrys):
            label_name = bdrys[boundary]['BdryName']
            if label_name not in legend_added:
                label = label_name
                legend_added[label_name] = True

        if boundary != 0 : ax.plot(
            [p1[0], p2[0]], [p1[1], p2[1]],
            color=colors[boundary],
            linewidth=linewidth,
            label=label
            )

    ax.set_aspect('equal')
    ax.set_title("Segments colorés par boundary")
    if bdrys is not None:
        ax.legend(fontsize=8, loc='best')

import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm
import matplotlib.colors as mcolors

def animer_champ_temperature(
    mesh_xs, mesh_ys, mesh_elems, all_Ts, dts,
    save_dir=None, save_name="temperature_evolution.gif",
    cmap_name="plasma", niveaux=50, interval=150
):
    """
    Anime un champ de température transitoire sur un maillage triangulaire.

    Paramètres
    ----------
    mesh_xs, mesh_ys : array-like
        Coordonnées des nœuds du maillage.
    mesh_elems : array-like
        Connectivité des éléments triangulaires (liste de triplets d’indices).
    all_Ts : array-like
        Liste (ou array 2D) contenant les champs de température à chaque instant.
    dts : array-like
        Liste des temps cumulés (en secondes) correspondant à chaque champ.
    save_dir : Path ou str, optionnel
        Dossier dans lequel sauvegarder l’animation (aucune sauvegarde si None).
    save_name : str, optionnel
        Nom du fichier de sauvegarde (l’extension définit le format : .gif, .mp4...).
    cmap_name : str, optionnel
        Nom de la colormap Matplotlib (par défaut "plasma").
    niveaux : int, optionnel
        Nombre de niveaux de contours.
    interval : int, optionnel
        Intervalle entre images (en ms) pour l’animation.
    """

    # --- Vérifications ---
    if len(all_Ts) != len(dts):
        raise ValueError("⚠️ 'all_Ts' et 'dts' doivent avoir la même longueur.")

    T_min = np.array(all_Ts).min()
    T_max = np.array(all_Ts).max()

    # --- Préparation de la figure ---
    fig, ax = plt.subplots(figsize=(8, 8))
    triang = tri.Triangulation(mesh_xs, mesh_ys, mesh_elems)

    norm = mcolors.Normalize(vmin=T_min, vmax=T_max)
    cmap = cm.get_cmap(cmap_name)

    # Premier affichage
    tcf = ax.tricontourf(triang, all_Ts[0], niveaux, cmap=cmap, vmin=T_min, vmax=T_max)
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="Température (°C)")

    ax.set_title("Champ de température - Time 0.00 h")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")

    # --- Fonction d’update ---
    def update(frame):
        nonlocal tcf

        if tcf is not None:
            tcf.remove()

        tcf = ax.tricontourf(triang, all_Ts[frame], niveaux,
                            cmap=cmap, vmin=T_min, vmax=T_max)

        time_h = dts[frame] / 3600
        ax.set_title(f"Champ de température - Time {time_h:.2f} h")

        return (tcf,)


    # --- Animation ---
    ani = FuncAnimation(fig, update, frames=len(all_Ts), blit=False, interval=interval)

    plt.show()

    # --- Sauvegarde optionnelle ---
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        filepath = save_dir / save_name
        ext = filepath.suffix.lower()

        if ext == ".gif":
            ani.save(filepath, writer="pillow")
        elif ext == ".mp4":
            ani.save(filepath, writer="ffmpeg", fps=2)
        else:
            print(f"⚠️ Format non reconnu : {ext}. Aucun fichier sauvegardé.")
            return ani

        print(f"✅ Animation sauvegardée : {filepath}")

    return ani

def plot_champ_temperature(mesh_xs, mesh_ys, mesh_elems, Ts,
                           cmap_name="plasma", niveaux=50,):
    triang = tri.Triangulation(mesh_xs, mesh_ys, mesh_elems)
    T_min, T_max = np.min(Ts), np.max(Ts)
    cmap = cm.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=T_min, vmax=T_max)

    fig, ax = plt.subplots(figsize=(8, 8))
    tcf = ax.tricontourf(triang, Ts, niveaux, cmap=cmap, vmin=T_min, vmax=T_max)
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="Température (°C)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.set_title("Champ de température")
    plt.show()


##


