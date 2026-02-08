from FEMM_Mesh_and_plot import *
c_sb = 5.67e-8

from femm import *

fileName = "plaque_acier_alu"

openfemm()
newdocument(2)

# -----------------------------
# Définition du problème FEMM
# -----------------------------
hi_probdef("millimeters", "planar", 1e-8, 1, 30)

# -----------------------------
# Matériaux
# -----------------------------
hi_addmaterial("Acier", 45, 45, 0, 460*7800/1e6)       # Cp*rho en MJ/m3/K
hi_addmaterial("Aluminium", 205, 205, 0, 900*2700/1e6)

# -----------------------------
# Conditions limites
# -----------------------------

# Plaque exposée à flux solaire + convection + rayonnement
# Note : FEMM attend T en °C, mais on peut injecter en K si on veut
hi_addboundprop("air_ext", 3, 0, -900, 40+273, 8, 0.8)
hi_addboundprop("sol_40C", 0, 40+273, 0, 0, 0, 0)  # Dirichlet : température imposée

# -----------------------------
# Géométrie
# -----------------------------
# Plaque acier :
hi_addRectangle(0, 0, 100, 5, "Aluminium")
# Tige aluminium
hi_addRectangle(45, 0, 50, -50, "Acier")

# -----------------------------
# Application des conditions limites
# -----------------------------
# Face supérieure de la plaque → flux solaire + convection + rayonnement
hi_setbound(50, 5, "air_ext")
# Face inférieure de la tige aluminium → contact sol T=40°C
hi_setbound(50, -50, "sol_40C")

# -----------------------------
# Zoom et sauvegarde
# -----------------------------
hi_zoomnatural()

hi_saveas(f"{fileName}.FEH")
hi_analyze()
hi_loadsolution()

xs, ys, Ts, elems, elem_values = read_ans_with_mesh(f'{fileName}.anh')
femm_data = read_femm_problem(f'{fileName}.anh')
elem_materials = assign_elements_to_material(
    xs, ys, elems, elem_values,
    femm_data["labels"]
)
plot_champ_temperature(xs, ys, elems, Ts-273)

fig, ax = plt.subplots(figsize=(8, 8))
plot_labels_on_mesh(
    ax, xs, ys, elems, elem_materials,
    femm_data["labels"], femm_data["blocks"],
    material_colors={
        1:'#777777', 2:'#6AF261', 4:'#3E55EF',
        5:'#DE9709', 6:'#EEEEEE'
    },
    edgecolor=None,
    linewidth=0.01
)
plt.show()