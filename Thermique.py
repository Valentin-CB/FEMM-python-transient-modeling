import os, sys, inspect

path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
if not path in sys.path : sys.path.append(path)
from Classe_VCT import *

## Caracteristiques_Materiaux
class Huile(Classe_VCT):
    type = "Nytro_4000X"
    #Coefficient de dilatation de l'huile
    def B(self, T):
        if self.type == "Nytro_4000X":
            return 0.00065
        elif self.type == "Diala_S4_ZX":
            return 0.00079
        elif self.type == "Nytro_Taurus":
            return 0.00074 #Tiré de peu de données (2 points)
        elif self.type == "Ester_Midel":
            return 0.00075

    #Masse volumique de l'huile
    def ro(self, T):
        if self.type == "Nytro_4000X"  :
            return 895 * (1 + self.B(T) * (15 - T))
        elif self.type == "Diala_S4_ZX"  :
            return 819 * (1 - self.B(T) * T)
        elif self.type == "Nytro_Taurus"  :
            return 878 * (1 - self.B(T) * T) #2 points seulement
        elif self.type == "Ester_Midel"  :
            return 985 * (1 - self.B(T) * T)

    #Conduction thermique de l'huile
    def lam(self, T):
        if self.type == "Nytro_4000X"  :
            return -0.00007407 * T + 0.1548
        elif self.type == "Diala_S4_ZX"  :
            return 0.145 * (1 - 0.000552 * T)
        elif self.type == "Nytro_Taurus"  :
            return 0.1327 * (1 - 0.000512 * T)
        elif self.type == "Ester_Midel"  :
            return -0.000000720395 * T**2 - 0.0000223562 * T + 0.145098

    #Chaleur massique de l'huile
    def Cp(self, T):
        if self.type == "Nytro_4000X"  :
            return 3.467 * (T + 273.15) + 913
        elif self.type == "Diala_S4_ZX"  :
            return 2085 * (1 + 0.00224 * T)
        elif self.type == "Nytro_Taurus"  :
            return 1800 * (1 + 0.001925 * T)
        elif self.type == "Ester_Midel"  :
            return 2.17 * T + 1842

    #Viscosité dynamique de l'huile
    def u(self, T):
        if self.type == "Nytro_4000X"  :
            if T > 115  : T = 115
            return 10**(8.52817E-11 * T**4 - 0.000000775237 * T**3 + 0.000212149 * T**2 - 0.0284215 * T - 1.24845)
        elif self.type == "Diala_S4_ZX"  :
            if T > 125  : T = 125
            return 10**(0.00000000568239 * T**4 - 0.00000178785 * T**3 + 0.000244942 * T**2 - 0.0254155 * T - 1.39522)
            #Aprox Excel T€[0°C ; 100°C]
        elif self.type == "Nytro_Taurus"  :
            if T > 115  : T = 115
            return 10**(0.00000000861444 * T**4 - 0.00000240169 * T**3 + 0.000292874 * T**2 - 0.0280988 * T - 1.25758)
            #Aprox Excel T€[0°C ; 100°C]
        elif self.type == "Ester_Midel"  :
            if T > 140  : T = 140
            return 10**(0.00000000486396 * T**4 - 0.00000174316 * T**3 + 0.000276922 * T**2 - 0.0319198 * T - 0.641261)

#les fonctions suivantes sont des fonctions utilisant les lois de la physique et les formules précédantes
    #Viscosité cinématique de l'huile
    def v(self, T):
        return self.u(T) / self.ro(T)

    #diffusivitée thermique de l'huile
    def a(self, T):
        return self.lam(T) / (self.ro(T) * self.Cp(T))

    #Nombre de Prandtl de l'huile
    def Pr(self, T):
        return self.Cp(T) * self.u(T) / self.lam(T)

huile = Huile()

class Midel_MoonWatt(Classe_VCT):
    def B(self, T):
        return 0.00075

    def ro(self, T):
        return 985 * (1 - self.B(T) * T)

    def lam(self, T):
        return 0.145098

    def Cp(self, T):
        return 2.17 * T + 1842

    def v(self, T):
        v0 = 26.7
        cste = 0.091
        v1 = 226.9
        return (v0 + v1 * np.exp(-cste*T) )/ 1e6

    def u(self, T):
        return self.v(T) * self.ro(T)

    #diffusivitée thermique de l'huile
    def a(self, T):
        return self.lam(T) / (self.ro(T) * self.Cp(T))

    #Nombre de Prandtl de l'huile
    def Pr(self, T):
        return self.Cp(T) * self.u(T) / self.lam(T)

midel = Midel_MoonWatt()



class Air(Classe_VCT):
    def B(self, T):
        Tk = T + 273.15
        return 1 / Tk

    def ro(self, T):
        ro_0 = 1.293
        Tk = T + 273.15
        return ro_0 * 273.15 / Tk

    def lam(self, T):
        Tk = T + 273.15
        return 0.000000000015207 * Tk**3 - 0.00000004857 * Tk**2 + 0.00010184 * Tk - 0.00039333

    def Cp(self, T):
        Tk = T + 273.15
        return 0.00000000019327 * Tk**4 - 0.00000079999 * Tk**3 + 0.0011407 * Tk**2 - 0.4489 * Tk + 1057.5

    def u(self, T):
        Tk = T + 273.15
        return 8.8848E-15 * Tk**3 - 0.000000000032398 * Tk**2 + 0.000000062657 * Tk + 0.0000023543

    def v(self, T):
        Tk = T + 273.15
        return -1.363528E-14 * Tk**3 + 1.00881778E-10 * Tk**2 + 0.00000003452139 * Tk - 0.000003400747

    #diffusivitée thermique de l'air
    def a(self, T):
        return self.lam(T) / (self.ro(T) * self.Cp(T))

    #Nombre de Prandtl de l'huile
    def Pr(self, T):
        return self.Cp(T) * self.u(T) / self.lam(T)

air = Air()


## PCM


## Convections

def convection_libre_plaque_horizontale(fluid, Text, DT, L):
    if DT == 0:
        return 10e15
#Caracteristiques du fluid
    b = fluid.B(Text)
    ro = fluid.ro(Text)
    lam = fluid.lam(Text)
    Cp = fluid.Cp(Text)
    u = fluid.u(Text)
    v = fluid.v(Text)
    a = fluid.a(Text)
    pr = fluid.Pr(Text)
#Calcul Rayleight et Nusselt
    g = 9.81 #Pesanteur terrestre
    Ra = g * b * abs(DT) * L**3 / (v * a)
    if Ra < 0 :
        Nu = 0
    elif Ra < 10e6 :
        Nu = 0.54 * Ra**0.25
    else :
        Nu = 0.14 * Ra**0.33
    h_conv = lam * Nu / L
    return h_conv

def convection_libre_plaque_verticale(fluid, Text, DT, Hauteur):
    if DT == 0:
        return 10e15
#Caracteristiques du fluid
    b = fluid.B(Text)
    ro = fluid.ro(Text)
    lam = fluid.lam(Text)
    Cp = fluid.Cp(Text)
    u = fluid.u(Text)
    v = fluid.v(Text)
    a = fluid.a(Text)
    pr = fluid.Pr(Text)
#Calcul Rayleight et Nusselt
    g = 9.81 #Pesanteur terrestre
    Ra = g * b * abs(DT) * Hauteur**3 / (v * a)
    if Ra < 1e4 :
        Nu = 0.39 * Ra**0.25
    elif Ra < 1e9 :#
        Nu = 0.59 * Ra**0.25
    else:
        Nu = 0.12 * Ra**0.33
    h_conv = lam * Nu / Hauteur
    return h_conv#, Ra

def convection_libre_entre_plaques_verticales(fluid, Text, DT, Hauteur, epaisseur):
    if DT == 0:
        return 1e-10
#Caracteristiques du fluid
    b = fluid.B(Text)
    ro = fluid.ro(Text)
    lam = fluid.lam(Text)
    Cp = fluid.Cp(Text)
    u = fluid.u(Text)
    v = fluid.v(Text)
    a = fluid.a(Text)
    pr = fluid.Pr(Text)
#Calcul Rayleight et Nusselt
    g = 9.81 # Pesanteur terrestre
    Ra = g * b * abs(DT) * epaisseur**3 / (v * a) * (epaisseur / Hauteur)
    Nu = (576 / Ra**2 + 2.873 / Ra**0.5)**-0.5
    h_conv = lam * Nu / epaisseur
    return h_conv

def puissance_rayonnante(emissivite, T1, T2, surface):
    b = 5.67e-8
    T1K = 273.15+T1
    T2K = 273.15+T2
    return surface*emissivite*b*(T1K**4-T2K**4)

def h_ray(emissivite, T1, T2):
    """
    Calcule le flux radiatif linéarisé et retourne h_rad
    T1, T2 en °C.
    """
    sigma = 5.67e-8
    T1K = T1 + 273.15
    T2K = T2 + 273.15

    # Coefficient d’échange radiatif linéarisé
    h_rad = emissivite * sigma * (T1K**2 + T2K**2) * (T1K + T2K)
    return h_rad

import numpy as np
import matplotlib.pyplot as plt

def solar_radiation(hour, sunrise=6, sunset=18, max_radiation=1000):
    """
    Calcule le rayonnement solaire en fonction de l'heure.

    :param hour: Heure (0-24)
    :param sunrise: Heure du lever du soleil
    :param sunset: Heure du coucher du soleil
    :param max_radiation: Rayonnement solaire maximal (à midi solaire)
    :return: Rayonnement solaire à l'heure donnée
    """
    # Si l'heure est en dehors du lever ou coucher de soleil, le rayonnement est 0
    if hour < sunrise or hour > sunset:
        return 0

    # Calcul du midi solaire
    solar_noon = (sunrise + sunset) / 2

    # Détermination de la durée du jour
    day_length = sunset - sunrise

    # Modélisation par une fonction cosinus
    radiation = max_radiation * np.cos(np.pi * (hour - solar_noon) / (day_length / 2))

    # On s'assure que le rayonnement ne soit pas négatif
    return max(0, radiation)

# if __name__ == '__main__':
#     DT = 5
#     Text = 25
#     fluid = Air()
#     Hauteur = 800e-3
#     epaisseur = 15e-3
#     h_conv_ail = convection_libre_entre_plaques_verticales(
#         fluid,
#         Text,
#         DT,
#         Hauteur,
#         epaisseur
#     )
#     h_conv_vertical = convection_libre_plaque_verticale(
#         fluid,
#         Text,
#         DT,
#         300e-3
#     )[0]
#     h_conv_horizontal = convection_libre_plaque_horizontale(
#         fluid,
#         Text,
#         DT,
#         300e-3
#     )
#     surface = 1
#     h_ray = puissance_rayonnante(
#         0.8,
#         Text + DT,
#         Text,
#         surface
#     ) / surface / DT
#
#     h_vert_test = 3.66 * (DT / 10)**0.25
#     h_horiz_test = 3.41 * (DT / 10)**0.25
#
#     print(f"h_conv_ail = {h_conv_ail}")
#     print(f"h_conv_vert = {h_conv_vertical} vs {h_vert_test}")
#     print(f"h_conv_horiz = {h_conv_horizontal} vs {h_horiz_test}")
#     print(f"h_ray = {h_ray}")