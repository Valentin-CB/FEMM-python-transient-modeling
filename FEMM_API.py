import math

# ============================================================
# CORE OBJECTS
# ============================================================

class ProblemDefinition:
    def __init__(self):
        self.Format = 0
        self.Precision = 1e-8
        self.MinAngle = 30
        self.DoSmartMesh = 1
        self.Depth = 1
        self.LengthUnits = "millimeters"
        self.ProblemType = "planar" #
        self.Coordinates = "cartesian"
        self.Comment = ""

class Node:
    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)


class Segment:
    def __init__(self, n1, n2, boundary=None, group=0):
        self.n1 = n1
        self.n2 = n2
        self.boundary = boundary
        self.meshsize = -1
        self.group = group
        self.hidepros = 0
        self.group = 0
        self.conductor = 0

class ArcSegment:
    def __init__(self, n1, n2, angle, maxseg=5, boundary=None):
        self.n1 = n1
        self.n2 = n2
        self.angle = angle
        self.maxseg = maxseg
        self.boundary = boundary
        self.hidepros = 0
        self.group = 0
        self.conductor = 0

class Material:
    def __init__(self, name, kx, ky=None, kt=0, qv=0):
        self.name = name
        self.kx = kx
        self.ky = ky if ky else kx
        self.kt = kt
        self.qv = qv


class Boundary:
    def __init__(self, name, BdryType=0, Tset=0, qs=0, beta=0, h=0, Tinf=0, TinfRad=0):
        self.name = name
        self.BdryType = BdryType
        self.Tset = Tset
        self.qs = qs
        self.beta = beta
        self.h = h
        self.Tinf = Tinf
        self.TinfRad = TinfRad


class BlockLabel:
    def __init__(self, x, y, material):
        self.x = x
        self.y = y
        self.material = material
        self.meshsize = -1
        self.group = 0
        self.default = 0

# ============================================================
# MAIN PROBLEM CLASS
# ============================================================

class FemmProblem:

    current = None

    def __init__(self):
        self.nodes = []
        self.segments = []
        self.arc_segments = []
        self.materials = []
        self.boundaries = []
        self.labels = []

        self.problem = ProblemDefinition()

    # -----------------------------
    # GEOMETRY
    # -----------------------------
    def add_node(self, x, y, tol=1e-6):
        for i, n in enumerate(self.nodes):
            if abs(n.x - x) < tol and abs(n.y - y) < tol:
                return i
        self.nodes.append(Node(x, y))
        return len(self.nodes) - 1

    def add_segment(self, x1, y1, x2, y2, boundary=None):
        n1 = self.add_node(x1, y1)
        n2 = self.add_node(x2, y2)

        # Vérification existence (dans les deux sens)
        for seg in self.segments:
            if (seg.n1 == n1 and seg.n2 == n2) or (seg.n1 == n2 and seg.n2 == n1):
                print(f"WARNING: segment déjà existant entre nodes {n1} et {n2}")
                return None
        self.segments.append(Segment(n1, n2, boundary))

    def add_arc(self, x1, y1, x2, y2, angle, maxseg=5, boundary=None):
        n1 = self.add_node(x1, y1)
        n2 = self.add_node(x2, y2)
        self.arc_segments.append(ArcSegment(n1, n2, angle, maxseg, boundary))

    # -----------------------------
    # PHYSICS
    # -----------------------------
    def add_material(self, name, kx, ky=None, kt=0, qv=0):
        idx = len(self.materials)
        self.materials.append(Material(name, kx, ky, kt, qv))

    def add_boundary(self, name, BdryType=0, Tset=0, qs=0, beta=0, h=0, Tinf=0, TinfRad=0):
        self.boundaries.append(Boundary(name, BdryType, Tset, qs, beta, h, Tinf, TinfRad))

    def add_blocklabel(self, x, y, material):
        self.labels.append(BlockLabel(x, y, material))

    # -----------------------------
    # EXPORT
    # -----------------------------
    def write(self, filename):

        with open(filename, "w") as f:

            # --- globals
            f.write(f"[Format] = {self.problem.Format}\n")
            f.write(f"[Precision] = {self.problem.Precision}\n")

            # --- materials
            f.write(f"[BlockProps] = {len(self.materials)}\n")
            for m in self.materials:
                f.write("<BeginBlock>\n")
                f.write(f'  <BlockName> = "{m.name}"\n')
                f.write(f"  <Kx> = {m.kx}\n")
                f.write(f"  <Ky> = {m.ky}\n")
                f.write(f"  <Kt> = {m.kt}\n")
                f.write(f"  <qv> = {m.qv}\n")
                f.write("<EndBlock>\n")

            # --- boundaries
            f.write(f"[BdryProps] = {len(self.boundaries)}\n")
            for b in self.boundaries:
                f.write("<BeginBdry>\n")
                f.write(f'<BdryName> = "{b.name}"\n')
                f.write(f"<BdryType> = {b.BdryType}\n")
                f.write(f"<Tset> = {b.Tset}\n")
                f.write(f"<qs> = {b.qs}\n")
                f.write(f"<beta> = {b.beta}\n")
                f.write(f"<h> = {b.h}\n")
                f.write(f"<Tinf> = {b.Tinf}\n")
                f.write(f"<TinfRad> = {b.TinfRad}\n")
                f.write("<EndBdry>\n")

            # --- nodes
            f.write(f"[NumPoints] = {len(self.nodes)}\n")
            for n in self.nodes:
                f.write(f"{n.x} {n.y} 0 0 0\n")

            # --- segments
            f.write(f"[NumSegments] = {len(self.segments)}\n")
            for s in self.segments:

                bound_id = 0
                if s.boundary:
                    for i, b in enumerate(self.boundaries):
                        if b.name == s.boundary:
                            bound_id = i+1
                            break

                f.write(f"{s.n1} {s.n2} {s.meshsize} {bound_id} {s.hidepros} {s.group} {s.conductor}\n")

            # --- arcs
            f.write(f"[NumArcSegments] = {len(self.arc_segments)}\n")
            for a in self.arc_segments:
                bound_id = 0
                if a.boundary:
                    for i, b in enumerate(self.boundaries):
                        if b.name == a.boundary:
                            bound_id = i+1
                            break

                f.write(f"{a.n1} {a.n2} {a.angle} {a.maxseg} {bound_id} {a.hidepros} {a.group} {a.conductor}\n")

            # --- labels
            f.write(f"[NumBlockLabels] = {len(self.labels)}\n")
            for l in self.labels:

                mat_id = 0
                if l.material:
                    for i, m in enumerate(self.materials):
                        if m.name == l.material:
                            mat_id = i+1
                            break

                f.write(f"{l.x} {l.y} {mat_id} {l.meshsize} {l.group} {l.default}\n")


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":
    import femm
    prob = FemmProblem()

    # --- materials
    prob.add_material("Core", 10)
    prob.add_material("Coating", 1)

    # --- boundary convection
    prob.add_boundary("conv", BdryType=3, h=50, Tinf=293)

    w, h, r = 10, 6, 1

    # rectangle arrondi
    prob.add_segment(r,0, w-r,0)
    prob.add_arc(w-r,0, w,r,90)
    prob.add_segment(w,r, w,h-r)
    prob.add_arc(w,h-r, w-r,h,90)
    prob.add_segment(w-r,h, r,h)
    prob.add_arc(r,h, 0,h-r,90)
    prob.add_segment(0,h-r, 0,r)
    prob.add_arc(0,r, r,0,90)

    prob.add_blocklabel(w/2, h/2, "Core")

    # enrobage
    margin = 2

    prob.add_segment(-margin,-margin, w+margin,-margin)
    prob.add_segment(w+margin,-margin, w+margin,h+margin, "conv")
    prob.add_segment(w+margin,h+margin, -margin,h+margin, "conv")
    prob.add_segment(-margin,h+margin, -margin,-margin, "conv")

    prob.add_blocklabel(w/2, h+1, "Coating")

    prob.write("test_full.feh")
    femm.openfemm()
    femm.opendocument("test_full.feh")