import numpy as np

particle_names = {
    13: "µ-",
    -13: "µ+",
    11: "e-",
    -11: "e+",
    22: "gamma",
    2112: "n",
    2212: "p",
    211: "π+",
    -211: "π-",
}


class PHSPFile:
    "Represent a TOPAS phase-space *input* file for checking its entries."
    # Copied from check_phsp_file.py
    def __init__(self, filename, hasAux=False):
        self.filename = filename
        if filename.endswith(".phsp"):
            filename = filename[:-5]

        dtype = []
        with open(filename+".header", "r") as h:
            # N = nbytes = 0
            for line in h.readlines():
                # if line.startswith("Number of Scored Particles: "):
                #     N = int(line.split(" ")[-1])
                # elif line.startswith("Number of Bytes per Particle: "):
                #     nbytes = int(line.split(" ")[-1])
                if line.startswith("f4: "):
                    dtype.append((line[4:-1], np.float32))
                elif line.startswith("i4: "):
                    dtype.append((line[4:-1], np.int32))
                elif line.startswith("b1: "):
                    dtype.append((line[4:-1], np.byte))
                elif line.startswith("Exposure Time: "):
                    self.time = float(line.split(" ")[2])
        self.dtype = dtype
        self.data = np.memmap(filename+".phsp", dtype=dtype, mode="r")
        self.x = self.data["Position X [cm]"]
        self.y = self.data["Position Y [cm]"]
        self.z = self.data["Position Z [cm]"]
        self.E = self.data["Energy [MeV]"]
        self.nx = self.data["Direction Cosine X"]
        self.ny = self.data["Direction Cosine Y"]
        nz = 1 - self.nx**2 - self.ny**2
        nz[nz < 0] = 0
        self.nz = -np.sqrt(nz)
        down = self.data["Flag to tell if Third Direction Cosine is Negative (1 means true)"]
        self.nz[down == 0] *= -1
        self.weight = self.data["Weight"]
        self.id = self.data["Particle Type (in PDG Format)"]
        with open(filename+".header", "r") as f:
            for line in f.readlines():
                if line.startswith("Exposure Time:"):
                    self.time = float(line.split(" ")[2])
        try:
            self.event = self.data["Event ID"]
        except ValueError:
            pass

        if hasAux:
            auxtype = [
                ("E", np.float32),
                ("costheta", np.float32),
                ("path15", np.float32),
                ("path05", np.float32),
            ]
            self.aux = np.memmap(filename+".aux", dtype=auxtype)

    @property
    def N(self):
        return len(self.E)

    @property
    def N_effective(self):
        return self.weight.sum()

    # def copy_weights(self, other):
    #     assert self is not other
    #     self.weight[:] = 1.0
    #     self.weight[:] = other.weight[self.data["Event ID"]]

    def summarize(self):
        Etotal = 0.0
        Ntotal = 0.0
        for id, name in particle_names.items():
            E = self.E[self.id == id]
            n = len(E)
            wt = self.weight[self.id == id]
            n_effective = wt.sum()
            if n <= 0:
                continue

            T = 1e6
            Ntotal += n_effective
            Emean = (E*wt).sum()/n_effective
            print(f"{name:5.5s}: med(E) {np.median(E):7.2f}  <E> {Emean:7.2f} MeV  total {Emean*n_effective/T:7.3f} "
                  f"MeV/CRprimary   {n_effective:9.2f}")
            Etotal += Emean*n_effective/T
        E = self.E
        name = "All"
        print(f"{name:5.5s}: med(E) {np.median(E):7.2f}  <E> {E.mean():7.2f} MeV  total {Etotal:7.3f} MeV/CRprimary   {Ntotal:9.2f}")

    def count_sources(self, before):
        for idb, nameb in particle_names.items():
            for id, name in particle_names.items():
                n = self.weight[(self.id == id) & (before.id[self.data["Event ID"]] == idb)].sum()
                if n > 0:
                    print(f"{nameb:5s} => {name:5s}  {n:8.1f}")

    def approximate_distributions(self):
        self.spectra = {}
        self.angles = {}
        cutoffs = {
            13: (-3, 2, 3, 4, 6),
            11: (-3, 0, 1, 2, 3, 6),
            22: (-3, -1, 0, 1, 2, 3, 6),
            211: (-3, 6),
            2112: (-3, -1, 0, 1, 2, 3, 6),
            2212: (-3, 2, 3, 6),
        }
        for id in np.unique(self.id):
            use = (self.id == id)
            logE = np.log10(self.E[use])
            weight = self.weight[use]
            pdf, b = np.histogram(logE, 180, [-3, 6], weights=weight)
            cdf = pdf.cumsum()
            self.spectra[id] = (b[1:], cdf)

        for id in np.unique(self.id):
            self.angles[id] = {}
            cutoff = cutoffs[abs(id)]
            for i in range(len(cutoff)-1):
                emin = 10**cutoff[i]
                emax = 10**cutoff[i+1]
                use = (self.id == id) & (self.E >= emin) & (self.E < emax) & (self.E > 0)
                weight = self.weight[use]
                pdf, b = np.histogram(-self.nz[use], 100, [0, 1], weights=weight)
                print(id, i, (-self.nz[use]*weight).sum()/weight.sum(), weight.sum(), pdf.sum())
                # cdf = pdf.cumsum()/pdf.sum()
                self.angles[id][i] = (b[1:], pdf)

    def restart_tracks(self, R, outputfile):
        resample_factors = {
            13: 3,
            -13: 3,
            11: 2,
            -11: 6,
            22: 1,
            2112: 1,
            2212: 1,
            211: 5,
            -211: 10,
        }
        for pid, resample in resample_factors.items():
            pname = particle_names[pid]
            print(f"Restarting tracks for {pname}")
            self.restart_tracks_onetype(pid, resample, R, f"{outputfile}_{pname}")

    def restart_tracks_onetype(self, particle_id, resample, R, outputfile, ignoresign=False, alltypes=False):
        """Create a new phase-space file where the tracks' (x,y,z) positions have been adjusted to aim
        at a sphere of radius R (centered at the origin). Store in `outputfile`"""
        assert outputfile != self.filename

        # Handle weighting:
        # Particles with higher than their natural weight get reused; lower get trimmed.
        default_weight = {2212: 0.01, 2112: 0.1, 211: 0.01, -211: 0.01}.get(particle_id, 1.0)
        correct_id = (self.id == particle_id)  # &(self.weight == 1.0)
        if ignoresign:
            correct_id |= (self.id == -particle_id)
        if alltypes:
            correct_id |= True

        use_idx = []
        correct_id_idx = np.nonzero(correct_id)[0]

        weight_ratio = self.weight[correct_id] / default_weight
        for idx, wr in zip(correct_id_idx, weight_ratio):
            if wr > 1.001:
                use_idx.extend([idx]*round(wr))
            elif wr > 0.999:
                use_idx.append(idx)
            elif rng.uniform() < wr:
                use_idx.append(idx)

        # 1. Aim at point (0,0,0)
        Nraw = len(use_idx)
        N = Nraw * resample
        if Nraw == 0:
            print("   no such particle. Skipping...")
            return
        print(f"Generating tracks for {N} ({Nraw}*{resample}) {particle_names[particle_id]}" +
              f" particles from {correct_id.sum()} weighted")
        x = np.zeros(N, dtype=np.float32)
        y = np.zeros(N, dtype=np.float32)
        z = np.zeros(N, dtype=np.float32)

        # 2. Back up from here until z = +R: pretend cryostat shell is that far away (for particle decays)
        nz = self.nz[use_idx]
        d = -R/nz
        d[nz == 0] = -R
        for i in range(resample):
            rs_range = range(i*Nraw, i*Nraw+Nraw)
            nx = self.nx[use_idx]
            ny = self.ny[use_idx]
            nz = self.nz[use_idx]
            x[rs_range] -= d*nx
            y[rs_range] -= d*ny
            z[rs_range] -= d*nz

            # 3. Choose a shift in the circle of radius R
            r = np.sqrt(rng.uniform(0, R**2, size=Nraw))
            phi = rng.uniform(-np.pi, np.pi, size=Nraw)
            ushift = r*np.cos(phi)
            vshift = r*np.sin(phi)
            nshift = -np.sqrt(R**2-r**2)

            # 4. Find unit vectors u,v mutually orthogonal to each other and self.n.
            u = np.zeros((Nraw, 3), dtype=float)
            bigx = np.abs(nx) > 0.7
            u[bigx, 0] = -nz[bigx]
            u[bigx, 2] = nx[bigx]
            smallx = ~bigx
            u[smallx, 1] = nz[smallx]
            u[smallx, 2] = -ny[smallx]
            u = (u.T / np.sqrt((u**2).sum(axis=1))).T
            n = np.vstack([nx, ny, nz]).T
            v = np.cross(n, u)
            shift = u.T*ushift + v.T*vshift + n.T*nshift
            x[rs_range] += shift[0, :]
            y[rs_range] += shift[1, :]
            z[rs_range] += shift[2, :]

        # Store...?
        with open(f"{self.filename}.header", "r") as fin:
            with open(f"{outputfile}.header", "w") as fout:
                for line in fin.readlines():
                    if line.startswith("Number of Scored Particles:"):
                        line = f"Number of Scored Particles: {N}\n"
                    if line.startswith("Number of Original Histories that Reached Phase Space:"):
                        line = f"Number of Original Histories that Reached Phase Space: {N}\n"
                    if line.startswith("Number of Original Histories:"):
                        line = f"Number of Original Histories: {N}\n"
                    if "Kinetic Energy of" in line:
                        continue
                    fout.write(line)

        fout = np.memmap(f"{outputfile}.phsp", dtype=self.dtype[:], shape=(N,), mode="w+")
        fout["Position X [cm]"][:] = x
        fout["Position Y [cm]"][:] = y
        fout["Position Z [cm]"][:] = z
        copyfields = (
            "Direction Cosine X",
            "Direction Cosine Y",
            "Energy [MeV]",
            "Weight",
            "Particle Type (in PDG Format)",
            "Flag to tell if Third Direction Cosine is Negative (1 means true)",
            "Flag to tell if this is the First Scored Particle from this History (1 means true)",
            "Event ID",
            "Parent ID",
        )
        for fname in copyfields:
            if fname not in self.data.dtype.fields:
                continue
            for i in range(resample):
                rs_range = range(i*Nraw, i*Nraw+Nraw)
                fout[fname][rs_range] = self.data[fname][use_idx]
        fout["Flag to tell if this is the First Scored Particle from this History (1 means true)"][:] = 1
        fout["Weight"][:] = 1.0

    def spectrum_by_weight(self, id=11):
        use = (self.E > 0) & (self.id == id)
        plt.clf()
        plt.hist(np.log10(self.E[use]), 180, [-3, 6], weights=self.weight[use], histtype="step", color="k")
        use2 = use & (self.weight == 1)
        print(self.weight[use2].sum())
        plt.hist(np.log10(self.E[use2]), 180, [-3, 6], weights=self.weight[use2], histtype="step", label="e±, µ±, gamma")
        use2 = use & (self.weight == .1)
        print(self.weight[use2].sum())
        plt.hist(np.log10(self.E[use2]), 180, [-3, 6], weights=self.weight[use2], histtype="step", label="neutron")
        use2 = use & (self.weight == .01)
        print(self.weight[use2].sum())
        plt.hist(np.log10(self.E[use2]), 180, [-3, 6], weights=self.weight[use2], histtype="step", label="proton")
        plt.legend()
        plt.xlabel("log10(E loss / 1 MeV)")

    def effective_area(self, lx, ly, lz):
        top = lx*ly
        sides = 2*(lx+ly)*lz
        s2 = self.nx**2 + self.ny**2
        s2[s2 > 1] = 1
        s = np.sqrt(s2)
        c = np.sqrt(1-s2)

        areas = top*c + sides*s/np.pi
        # plt.clf()
        # plt.hist(areas, 100, [0, lx*ly*1.4], histtype="step")
        return np.mean(areas)

    def effective_area_length(self, lx, ly, lz):
        top = lx*ly
        sides = ly*lz, lx*lz
        # s2 = self.nx**2 + self.ny**2
        # s2[s2 > 1] = 1
        # nz = np.sqrt(1-s2)

        areas = top*np.abs(self.nz) + sides[0]*np.abs(self.nx) + sides[1]*np.abs(self.ny)
        lengths = lx*ly*lz/areas
        plt.clf()
        plt.subplot(211)
        plt.hist(areas, 100, histtype="step")
        plt.subplot(212)
        plt.hist(lengths, 100, histtype="step")
        A, L = np.mean(areas), np.mean(lengths)
        print(f"{A:.4f}  {L:.4f}  {A*L:.3f} {np.abs(self.nz).mean():.4f}")
        return A, L

    def effective_area_length_cylinder(self, r, h, vertical=False):
        caps = np.pi*r*r
        lateral = 2*r*h
        volume = caps*h
        if vertical:
            side = np.sqrt(self.nx**2 + self.ny**2)
            side[side > 1] = 1
            nz = np.abs(self.nz)
        else:
            side = np.sqrt(self.nz**2 + self.ny**2)
            side[side > 1] = 1
            nz = np.abs(self.nx)

        areas = caps*nz + lateral*side
        lengths = volume/areas
        plt.clf()
        plt.subplot(211)
        plt.title("Effective area")
        plt.hist(areas, 100, histtype="step")
        plt.grid()
        plt.subplot(212)
        plt.title("Path length")
        plt.hist(lengths, 100, histtype="step")
        plt.grid()
        A, L = np.mean(areas), np.mean(lengths)
        print(f"{A:.4f}  {L:.4f}  {A*L:.3f} {np.abs(nz).mean():.4f}")
        return A, L
