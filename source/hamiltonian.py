""" This file contains the Hamiltonian and Ergodicity classes. """

from .base import * # imports BaseSeedClass, numpy, matplotlib, joblib (delayed and Parallel)
from qutip import *
from quspin.operators import hamiltonian
from quspin.basis import spin_basis_1d

import itertools

class Hamiltonian(BaseSeededClass):

    """ Reservoir of spins described by an Ising hamiltonian with transverse magnetic field and local disorder:

    \mathcal{H} = \sum_{i>j=1}^N J_{ij}\sigma^x_i\sigma^x_j + \frac{1}{2}\sum_{i=1}^N(h+D_i)\sigma_i^z\,.

    where \sigma_i^x, \sigma_i^z are the Pauli matrices acting on the i-th spin, J_{ij} are the coupling constants between the spins, h is the transverse magnetic field, and D_i are the local disorder terms.
    """

    def __init__(self, L, J, h, W, check_symm=False, check_pcon=False, check_herm=False, zblock=None, seed=None, L_max=None, **kwargs):

        """ Initialize the Hamiltonian class.
        Parameters:
        L (int): number of qubits
        J (float): coupling constant
        h (float): transverse field
        W (float): local disorder strength
        check_symm (bool): check for symmetry
        check_pcon (bool): check for parity conservation (do not check if zblock is not None)
        check_herm (bool): check for hermiticity
        zblock (int): parity block (1 for even, -1 for odd, None for no parity block)
        seed (int): random seed for reproducibility
        L_max (int): Maximum number of qubits for the observables
        """
        
        
        if not isinstance(L, int) or L <= 1:
            raise ValueError("L must be an integer greater than 1.")
        if not isinstance(J, (int, float)) or J < 0:
            raise ValueError("J must be a non-negative number.")
        if not isinstance(h, (int, float)):
            raise ValueError("h must be a number.")
        if not isinstance(W, (int, float)) or W < 0:
            raise ValueError("W must be a non-negative number.")
        if zblock not in [1, -1, None]:
            raise ValueError("zblock must be 1, -1, or None.")
        
        super().__init__(seed=seed, **kwargs) # initialize the parent class

        self.L = L # number of qubits
        self.Lc = np.sum(range(L)) # number of pair correlations
        self.Lcc = 2*self.Lc
        self.Js = J # coupling constant
        self.h = h # transverse field
        self.W = W # local disorder strength

        self.check_symm = check_symm # check for symmetry
        self.check_pcon = check_pcon # check for parity conservation (do not check if zblock is not None)
        self.check_herm = check_herm # check for hermiticity 
        self.zblock = zblock # parity block (1 for even, -1 for odd, None for no parity block)

        # print(f"[Hamiltonian] Random seed: {self.seed}")

        self.L_max = L if L_max is None else L_max # maximum number of qubits for the observables
        if self.L_max > L:
            raise ValueError("L_max must be less than or equal to L.")
        
    def __repr__(self):

        """ Return a representation of the Hamiltonian class. """

        return f"Hamiltonian(L={self.L}, J={self.Js}, h={self.h}, W={self.W}, zblock={self.zblock})"

    def __str__(self):

        """ Return a string description of the Hamiltonian class. """

        if self.zblock == 1 or self.zblock == -1:
            return f"Ising Hamiltonian with {self.L} qubits, coupling constant J={self.Js}, transverse field h={self.h}, and local disorder strength W={self.W}, with parity block {self.zblock}."
        else:
            return f"Ising Hamiltonian with {self.L} qubits, coupling constant J={self.Js}, transverse field h={self.h}, and local disorder strength W={self.W}."

    def generate_hamiltonian(self):

        """
        Generates the parameters of the Hamiltonian: the coupling constants and the local transverse field.
        The coupling constants are uniformly distributed in the range [-Js/2, Js/2], and the local transverse field is uniformly distributed in the range [-W, W].
        """

        Js2 = self.Js / 2 # absolute value of the spin-spin interaction boundary values
        N = np.sum(range(1, self.L)) # number of interactions
        J_val = self.rng.uniform(-Js2, Js2, N) # random coupling constants
        D = self.rng.uniform(-self.W, self.W, self.L) # random disorder values
        D = 0.5*(D + self.h) # disorder values are shifted by the transverse field

        self.J_xx = [[J_val[k], i, j] for k, (i,j) in enumerate(itertools.combinations(range(self.L), 2))] # spin-spin interaction terms
        self.h_z = [[D[i], i] for i in range(self.L)] # local transverse field terms

        return self.J_xx, self.h_z

    def construct_hamiltonian(self):

        """
        Constructs the Hamiltonian from the list of operators.
        """

        if self.zblock == 1 or self.zblock == -1:
            self.basis = spin_basis_1d(self.L, pauli=1, zblock=self.zblock) # spin basis, only consider positive parity sector (model possesses a Z_2 symmetry)
            static = [["zz", self.J_xx], ["x", self.h_z]] # static Hamiltonian terms (by applying the Hadamard transformation to the Hamiltonian)
        elif self.zblock == None:
            self.basis = spin_basis_1d(self.L, pauli=1)
            static = [["xx", self.J_xx], ["z", self.h_z]] # static Hamiltonian terms

        # Unpack parameters
        dynamic = [] # no time-dependence
        return hamiltonian(static, dynamic, basis=self.basis, dtype=np.float64, check_symm=self.check_symm, check_pcon=self.check_pcon, check_herm=self.check_herm)

    def get_hamiltonian(self):

        """
        Returns the Hamiltonian object.
        """
        self.generate_hamiltonian()
        return self.construct_hamiltonian()

    def get_E_V(self):

        """
        Returns the eigenvalues and eigenstates of the Hamiltonian.
        """
        H = self.get_hamiltonian() # get Hamiltonian
        self.E, self.V = H.eigh() # eigenvalues and eigenstates of the Hamiltonian
        return self.E, self.V
    
    #---------------------------------------------------------------------

    def local_spin_operators(self, axis='z'):
        
        """ This function defines the local spin operators for each spin up to L_max
        
        Parameters:
        axis (str): axis of the spin operator ('x', 'y', 'z')

        returns:
        sigmax, sigmay, sigmaz: list of local spin operators for each spin
            local spin operators for the x, y, and z axes respectively
        """

        if axis == 'x':
            self.sigmax = [tensor([qeye(2) if i != j else sigmax() for j in range(self.L)]) for i in range(self.L_max)] # sigma_x operator
            return self.sigmax
        elif axis == 'y':
            self.sigmay = [tensor([qeye(2) if i != j else sigmay() for j in range(self.L)]) for i in range(self.L_max)] # sigma_y operator
            return self.sigmay
        elif axis == 'z':
            self.sigmaz = [tensor([qeye(2) if i != j else sigmaz() for j in range(self.L)]) for i in range(self.L_max)] # sigma_z operator
            return self.sigmaz
        else:
            raise ValueError("The axis should be 'x', 'y', 'z'")
        
    def two_spins_correlations(self, axis='z'):

        """
        Calculate two-spin correlation operators for a chosen axis or axis-pair.

        Parameters
        axis : str
            'x', 'y', 'z' for same-axis correlations,
            or two-letter strings like 'xy', 'yz', 'yx', etc.
            - 'xy' means sigma_x on site i and sigma_y on site j.
            - 'yx' means sigma_y on site i and sigma_x on site j.

        Returns
        correlations : list of Qobj
            List of two-spin correlation operators for the chosen axis/axes.
        """

        if self.L_max < 2:
            raise ValueError("L_max must be greater than or equal to 2 for two spins correlations.")

        if axis == 'x':
            # Ensures that sigmax is initialized
            if not hasattr(self, 'sigmax') or self.sigmax is None:
                self.local_spin_operators(axis='x')
            sigma = self.sigmax
        if axis == 'y':
            # Ensures that sigmay is initialized
            if not hasattr(self, 'sigmay') or self.sigmay is None:
                self.local_spin_operators(axis='y')
            sigma = self.sigmay
        if axis == 'z':
            # Ensures that sigmaz is initialized
            if not hasattr(self, 'sigmaz') or self.sigmaz is None:
                self.local_spin_operators(axis='z')
            sigma = self.sigmaz

        sigma_dict = {
            'x': getattr(self, 'sigmax', None),
            'y': getattr(self, 'sigmay', None),
            'z': getattr(self, 'sigmaz', None)
        }

        correlations = []

        if len(axis) == 1:
            sigma = sigma_dict[axis]
            for i in range(self.L_max):
                for j in range(self.L_max):
                    if i < j:
                        correlations.append(sigma[i] @ sigma[j])

        elif len(axis) == 2:
            a1, a2 = axis[0], axis[1]
            sigma1, sigma2 = sigma_dict[a1], sigma_dict[a2]
            sigma1_alt, sigma2_alt = sigma_dict[a2], sigma_dict[a1]  # swapped order

            for i in range(self.L_max):
                for j in range(self.L_max):
                    if i < j:
                        correlations.append(sigma1[i] @ sigma2[j])
                        correlations.append(sigma1_alt[i] @ sigma2_alt[j])

        if axis == 'x':
            self.correlations_x = correlations
        if axis == 'y':
            self.correlations_y = correlations
        if axis == 'z':
            self.correlations_z = correlations
        if axis == 'xy' or axis == 'yx':
            self.correlations_xy = correlations
        if axis == 'xz' or axis == 'zx':
            self.correlations_zx = correlations
        if axis == 'zy' or axis == 'yz':
            self.correlations_zy = correlations

        return correlations
    
    @staticmethod
    def get_all_observables(L,L_max=None):

        """
        Returns all the observables of the system:
        local spin operators and two spins correlations for the z-axis (only the upper triangular part of the matrix).
        """

        h = Hamiltonian(L, 0, 0, 0, L_max=L_max) # dummy Hamiltonian object to call the methods

        sx = h.local_spin_operators(axis='x')
        sy = h.local_spin_operators(axis='y')
        sz = h.local_spin_operators(axis='z')
        correlations_x = h.two_spins_correlations(axis='x')
        correlations_y = h.two_spins_correlations(axis='y')
        correlations_z = h.two_spins_correlations(axis='z')
        correlations_xy = h.two_spins_correlations(axis='xy')
        correlations_zx = h.two_spins_correlations(axis='zx')
        correlations_zy = h.two_spins_correlations(axis='zy')


        return sx, sy, sz, correlations_x, correlations_y, correlations_z, correlations_xy, correlations_zx, correlations_zy
    
    @staticmethod
    def average_values(rho, observables):

        """
        Computes the expectation values of given observables.

        Parameters:
        rho: density matrix (Qobj)
        observables: list of observables (Qobj)
        """

        if not all(isinstance(obs, Qobj) for obs in observables):
            raise ValueError("All observables must be instances of Qobj.")
        
        average_values = qutip.expect(observables, rho) # expectation values of the observables

        return average_values
    

class Ergodicity(Hamiltonian):

    """
    Class to check the dynamical phase of the system by computing the gap ratio.
    The gap ratio is defined as the ratio of the minimum and maximum gaps between consecutive energy levels in the spectrum.
    The gap ratio is a measure of the level spacing distribution and can be used to determine whether the system is ergodic or not.
    If the level spacing is approximately Poissonian with <r_n> = 0.386, the system is localized/non-ergodic.
    If the level spacing is approximately Wigner-Dyson with <r_n> = 0.53, the system is chaotic/ergodic.
    """

    def __init__(self, L, J, h, W, zblock=1, n_iter=1, alpha=0, seed=None, **kwargs):

        """
        Initialize the Ergodicity class.
        Parameters:
        L (int): number of qubits
        J (float): coupling constant
        h (float): transverse field
        W (float): local disorder strength
        zblock (int): parity block (1 for even, -1 for odd, None for no parity block)
        n_iter (int): number of iterations for the random Hamiltonian generation
        alpha (float): boundary parameter modifier (0 < alpha < 0.5)
        seed (int): random seed for reproducibility
        """


        if not isinstance(n_iter, int) or n_iter <= 0:
            raise ValueError("n_iter must be a positive integer.")
        if not isinstance(alpha, (int, float)) or alpha < 0 or alpha >= 0.5:
            raise ValueError("alpha must be a number between 0 and 0.5.")

        super().__init__(L, J, h, W, zblock=zblock, seed=seed, **kwargs) # initialize the Hamiltonian class
        self.n_iter = n_iter # number of iterations for the random Hamiltonian generation
        self.alpha = alpha # Bpundary parameter modifier

    def __repr__(self):

        """ Return a representation of the Ergodicity class. """

        return f"Ergodicity(L={self.L}, J={self.Js}, h={self.h}, W={self.W}, zblock={self.zblock}, n_iter={self.n_iter}, alpha={self.alpha})"
    
    def __str__(self):

        """ Return a string description of the Ergodicity class. """

        return f"Ergodicity class with {self.L} qubits, coupling constant J={self.Js}, transverse field h={self.h}, disorder strength W={self.W}, parity block {self.zblock}, number of iterations {self.n_iter}, and alpha {self.alpha}."

    def center_ernergy(self):

        """
        Returns the center of the energy spectrum.
        Since the systme is finite, to observe thermodynamic limit behaviour we need to consider the center of the energy spectrum to avoid finite size effects.
        """
        
        E, _ = self.get_E_V()
        E = np.sort(E)

        max_index = int( (len(E) - 1) * (1 - self.alpha) ) # index of the maximum energy in the spectrum
        min_index = int( (len(E) - 1) * self.alpha )

        E_center = E[min_index:max_index] # energy values in the range (E_min, E_max)

        return E_center
    
    def check_ergodicity(self):

        """
        Function to check the ergodicity of the system by computing the gap ratio.
        """

        E = self.center_ernergy()
        E_gaps = np.diff(E)

        if len(E_gaps) < 2:
            raise ValueError("Not enough energy gaps to compute the gap ratio. Decrease alpha.")

        self.r_n_values = np.minimum(E_gaps[:-1], E_gaps[1:]) / np.maximum(E_gaps[:-1], E_gaps[1:]) # gap ratio
        self.r_n = np.mean(self.r_n_values) # mean gap ratio

        return self.r_n

    def mean_ergodicity(self):
            
        """
        Function to compute the mean ergodicity over n_iter iterations.
        """

        self.r_n_values_all = []

        for i in range(self.n_iter):

            seed = None if self.seed is None else self.seed + i
            erg = Ergodicity(self.L, self.Js, self.h, self.W, zblock=self.zblock, n_iter=1, alpha=self.alpha, seed=seed)
            erg.check_ergodicity()
            self.r_n_values_all.extend(erg.r_n_values)

        self.mean_r_n = np.mean(self.r_n_values_all)
        self.std_r_n_list = np.std(self.r_n_values_all)
        self.error_r_n = self.std_r_n_list / np.sqrt(len(self.r_n_values_all))

        return self.mean_r_n, self.std_r_n_list, self.error_r_n, self.r_n_values_all
    
    @staticmethod
    def gap_ratio_W_h_worker(L, Js, h, W, zblock, alpha, n_iter, seed):
        """
        Worker function to compute the mean gap ratio for a single combination of h, W, and seed.
        This function is used for parallel computation.
        """

        # Create an instance of the Ergodicity class
        erg = Ergodicity(L, Js, h, W, zblock, n_iter=n_iter, alpha=alpha, seed=seed)

        # Compute the mean gap ratio
        mean_gap_ratio, _, _, _ = erg.mean_ergodicity()

        return mean_gap_ratio
    
    
    @staticmethod
    def gap_ratio_W_h(L, Js, n_h, n_W, zblock, alpha, n_iter, seed=None, store=True, parallel=True):
        """
        Function to compute the gap ratio for a range of h and W values.
        Additional parameters:
        store (bool): whether to store the results in a file
        plot (bool): whether to plot the results
        parallel (bool): whether to use parallel computation
        Returns:
        gap (ndarray): gap ratio values
        """

        print(f"Gap ratio for L={L}, Js={Js}, n_h={n_h}, n_W={n_W}, zblock={zblock}, alpha={alpha}, n_iter={n_iter}, seed={seed}")
        print("Applying Hadamard transformation to the Hamiltonian...")

        y_list = np.logspace(-2, 2, num=n_W)  # disorder strength values
        x_list = np.logspace(-2, 2, num=n_h)  # transverse field values
        W_list = Js * y_list  # disorder strength values
        h_list = Js * x_list  # transverse field values
        gap = np.empty((n_h, n_W), dtype=float)  # store the gap ratio

        rng = np.random.default_rng(seed)  # random seed for reproducibility
        seeds = rng.integers(0, 1e9, size=n_h * n_W)  # generate random seeds for each iteration

        if parallel:
            
            # Prepare arguments for parallel computation
            args = [
                (L, Js, h, W, zblock, alpha, n_iter, seeds[k])
                for k, (h, W) in enumerate(itertools.product(h_list, W_list))
            ]

            # Use joblib.Parallel to parallelize the computation
            results = Parallel(n_jobs=-1)(  # use all available cores
                delayed(Ergodicity.gap_ratio_W_h_worker)(*arg) for arg in args
            )

            # Reshape results into a 2D array
            gap = np.array(results).reshape((n_h, n_W))

        else:

            k = 0

            # Sequential computation
            for i, h in enumerate(h_list):
                for j, W in enumerate(W_list):
                    erg = Ergodicity(L, Js, h, W, zblock, n_iter=n_iter, alpha=alpha, seed=seeds[k])
                    gap[i,j], _, _, _ = erg.mean_ergodicity()
                    k += 1

        if store:
            data = np.vstack((x_list, y_list, gap))
            np.save(f'results/data/gap_ratio_L_{L}', data)

        return gap
    
    @staticmethod
    def rn_distribution(L, Js, h, W, zblock, alpha, n_iter, seed=None, store=True):

        """
        Function to compute all the energly level gap ratios 
        """

        erg = Ergodicity(L, Js, h, W, n_iter=n_iter, zblock=zblock, alpha=alpha, seed=seed)
        _, _, _, rn_all = erg.mean_ergodicity()

        if store:
            np.savez_compressed(f'results/data/all_rn_h{h}_W{W}.npz', rn = rn_all)

        return rn_all