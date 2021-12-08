import tensorflow as tf
from tensorpotential.functions import radial_functions, spherical_harmonics
import numpy as np
import pandas as pd


class ACE():
    def __init__(self, potconfig, rcut=5, lmbda=5.25, nradmax=5, lmax=4, nradbase=12,
                 rankmax=2, ndensity=2, core_pre=0., core_lmbda=1., core_cut=100000.0, core_dcut=250.0,
                 fs_parameters = None, compute_smoothness=False): #, potconfile='./potfile.dat'
        self.rcut = rcut
        self.nradmax = nradmax
        self.nradbase = nradbase
        self.lmbda = lmbda
        self.lmax = lmax
        self.rankmax = rankmax
        self.ndensity = ndensity
        self.core_pre = core_pre
        self.core_lmbda = core_lmbda
        self.core_cut = core_cut
        self.core_dcut = core_dcut
        self.ranks_sizes = []
        self.tmp_coefs = None
        self.deltaSplineBins = 0.001
        self.embedingtype = 'FinnisSinclairShiftedScaled'
        #radbase
        self.radbasetype = "ChebExpCos" # "ChebPow"
        self.compute_smoothness = compute_smoothness
        if fs_parameters is not None:
            self.fs_parameters = fs_parameters
        else:
            self.fs_parameters = [1., 1., 1., 0.5][:2*self.ndensity]

        self.BbasisFuncs = None
        # Basis configuration
        if isinstance(potconfig, str):
            if '.dat' in potconfig:
                self._init_basis_configuration_from_dat_file(potconfig)
            else:
                self._init_basis_configuration_from_df(potconfig)
        else:
            self._init_basis_configuration_from_BBasisConf(potconfig)

        self.ncoef = 0
        ncrad = self.nradmax * (self.lmax + 1) * self.nradbase
        self.ranks_sizes.append(ncrad)
        for i in range(self.rankmax):
            self.ncoef += np.max(self.config.ranks[i].msum)
            self.ncoef += 1
            self.ranks_sizes.append((np.max(self.config.ranks[i].msum) + 1) * self.ndensity)
        self.ncoef *= self.ndensity
        self.ncoef += ncrad

        if self.tmp_coefs is None:
            np.random.seed(42)
            self.tmp_coefs = np.random.normal(0.001, 0.5, self.ncoef)
            #self.tmp_coefs[:self.nradmax * (self.lmax + 1) * self.nradbase] = self.eye().numpy().flatten()
        else:
            self.ncoef = len(self.tmp_coefs)

        self.fit_coefs = tf.Variable(self.tmp_coefs, dtype=tf.float64)

        if self.compute_smoothness:
            self.aux = []
        else:
            self.aux = None

    def compute_regularization(self):
        basis_coefs = self.fit_coefs[self.nradmax * (self.lmax + 1) * self.nradbase:]
        self.reg_l1 = tf.reduce_sum(tf.abs(basis_coefs))
        self.reg_l2 = tf.reduce_sum(basis_coefs**2)

    def _init_basis_configuration_from_dat_file(self, potconfile):
        self.config = ReadPotConfig(potconfile, self.lmax)

        for k in range(self.rankmax):
            if k == 0:
                self.config.ranks[k].apply_constrains(nmax=self.nradbase, lmax=0)
            else:
                self.config.ranks[k].apply_constrains(nmax=self.nradmax, lmax=self.lmax)

    def _init_basis_configuration_from_df(self, potconfile):
        if isinstance(potconfile, str):
            df = pd.read_pickle(potconfile)
        else:
            df = potconfile

        self.config = ConfigBasis(df, self.rankmax, self.lmax)

    def _init_basis_configuration_from_BBasisConf(self, bbasisfunconf):
        from pyace.basis import ACEBBasisFunction

        block = bbasisfunconf.funcspecs_blocks[0]  # potentialy iterable
        self.rcut = block.rcutij
        self.nradmax = block.nradmaxi
        self.nradbase = block.nradbaseij
        self.lmbda = block.radparameters[0]
        self.lmax = block.lmaxi
        self.fs_parameters = block.fs_parameters
        self.radbasetype = block.radbase
        self.deltaSplineBins = bbasisfunconf.deltaSplineBins

        if block.npoti is not None:
            if block.npoti in ['FinnisSinclairShiftedScaled', 'FinnisSinclair']:
                self.embedingtype = block.npoti
            else:
                raise NotImplementedError('Selected embedding function type is not implemented')
        else:
            raise ValueError('Embedding function type is not specified')
        # Core-rep parameters
        if block.core_rep_parameters is not None:
            assert len(block.core_rep_parameters) == 2, 'Lenth of the core_rep_parameters != 2'
            self.core_pre = block.core_rep_parameters[0]
            self.core_lmbda = block.core_rep_parameters[1]
        if block.rho_cut is not None:
            self.core_cut = block.rho_cut
        if block.drho_cut is not None:
            self.core_dcut = block.drho_cut

        crad_tmp = np.array(block.radcoefficients).flatten().astype(np.float64)
        bbasisfuncspecs = block.funcspecs
        bbasisfuncspecs = [BBasisFunc(ACEBBasisFunction(f)) for f in bbasisfuncspecs]
        self.rankmax = max([f.rank for f in bbasisfuncspecs])
        self.ndensity = len(bbasisfuncspecs[0].coefs)

        coefs_tmp = np.array([f.coefs for f in bbasisfuncspecs]).flatten().astype(np.float64)
        coefs_tmp[coefs_tmp==0] += 1e-20
        crad_tmp[crad_tmp==0] += 1e-20
        self.tmp_coefs = np.hstack((crad_tmp, coefs_tmp))

        self.config = ConfigBasis(bbasisfuncspecs, self.rankmax, self.lmax)

    def set_coefs(self, coefs):
        self.fit_coefs.assign(tf.Variable(coefs, dtype=tf.float64))

    def get_coefs(self):
        return self.fit_coefs

    def cheb_exp_cos(self, d_ij):
        d_ij_ch_domain = radial_functions.scale_distance(d_ij, tf.constant(self.lmbda, dtype=tf.float64), self.rcut)
        chebpol = radial_functions.chebvander(d_ij_ch_domain, self.nradbase)
        chebpol = tf.reshape(chebpol, [-1, self.nradbase])
        cos_cut = radial_functions.cutoff_func_cos(d_ij, self.rcut)
        gk = tf.where(tf.math.equal(chebpol, 1),
                      chebpol * cos_cut,
                      0.5 * (1 - chebpol) * cos_cut)

        y00 = 1  # / tf.sqrt(4 * tf.constant(np.pi, dtype=var_dtype))

        return gk * y00

    def cheb_pow(self, d_ij):
        d_ij_ch_domain = 2.0 * (1.0 - (1.0 - d_ij / self.rcut) ** self.lmbda) - 1.0

        chebpol = radial_functions.chebvander(d_ij_ch_domain, self.nradbase + 1)
        chebpol = tf.reshape(chebpol, [-1, self.nradbase + 1])
        chebpol = 0.5 - 0.5 * chebpol[:, 1:]
        res = tf.where(tf.less(d_ij, self.rcut), chebpol, tf.zeros_like(chebpol, dtype=tf.float64))

        return res

    def radial_function(self, d_ij):
        if self.radbasetype == 'ChebExpCos':
            return self.cheb_exp_cos(d_ij)
        elif self.radbasetype == 'ChebPow':
            return self.cheb_pow(d_ij)
        else:
            raise ValueError('Unknown radial function type {}'.format(self.radbasetype))

    def eye(self):
        e = []
        for i in range(self.nradmax):
            for j in range(self.lmax + 1):
                for t in range(self.nradbase):
                    if i == t:
                        e += [tf.constant(1.0, dtype=tf.float64)]
                    else:
                        e += [tf.constant(0.0, dtype=tf.float64)]
        e = tf.stack(e)
        return tf.reshape(e, [self.nradmax, self.lmax + 1, self.nradbase])

    def complexmul(self, r1, im1, r2, im2):
        real_part = r1 * r2 - im1 * im2
        imag_part = im2 * r1 + im1 * r2

        return real_part, imag_part

    def flat_gather_nd(self, params, indices):
        idx_shape = tf.shape(indices)
        params_shape = tf.shape(params)
        idx_dims = idx_shape[-1]
        gather_shape = params_shape[idx_dims:]
        params_flat = tf.reshape(params, tf.concat([[-1], gather_shape], axis=0))
        axis_step = tf.math.cumprod(params_shape[:idx_dims], exclusive=True, reverse=True)
        indices_flat = tf.reduce_sum(indices * axis_step, axis=-1)
        result_flat = tf.gather(params_flat, indices_flat)

        return tf.reshape(result_flat, tf.concat([idx_shape[:-1], gather_shape], axis=0))

    def compute_Bbasis_funcs_(self, a_nlm_r, a_nlm_i, rank):
        a_1_r = tf.gather_nd(a_nlm_r, self.config.ranks[rank - 1].munlm[0])
        a_2_r = tf.gather_nd(a_nlm_r, self.config.ranks[rank - 1].munlm[1])
        a_1_i = tf.gather_nd(a_nlm_i, self.config.ranks[rank - 1].munlm[0])
        a_2_i = tf.gather_nd(a_nlm_i, self.config.ranks[rank - 1].munlm[1])

        prod_r, prod_i = self.complexmul(a_1_r, a_1_i, a_2_r, a_2_i)

        if rank > 2:
            for k in range(2, rank):
                a_k_r = tf.gather_nd(a_nlm_r, self.config.ranks[rank - 1].munlm[k])
                a_k_i = tf.gather_nd(a_nlm_i, self.config.ranks[rank - 1].munlm[k])

                prod_r, prod_i = self.complexmul(prod_r, prod_i, a_k_r, a_k_i)

        b_base = prod_r * self.config.ranks[rank-1].genCG
        b_base = tf.transpose(tf.math.segment_sum(b_base, self.config.ranks[rank-1].msum), [1, 0])

        return b_base

    def compute_core_repulsion(self, d_ij, ind_i):
        phi_core = tf.math.exp(-self.core_lmbda * d_ij**2) / d_ij
        phi_core = self.core_pre * \
                   tf.math.segment_sum(phi_core * radial_functions.cutoff_func_cos(d_ij, self.rcut), ind_i)
        condition_1 = tf.math.less_equal(phi_core, self.core_cut - self.core_dcut)
        condition_2 = tf.math.less(phi_core, self.core_cut)
        inner_cut = tf.where(condition_1,
                            tf.ones_like(phi_core, dtype=tf.float64),
                            tf.where(condition_2,
                                    radial_functions.cutoff_func_cos(phi_core - self.core_dcut, self.core_dcut),
                                    tf.zeros_like(phi_core, dtype=tf.float64)))

        return phi_core, inner_cut

    def compute_Bbasis_funcs(self, a_nlm_r, a_nlm_i, rank):
        a_1_r = self.flat_gather_nd(a_nlm_r, self.config.ranks[rank - 1].munlm[0])
        a_2_r = self.flat_gather_nd(a_nlm_r, self.config.ranks[rank - 1].munlm[1])
        a_1_i = self.flat_gather_nd(a_nlm_i, self.config.ranks[rank - 1].munlm[0])
        a_2_i = self.flat_gather_nd(a_nlm_i, self.config.ranks[rank - 1].munlm[1])

        prod_r, prod_i = self.complexmul(a_1_r, a_1_i, a_2_r, a_2_i)

        if rank > 2:
            for k in range(2, rank):
                a_k_r = self.flat_gather_nd(a_nlm_r, self.config.ranks[rank - 1].munlm[k])
                a_k_i = self.flat_gather_nd(a_nlm_i, self.config.ranks[rank - 1].munlm[k])

                prod_r, prod_i = self.complexmul(prod_r, prod_i, a_k_r, a_k_i)

        b_base = prod_r * tf.convert_to_tensor(self.config.ranks[rank - 1].genCG, dtype=tf.float64)
        b_base = tf.transpose(tf.math.segment_sum(b_base, self.config.ranks[rank - 1].msum), [1, 0])

        return b_base

    def integrate(self, func, x, dx):
        f = tf.reshape(tf.reduce_sum(tf.abs(func), axis=[1, 2]), [-1, 1])
        # trapz = tf.reduce_sum( x**2 * tf.abs(func))/2*dx
        trapz = tf.reduce_sum(x ** 2 * f) * dx
        trapz /= self.rcut ** 2

        return tf.reshape(trapz, [-1, 1])

    def compute_B_basis(self, r_ij, ind_i):
        d_ij = tf.reshape(tf.linalg.norm(r_ij, axis=1), [-1, 1])
        rhat = r_ij / d_ij

        phi_core, inner_cut = self.compute_core_repulsion(d_ij, ind_i)

        sh = spherical_harmonics.SphericalHarmonics(self.lmax, prec='DOUBLE')
        ylm_r, ylm_i = sh.compute_ylm(rhat)
        ynlm_r = tf.expand_dims(ylm_r, 1) * \
                 tf.sqrt(4 * tf.constant(np.pi, dtype=tf.float64))
        ynlm_i = tf.expand_dims(ylm_i, 1) * \
                 tf.sqrt(4 * tf.constant(np.pi, dtype=tf.float64))

        crad = tf.reshape(self.fit_coefs[:self.nradmax * (self.lmax + 1) * self.nradbase],
                          [self.nradmax, self.lmax + 1, self.nradbase])

        if self.compute_smoothness:
            start = tf.constant(0, dtype=tf.float64)
            stop = tf.constant(self.rcut, dtype=tf.float64)
            d_cont = tf.reshape(tf.linspace(start, stop - 1e-5, 100), [-1, 1])
            delta_d = d_cont[1]-d_cont[0]
            with tf.GradientTape(persistent=False) as tape1:
                tape1.watch(d_cont)
                with tf.GradientTape(persistent=False) as tape2:
                    tape2.watch(d_cont)
                    g_cont = self.radial_function(d_cont)
                    r_nl_cont = tf.einsum('jk,nlk->jnl', g_cont, crad)
                    self.aux += [self.integrate(r_nl_cont, d_cont, delta_d)]
                dRnl_dr = tf.squeeze(tape2.batch_jacobian(r_nl_cont, d_cont), axis=-1)
                self.aux += [self.integrate(dRnl_dr, d_cont, delta_d)]
            d2Rnl_dr2 = tf.squeeze(tape1.batch_jacobian(dRnl_dr, d_cont), axis=-1)
            self.aux += [self.integrate(d2Rnl_dr2, d_cont, delta_d)]

        gk = self.radial_function(d_ij)
        rj_nl = tf.einsum('jk,nlk->jnl', gk, crad)
        # Rank1
        Br1 = tf.math.segment_sum(gk, ind_i)
        Br1 = tf.gather(tf.transpose(Br1, [1,0]), self.config.ranks[0].munlm[0][:,0])
        Br1 = tf.transpose(Br1, [1,0])

        rj_nlm = tf.gather(rj_nl, sh.l_tile, axis=2)

        aj_nlm_r = rj_nlm * ynlm_r
        aj_nlm_i = rj_nlm * ynlm_i
        a_nlm_r = tf.math.segment_sum(aj_nlm_r, ind_i)
        a_nlm_r = tf.transpose(a_nlm_r, [1, 2, 0])
        a_nlm_i = tf.math.segment_sum(aj_nlm_i, ind_i)
        a_nlm_i = tf.transpose(a_nlm_i, [1, 2, 0])

        bbasis_expansion = [Br1]
        for k in range(2, self.rankmax+1):
            bbasis_expansion += [self.compute_Bbasis_funcs(a_nlm_r, a_nlm_i, k)]

        return bbasis_expansion, phi_core, inner_cut

    def compute_atomic_energy(self, r_ij, ind_i):
        self.BbasisFuncs, phi_core, inner_cut = self.compute_B_basis(r_ij, ind_i)
        totalbasis = tf.concat(self.BbasisFuncs, axis=1, name='totalBasis_concat')
        fitcoefs = tf.reshape(self.fit_coefs[self.nradmax * (self.lmax + 1) * self.nradbase:], [-1, self.ndensity])
        en = tf.matmul(totalbasis, fitcoefs)
        safe_rho = tf.where(tf.not_equal(en, 0.), en, en + 1e-32)
        en_sum = 0
        for dens in range(self.ndensity):
            en_sum += tf.constant(self.fs_parameters[2*dens], dtype=tf.float64)\
                      * self.embedding_function(safe_rho[:, dens], tf.constant(self.fs_parameters[2*dens+1], dtype=tf.float64))

        e_atom = tf.math.add(tf.reshape(en_sum, [-1, 1]) * inner_cut, phi_core, 'atomic_energies')

        return e_atom

    def embedding_function(self, rho, mexp):
        if self.embedingtype == 'FinnisSinclairShiftedScaled':
            return self.f_exp_shsc(rho, mexp)
        elif self.embedingtype == 'FinnisSinclair':
            return self.f_exp_old(rho, mexp)

    def f_exp_old(self, rho, mexp):
        return tf.where(tf.less(tf.abs(rho), tf.constant(1e-10, dtype=tf.float64)), mexp * rho, self.en_func_old(rho, mexp))

    @staticmethod
    def en_func_old(rho, mexp):
        w = tf.constant(10., dtype=tf.float64)
        y1 = w * rho ** 2
        g = tf.where(tf.less(tf.constant(30., dtype=tf.float64), y1), 0. * rho, tf.exp(tf.negative(y1)))

        omg = 1. - g
        a = tf.abs(rho)
        y3 = tf.pow(omg * a+1e-20, mexp)
        y2 = mexp * g * a
        f = tf.sign(rho) * (y3 + y2)
        return f

    @staticmethod
    def f_exp_shsc(rho, mexp):
        eps = tf.constant(1e-10, dtype=tf.float64)
        @tf.function
        def func():
            f = tf.cond(tf.less(tf.abs(mexp - tf.constant(1., dtype=tf.float64)), eps),
                        lambda: rho,
                        lambda: tf.sign(rho)*(tf.sqrt(tf.abs(tf.abs(rho)
                                                     + 0.25*tf.exp(-tf.abs(rho)))) - 0.5*tf.exp(-tf.abs(rho)))
                        )
            return f
        return func()

    def selective_fitting(self, list_of_flags):
        try:
            if self.rankmax == 1:
                assert len(list_of_flags) == 1
            else:
                assert len(list_of_flags) == self.rankmax+1
        except:
            raise ValueError('Size of the requested configurations is not compatible with the potential rank')

        factor_list = []
        for i in range(self.rankmax+1):
            factor_list += [list_of_flags[i]]*self.ranks_sizes[i]

        return factor_list


@tf.custom_gradient
def en_func_cg(rho, mexp):
    mexp = tf.convert_to_tensor(mexp, dtype=tf.float64)
    w = tf.constant(10.0, dtype=tf.float64)
    y1 = w * rho * rho
    g = tf.where(tf.less(tf.constant(30., dtype=tf.float64), y1), tf.zeros_like(rho), tf.exp(tf.negative(y1)))

    omg = tf.constant(1., dtype=tf.float64) - g
    a = tf.abs(rho)
    y1 = tf.pow(omg * a, mexp)
    y2 = mexp * g * a

    def grad(dy):
        dg = tf.constant(-2.0, dtype=tf.float64) * w * rho * g
        da = tf.sign(rho) * tf.constant(1.0, dtype=tf.float64)
        dy11 = tf.where(tf.less(abs(y1), tf.constant(1e-10, dtype=tf.float64)), tf.zeros_like(rho), mexp * y1 / (omg * a))

        dy1 = dy11 * (-dg * a + omg * da)
        dy2 = mexp * (dg * a + g * da)
        DF = tf.sign(rho) * (dy1 + dy2)
        return DF*dy, None
    return tf.sign(rho) * (y1 + y2), grad

class ReadPotConfig(object):
    """Class for reading indexes and parameters for a given potential"""

    def __init__(self, filename, lmax):
        # self.nmax  = nmax
        self.lmax = lmax
        self.crad = None
        with open(filename) as f:
            self.lines = f.readlines()

        self.ranks = self.make_rank()

    def check_int(self, s):
        if s[0] in ('-', '+'):
            return s[1:].isdigit()
        return s.isdigit()

    def map_lines(self):
        c0 = 0
        c1 = 0
        rnks = []
        for n, line in enumerate(self.lines):
            if '#crad' in line:
                c0 = n + 1
            elif 'end crad' in line:
                c1 = n
            elif 'another rank' in line:
                # print('Rank++')
                rnks.append(n)

        crads = self.lines[c0:c1]
        crad = []
        for line in crads:
            data = line.split(' ')
            crad.append(np.array([float(d) for d in data])[-1:])

        if len(crad) > 0:
            self.crad = np.vstack(crad)

        ranks = []
        for i in range(len(rnks)):
            s = rnks[i] + 1
            if i + 1 == len(rnks):
                ranks += [self.lines[s:]]
            else:
                e = rnks[i + 1]
                ranks += [self.lines[s:e]]

        return ranks

    def read_rank_data(self, lines, rank):
        if rank == 1:
            cgc_i = 4
        else:
            cgc_i = 3 + rank

        rank_data = []
        for line in lines:
            data = line.split('\t')
            gcg = np.array(float(data[cgc_i].split('=')[1])).reshape(-1, 1)
            coefs = np.array([float(l.split('coef=')[1]) for l in data if 'coef=' in l]).reshape(-1, 1)
            f_ind = np.array(float(data[1].split('=')[1])).reshape(-1, 1)

            ind_acc = []
            for k in range(rank):
                ind = data[3 + k]
                ind = np.array([int(i) for i in ind.split() if self.check_int(i)]).reshape(-1, 1)
                ind[1] -= 1  # indexing should start from zero
                ind_acc.append(ind)
            ind_acc = np.vstack(ind_acc)
            # print(ind_acc, ind_acc.shape)
            comb_2 = np.concatenate((f_ind, ind_acc, gcg, coefs), axis=0)
            # print(comb_2, comb_2.shape)
            rank_data.append(comb_2.reshape(1, -1))
        rank_data = np.vstack(rank_data)
        rank_data[:, 0] -= np.min(rank_data[:, 0])

        #for i in range(rank):  # adjust m index
        #    rank_data[:, 4 + i * 4] += self.lmax

        # print(rank_data, rank_data.shape)
        return rank_data

    def make_rank(self):
        ranks_data = []
        ranks = self.map_lines()
        for i in range(len(ranks)):
            rdata = self.read_rank_data(ranks[i], i + 1)
            ranks_data.append(Rank(rdata, i + 1))

        return ranks_data


class Rank(object):
    """Clss of the Rank object"""

    def __init__(self, rank_data, rank):
        self.rank = rank
        self.munlm = self.get_munlm(rank_data)
        self.msum = self.get_msum(rank_data)
        self.genCG = self.get_genCG(rank_data)
        self.coefs = self.get_coefs(rank_data)

        for r in range(self.rank):
            self.munlm[r][:, -2] = merge_lm(self.munlm[r][:, -2], self.munlm[r][:, -1])
            self.munlm[r] = self.munlm[r][:, :-1]

    def get_munlm(self, data):
        nlms = []
        for i in range(self.rank):
            # nlms.append(tf.constant(data[:, 2+i*4:5+i*4], dtype=tf.int64)) #actually takes nlm not munlm
            nlms.append(np.array(data[:, 2 + i * 4:5 + i * 4]).astype(np.int32))  # actually takes nlm not munlm

        return nlms

    def get_msum(self, data):
        # return tf.constant(data[:, 0].reshape(-1,), dtype=tf.int64)
        return data[:, 0].reshape(-1, ).astype(np.int32)

    def get_genCG(self, data):
        # return tf.constant(data[:, 4*self.rank + 1].reshape(-1, 1), dtype=tf.float64)
        return data[:, 4 * self.rank + 1].reshape(-1, 1).astype(np.float64)

    def get_coefs(self, data):
        # return tf.constant(data[:, 4*self.rank + 2:], dtype=tf.float64)
        return data[:, 4 * self.rank + 2:].astype(np.float64)

    def rearrange_msum(self, msum):
        new_msum = []
        j = 0
        for i in range(len(msum)):
            if i == 0:
                new_msum += [j]
            else:
                if msum[i] == msum[i - 1]:
                    new_msum += [j]
                else:
                    j += 1
                    new_msum += [j]

        return np.array(new_msum).astype(np.int32)

    def apply_constrains(self, nmax=2, lmax=0):
        mask = np.ones_like(self.munlm[0][:, 0])
        for i in range(self.rank):
            cond = (self.munlm[i][:, 0] < nmax) & (self.munlm[i][:, 1] <= (lmax*(lmax+1)+lmax))
            mask = np.logical_and(mask, cond)

        for i in range(self.rank):
            self.munlm[i] = self.munlm[i][mask]

        self.msum = self.rearrange_msum(self.msum[mask] - np.min(self.msum[mask]))
        self.genCG = self.genCG[mask]
        self.coefs = self.coefs[mask]

class BBasisFunc():
    def __init__(self, bbasisfunc, lmax=None):
        self.rank = bbasisfunc.rank
        self.ns = bbasisfunc.ns
        self.ls = bbasisfunc.ls
        self.mu0 = bbasisfunc.mu0
        self.mus = bbasisfunc.mus
        self.genCG = np.reshape(bbasisfunc.gen_cgs, [-1, 1])
        self.ms = bbasisfunc.ms_combs
        self.coefs = bbasisfunc.coeffs

        self.munlm = self.get_munlm()
        self.msum = np.zeros(len(self.ms)).astype(np.int32)

        if lmax is not None:
            self.adjust_m_index(lmax)

    def get_munlm(self):
        munlm = [np.zeros((len(self.ms), 3)).astype(np.int32) for i in range(self.rank)]  # 4 if with mu
        for c in range(len(self.ms)):
            for r in range(self.rank):
                munlm[r][c] = np.array([self.ns[r] - 1, self.ls[r], self.ms[c][r]])

        return munlm

    def adjust_m_index(self, lmax):
        for r in range(self.rank):
            self.munlm[r][:,-1] += lmax


class BBasisFuncSet():
    def __init__(self, list_of_bbasisfunc, rank, lmax):
        self.rank = rank
        self.munlm = self.get_munlm(list_of_bbasisfunc)
        self.msum = self.get_msum(list_of_bbasisfunc)
        self.genCG = self.get_genCG(list_of_bbasisfunc)
        self.coefs = self.get_coefs(list_of_bbasisfunc)

        for r in range(self.rank):
            self.munlm[r][:, -2] = merge_lm(self.munlm[r][:, -2], self.munlm[r][:, -1])
            self.munlm[r] = self.munlm[r][:, :-1]

    def get_munlm(self, list_of_bbasisfunc):
        total_munlm = []
        for i in range(self.rank):
            total_munlm.append(np.vstack([bbasisfunc.munlm[i] for bbasisfunc in list_of_bbasisfunc]))

        return total_munlm

    def get_genCG(self, list_of_bbasisfunc):
        return np.vstack([bbasisfunc.genCG for bbasisfunc in list_of_bbasisfunc])

    def get_msum(self, list_of_bbasisfunc):
        return np.hstack([bbasisfunc.msum + i for i, bbasisfunc in enumerate(list_of_bbasisfunc)])

    def get_coefs(self, list_of_bbasisfunc):
        return np.vstack([bbasisfunc.coefs for bbasisfunc in list_of_bbasisfunc])

    def rearrange_msum(self, msum):
        new_msum = []
        j = 0
        for i in range(len(msum)):
            if i == 0:
                new_msum += [j]
            else:
                if msum[i] == msum[i - 1]:
                    new_msum += [j]
                else:
                    j += 1
                    new_msum += [j]

        return np.array(new_msum).astype(np.int32)

    def apply_constrains(self, nmax=2, lmax=0):
        mask = np.ones_like(self.munlm[0][:, 0])
        for i in range(self.rank):
            cond = (self.munlm[i][:, 0] < nmax) & (self.munlm[i][:, 1] <= lmax)
            mask = np.logical_and(mask, cond)

        for i in range(self.rank):
            self.munlm[i] = self.munlm[i][mask]

        self.msum = self.rearrange_msum(self.msum[mask] - np.min(self.msum[mask]))
        self.genCG = self.genCG[mask]
        #self.coefs = self.coefs[mask]


class ConfigBasis():
    def __init__(self, bbasisfunc, rankmax, lmax):
        self.ranks = []

        if isinstance(bbasisfunc, pd.DataFrame):
            self.set_basis_from_df(bbasisfunc, rankmax, lmax)
        elif isinstance(bbasisfunc, list):
            self.set_basis_from_list(bbasisfunc, rankmax, lmax)

    def set_basis_from_df(self, bbasisfunc, rankmax, lmax):
        for r in range(rankmax):
            rank_data = bbasisfunc.loc[bbasisfunc['rank'] == r + 1]
            rank_data = rank_data['func']
            # basisfuncs = rank_data.apply(BBasisFunc, lmax=lmax)
            basisfuncs = rank_data.tolist()
            basisset = BBasisFuncSet(basisfuncs, rank=r+1, lmax=lmax)
            self.ranks.append(basisset)

    def set_basis_from_list(self, bbasisfunc, rankmax, lmax):
        for r in range(rankmax):
            funcs_of_rank = [f for f in bbasisfunc if f.rank == r+1]
            basisset = BBasisFuncSet(funcs_of_rank, rank=r + 1, lmax=lmax)
            self.ranks.append(basisset)

class ConfigFromBBasisConf():
    def __init__(self, df, rankmax, lmax):
        self.ranks = []
        for r in range(rankmax):
            rank_data = df.loc[df['rank'] == r + 1]
            rank_data = rank_data['func']
            # basisfuncs = rank_data.apply(BBasisFunc, lmax=lmax)
            basisfuncs = rank_data.tolist()
            basisset = BBasisFuncSet(basisfuncs, rank=r+1, lmax=lmax)
            self.ranks.append(basisset)


def merge_lm(l, m):
    return l * (l + 1) + m