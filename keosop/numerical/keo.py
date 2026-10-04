import functools
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from scipy import constants

jax.config.update("jax_enable_x64", True)

G_to_invcm = (
    constants.value("Planck constant")
    * constants.value("Avogadro constant")
    * 1e16
    / (4.0 * np.pi**2 * constants.value("speed of light in vacuum"))
    * 1e5
)

EPS = jnp.array(
    [
        [[int((i - j) * (j - k) * (k - i) * 0.5) for k in range(3)] for j in range(3)]
        for i in range(3)
    ],
    dtype=jnp.float64,
)


def com(masses: np.ndarray):
    """Wrapper function for `internal_to_cartesian` that computes the Cartesian coordinates
    of atoms from given internal coordinates and shifts them to the center of mass.

    Args:
        masses (np.ndarray): An array containing the masses of the atoms. The order of atoms
            in `masses` must match the order in the output of `internal_to_cartesian`.

    Returns:
        A function that first computes the Cartesian coordinates using `internal_to_cartesian`
        and then shifts them to the center of mass.
    """

    def wrapper(internal_to_cartesian):
        @functools.wraps(internal_to_cartesian)
        def wrapper_com(*args, **kwargs):
            xyz = internal_to_cartesian(*args, **kwargs)
            assert len(xyz) == len(masses), (
                "The number of elements in 'masses' must match the leading dimension of the array "
                "returned by the 'internal_to_cartesian' function"
            )
            masses_ = jnp.asarray(masses)
            com = masses_ @ xyz / jnp.sum(masses_)
            return xyz - com[None, :]

        return wrapper_com

    return wrapper


@functools.partial(jax.jit, static_argnums=2)
def gmat(q, masses, internal_to_cartesian):
    # xyz_g = jax.jacfwd(internal_to_cartesian)(jnp.asarray(q))
    xyz_g = jax.jacrev(internal_to_cartesian)(jnp.asarray(q))
    tvib = xyz_g
    xyz = internal_to_cartesian(jnp.asarray(q))
    trot = jnp.transpose(EPS @ xyz.T, (2, 0, 1))
    ttra = jnp.array([jnp.eye(3, dtype=jnp.float64) for _ in range(len(xyz))])
    tvec = jnp.concatenate((tvib, trot, ttra), axis=2)
    masses_sq = jnp.sqrt(jnp.asarray(masses))
    tvec = tvec * masses_sq[:, None, None]
    tvec = jnp.reshape(tvec, (len(xyz) * 3, len(q) + 6))
    return tvec.T @ tvec


@functools.partial(jax.jit, static_argnums=2)
def Gmat(
    q: np.ndarray,
    masses: np.ndarray,
    internal_to_cartesian: Callable[[jnp.ndarray], jnp.ndarray],
):
    """Computes the kinetic energy G-matrix for a molecular system.

    Args:
        q (np.ndarray): An array of internal coordinates with shape (3N-6,),
            where N is the number of atoms. Bond lengths are given in Angstroms,
            and angles are in radians.
        masses (np.ndarray): A 1D array containing the atomic masses. The order of atoms
            in `masses` must match the order of atoms in the output of `internal_to_cartesian`.
        internal_to_cartesian (Callable): A function that converts internal coordinates `q`
            into Cartesian coordinates, returning an array of shape (number of atoms, 3).

    Returns:
        np.ndarray: A square matrix of shape (ncoo+3+3, ncoo+3+3), representing the elements
        of the kinetic energy G-matrix. The first `ncoo` rows and columns correspond to
        vibrational coordinates, followed by three rotational and three translational
        coordinates. The units of the G-matrix are inverse centimeters.
    """
    return jnp.linalg.inv(gmat(q, masses, internal_to_cartesian)) * G_to_invcm


batch_Gmat = jax.jit(jax.vmap(Gmat, in_axes=(0, None, None)), static_argnums=2)


@functools.partial(jax.jit, static_argnums=(2,))
def jac_Gmat(q, masses, internal_to_cartesian):
    return jax.jacfwd(Gmat)(q, masses, internal_to_cartesian)


@functools.partial(jax.jit, static_argnums=(2,))
def jac_Gmat_vib(q, masses, internal_to_cartesian):
    nq = len(q)
    return jax.jacfwd(lambda *arg: Gmat(*arg)[:nq, :nq])(
        q, masses, internal_to_cartesian
    )


@jax.jit
def log_abs_det(a):
    return jnp.log(jnp.abs(jnp.linalg.det(a)))


@functools.partial(jax.jit, static_argnums=(2,))
def det_gmat(q, masses, internal_to_cartesian):
    nq = len(q)
    return jnp.linalg.det(gmat(q, masses, internal_to_cartesian)[: nq + 3, : nq + 3])


@functools.partial(jax.jit, static_argnums=(2,))
def log_abs_det_gmat(q, masses, internal_to_cartesian):
    nq = len(q)
    return log_abs_det(gmat(q, masses, internal_to_cartesian)[: nq + 3, : nq + 3])


@functools.partial(jax.jit, static_argnums=(2,))
def jac_log_abs_det_gmat(q, masses, internal_to_cartesian):
    return jax.jacrev(log_abs_det_gmat)(q, masses, internal_to_cartesian)


@functools.partial(jax.jit, static_argnums=(2,))
def hess_log_abs_det_gmat(q, masses, internal_to_cartesian):
    return jax.jacfwd(jax.jacrev(log_abs_det_gmat))(q, masses, internal_to_cartesian)


@functools.partial(jax.jit, static_argnums=(2,))
def pseudo(
    q: np.ndarray,
    masses: np.ndarray,
    internal_to_cartesian: Callable[[jnp.ndarray], jnp.ndarray],
):
    """Pseudopotential implementation according to Eq. (21)
    in Edit Mátyus, Gábor Czakó, and Attila G. Császár,
    J. Chem. Phys. 130, 134112 (2009)
    http://dx.doi.org/10.1063/1.3076742
    """
    nq = len(q)
    G = Gmat(q, masses, internal_to_cartesian)[:nq, :nq]
    dG = jac_Gmat_vib(q, masses, internal_to_cartesian)
    k = jnp.arange(nq)
    dG = dG[k, :, k]
    dlogdet = jac_log_abs_det_gmat(q, masses, internal_to_cartesian)
    hlogdet = hess_log_abs_det_gmat(q, masses, internal_to_cartesian)
    pseudo1 = dlogdet @ G @ dlogdet
    pseudo2 = jnp.sum(dG @ dlogdet) + jnp.sum(G * hlogdet)
    return (pseudo1 + 4 * pseudo2) / 32.0


batch_pseudo = jax.jit(jax.vmap(pseudo, in_axes=(0, None, None)), static_argnums=2)
