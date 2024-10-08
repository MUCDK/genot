# jax
import jax
import jax.numpy as jnp
import jax.random as random
from jax._src.prng import PRNGKeyArray

# numpyro
import numpyro.distributions as numpyro_dist

# typing
from typing import Any, Iterator, NamedTuple, Optional, Union
from typing_extensions import Literal

# sklearn
import sklearn.datasets


import abc


class Dataset(NamedTuple):
    """Samplers from source and target measures.

    Args:
      source_iter: loader for the source measure
      target_iter: loader for the target measure
    """
    source_iter: Iterator[jnp.ndarray]
    target_iter: Iterator[jnp.ndarray]


class Distribution(abc.ABC):
    """Probability distribution class.

    Args:
      batch_size: the batch used to define the iterator associated 
        to the probability distribution class. Note that one can sample
        an arbitrary number of samples using the method :meth:`generate_samples`.
      init_rng: initial PRNG key to define the iterator. 
    """

    def __init__(
        self,
        batch_size: int = 1_024,
        init_rng: Optional[jax.random.PRNGKeyArray] = None,
    ):
        self.batch_size = batch_size
        self.init_rng = (
            jax.random.PRNGKey(0)
            if init_rng is None else init_rng
        )

    @abc.abstractmethod
    def generate_samples(
        self, rng: jax.random.PRNGKeyArray, num_samples: int
    ) -> Any:
        """Args:
          rng: PRNG key.
          num_samples: number of samples to generate.
        Returns:
          Samples from the distribution and eventually more
            information about the samples, such as, when the 
            distribution is a mixture, the index of each component 
            of the mixture to which each generated samples is 
            associated to.
        """

    def _create_sample_generators(self) -> Iterator[jnp.array]:
        rng = self.init_rng
        while True:
            rng, _ = jax.random.split(rng)
            samples = self.generate_samples(
                rng,
                self.batch_size,
            )
            yield samples

    def __iter__(self) -> Iterator[jnp.array]:
        """Get an iterator defining a sample generator
        from the Probability distribution.

        Returns:
          The iterator.
        """
        return self._create_sample_generators()


class Gaussian(Distribution):
    """Gaussian probability distribution class.

    Args:
      mean: means of the Gaussian 
      cov: covariance of the Gaussian
      batch_size: batch size of the samples
      init_rng: initial PRNG key
    """

    def __init__(
        self,
        mean: jnp.ndarray,
        cov: jnp.ndarray,
        batch_size: int = 1_024,
        init_rng: Optional[jax.random.PRNGKeyArray] = None,
    ):
        super(Gaussian, self).__init__(
            batch_size=batch_size, init_rng=init_rng
        )
        self.setup(mean, cov)

    def setup(
        self, mean: jnp.ndarray, cov: jnp.ndarray,
    ):
        """Setup each feature of the distribution."""

        # check dimension consistency
        # and define each feature of the mixture distribution
        dim_mean = 1 if mean.ndim == 1 else len(mean)
        dim_cov = 1 if cov.ndim == 1 else cov.shape[1]
        assert (
            dim_mean == dim_cov
        ), (
            "Mean and covariance must have the same number of dimensions."
        )
        self.mean = mean
        self.cov = cov
        self.dim_data = dim_mean

        # define an associated numpyro distribution
        # for fast sampling
        self.gaussian_dist = numpyro_dist.MultivariateNormal(
            loc=self.mean, covariance_matrix=self.cov
        )

    def generate_samples(
        self,
        rng: jax.random.PRNGKeyArray,
        num_samples: int,
    ) -> jnp.array:
        """Sample generator from the distribution.
        Can be used to sample once the distribution, without
        having to define an iterator.

        Args:
          rng: PRNG key.
          num_samples: number of samples to generate.
        Returns:
          Samples from the distribution.
        """
        return self.gaussian_dist.sample(rng, (num_samples,))


class GaussianMixture(Distribution):
    """Mixture of Gaussian probability distribution class.

    Args:
      means: array of means of 
       each component of the mixture
      covs: array of covarainces of 
        each component of the mixture
      proportions: array of the proportions associated to 
        each componenent of the mixture
      batch_size: batch size of the samples
      init_rng: initial PRNG key
    """

    def __init__(
        self,
        means: jnp.ndarray,
        covs: jnp.ndarray,
        proportions: Optional[jnp.ndarray] = None,
        batch_size: int = 1_024,
        init_rng: Optional[jax.random.PRNGKeyArray] = None,
    ):
        super(GaussianMixture, self).__init__(
            batch_size=batch_size, init_rng=init_rng
        )
        self.setup(means, covs, proportions)

    def setup(
        self,
        means: jnp.ndarray,
        covs: jnp.ndarray,
        proportions: jnp.ndarray,
    ):
        """Setup each feature of the distribution."""

        # check dimension consistency
        # and define each feature of the mixture distribution
        self.dim_data = means.shape[1]
        if self.dim_data > 1:
           bool_check_dim = (means.shape[1] == covs.shape[2])
        else:
            bool_check_dim = (len(covs[0]) == 1)
        assert (
            bool_check_dim
        ), (
            "Means and covariances must have the same number of dimensions."
        )
        self.means = means
        self.covs = covs
        self.proportions = (
            proportions if proportions is not None
            else jnp.ones(len(means)) / len(means)
        )

        # check number of components consistency
        first = len(self.means)
        assert all(
            x == first for x in [
                len(self.covs),
                len(self.proportions)
            ]
        ), (
            "Inconsistency in the number of components in the mixture."
        )

        # define an associated numpyro distribution
        # for fast sampling
        self.mixture_dist = numpyro_dist.MixtureSameFamily(
            mixing_distribution=numpyro_dist.Categorical(
                probs=self.proportions
            ),
            component_distribution=numpyro_dist.MultivariateNormal(
                loc=self.means, covariance_matrix=self.covs
            ),
        )

    def generate_samples(
        self,
        rng: jax.random.PRNGKeyArray,
        num_samples: int,
        return_indices: bool = False,
    ) -> jnp.array:
        """Sample generator from the distribution.
        Can be used to sample once the distribution, without
        having to define an iterator.

        Can be used to samples points by returning 
        also the component of the mixture to which each generated sample 
        is asscoiated. This is useful for plotting purposes.
        Args:
          rng: PRNG key.
          num_samples: number of samples to generate.
          return_indices: whether to return the indices corresponding to
            the component of the mixture to which each generated sample is asscoiated.
        Returns:
          Samples from the probability distribution, and if ``return_indices`` 
            is ``True``, the associated components.
        """

        samples, indices = self.mixture_dist.sample_with_intermediates(
            rng, sample_shape=(num_samples,)
        )
        to_return = (
            samples if not return_indices
            else (samples, jnp.squeeze(jnp.array(indices)))
        )
        return to_return


class BallUniform(Distribution):
    """Uniform probability distribution on a ball class.

    Args:
      dim_data: dimension of the data
      radius: radius of the ball
      batch_size: batch size of the samples
      init_rng: initial PRNG key
    """

    def __init__(
        self,
        dim_data: int,
        radius: float = 1.,
        batch_size: int = 1_024,
        init_rng: Optional[jax.random.PRNGKeyArray] = None,
    ):
        self.dim_data = dim_data
        super(BallUniform, self).__init__(
            batch_size=batch_size, init_rng=init_rng
        )
        self.setup(radius)

    def setup(
        self, radius: float,
    ):
        """Setup each feature of the distribution."""
        assert radius > 0, "Radius must be positive."
        self.radius = radius

        # define numpyro distributions
        # to sample from the uniform distribution on the ball
        self.uniform_dist = numpyro_dist.Uniform(low=0., high=self.radius)
        self.stand_gaussian_dist = numpyro_dist.MultivariateNormal(
            loc=jnp.zeros(self.dim_data),
            covariance_matrix=jnp.eye(self.dim_data)
        )

    def generate_samples(
        self,
        rng: jax.random.PRNGKeyArray,
        num_samples: int,
    ) -> jnp.array:
        """Sample generator from the distribution.
        Can be used to sample once the distribution, without
        having to define an iterator.

        Args:
          rng: PRNG key.
          num_samples: number of samples to generate.
        Returns:
          Samples from the probability distribution.
        """
        r = self.uniform_dist.sample(rng, sample_shape=(num_samples,))
        scaled_r = r ** (1 / self.dim_data)
        gauss_vec = self.stand_gaussian_dist.sample(
            rng, sample_shape=(num_samples,)
        )
        scaled_gauss_vec = (
            1 / jnp.linalg.norm(gauss_vec, axis=1, keepdims=True)
        ) * gauss_vec
        samples = scaled_r[:, None] * scaled_gauss_vec
        return samples


class SphereUniform(Distribution):
    """Uniform probability distribution on a sphere class.

    On the contrary to the ``SphereDistribution``class,
    it can be used for other dimension than 2.
    Args:
      dim_data: dimension of the data
      radius: radius of the ball
      batch_size: batch size of the samples
      init_rng: initial PRNG key
    """

    def __init__(
        self,
        dim_data: int,
        radius: float = 1.,
        batch_size: int = 1_024,
        init_rng: Optional[jax.random.PRNGKeyArray] = None,
    ):
        self.dim_data = dim_data
        super(SphereUniform, self).__init__(
            batch_size=batch_size, init_rng=init_rng
        )
        self.setup(radius)

    def setup(
        self, radius: float,
    ):
        """Setup each feature of the distribution."""
        assert radius > 0, "Radius must be positive."
        self.radius = radius

        # define numpyro distributions
        # to sample from the uniform distribution on the ball
        self.stand_gaussian_dist = numpyro_dist.MultivariateNormal(
            loc=jnp.zeros(self.dim_data),
            covariance_matrix=jnp.eye(self.dim_data)
        )

    def generate_samples(
        self,
        rng: jax.random.PRNGKeyArray,
        num_samples: int,
    ) -> jnp.array:
        """Sample generator from the distribution.
        Can be used to sample once the distribution, without
        having to define an iterator.

        Args:
          rng: PRNG key.
          num_samples: number of samples to generate.
        Returns:
          Samples from the probability distribution.
        """
        gauss_vec = self.stand_gaussian_dist.sample(
            rng, sample_shape=(num_samples,)
        )
        samples = (
            1 / jnp.linalg.norm(gauss_vec, axis=1, keepdims=True)
        ) * gauss_vec
        return samples


class RectangleUniform(Distribution):
    """
    Uniform distribution on limits-rectangle.
    Defines jittable sampler with NumPyro distributions to reach fast sampling.
    """

    def __init__(
        self,
        limits: jnp.ndarray,
        batch_size: int = 1_024,
        init_rng: Optional[jax.random.PRNGKeyArray] = None,
    ):
        super(RectangleUniform, self).__init__(
            batch_size=batch_size, init_rng=init_rng
        )
        self.setup(limits)

    def setup(self, limits: jnp.ndarray):
        """Setup each feature of the distribution."""
        limits = jnp.asarray(limits)
        assert (
            limits.ndim == 2
            and
            limits.shape[1] == 2
        ), (
            "Limits must be a 2D array where the i-th "
            "row is the 2D vector of the lower and upper limits "
            "of the uniform distribution on the i-th dimension."
        )
        self.limits = limits

        # dimensionality of the data
        self.dim_data = len(limits)

        # define numpyro distributions
        self.uniform_dist = numpyro_dist.Uniform(
            low=self.limits[:, 0], high=self.limits[:, 1]
        )

    def generate_samples(
        self,
        rng: jax.random.PRNGKeyArray,
        num_samples: int,
    ) -> jnp.array:
        """Sample generator from the distribution.
        Can be used to sample once the distribution, without
        having to define an iterator.

        Args:
          rng: PRNG key.
          num_samples: number of samples to generate.
        Returns:
          Samples from the probability distribution.
        """
        return self.uniform_dist.sample(
            rng, sample_shape=(num_samples,)
        )


class SphereDistribution(Distribution):
    """Probability distribution on the 2-sphere class.

    The distribution is defined distributions on the colatitude 
    :math:`\theta` and the latitude :math:`\phi` of the sphere.
    These distribution must be either a ``Distribution`` or a numpyro
    distribution, i.e. an object of 
    {class}`~numpyro.distributions.Distribution."
    Args:
      theta_dist: distribution on the colatitude
      phi_dist: distribution on the latitude
      dim_data: dimension of the data
      radius: radius of the ball
      batch_size: batch size of the samples
      init_rng: initial PRNG key
    """

    def __init__(
        self,
        theta_dist: Union[Distribution, numpyro_dist.Distribution],
        phi_dist: Union[Distribution, numpyro_dist.Distribution],
        radius: float = 1.,
        batch_size: int = 1_024,
        init_rng: Optional[jax.random.PRNGKeyArray] = None,
    ):
        super(SphereDistribution, self).__init__(
            batch_size=batch_size, init_rng=init_rng
        )
        self.setup(theta_dist, phi_dist, radius)

    def setup(
        self,
        theta_dist: Union[Distribution, numpyro_dist.Distribution],
        phi_dist: Union[Distribution, numpyro_dist.Distribution],
        radius: float,
    ):
        """Setup each feature of the distribution."""
        assert radius > 0, "Radius must be positive."
        self.radius = radius

        # data is 3 dimensional by construction
        self.dim_data = 3

        # distributions on the colatitude and the latitude
        assert all(
            [
                isinstance(dist, Distribution)
                or
                isinstance(dist, numpyro_dist.Distribution)
                for dist in [theta_dist, phi_dist]
            ]
        ), (
            "Distributions on the colatitude and the latitude "
            "must be either a ``Distribution`` or a numpyro "
            "distribution, i.e. an object of "
            "{class}`~numpyro.distributions.Distribution`."
        )
        self.theta_dist = theta_dist
        self.phi_dist = phi_dist

        # boolean to assert if the distributions
        # of the latitude and colatitude are mixtures
        # to eventually return the indices to which
        # each sample is associated to while generated samples
        self.theta_is_mixture = (
            isinstance(self.theta_dist, numpyro_dist.MixtureSameFamily)
        )
        self.phi_is_mixture = (
            isinstance(self.phi_dist, numpyro_dist.MixtureSameFamily)
        )
        assert not (
            self.theta_is_mixture
            and 
            self.phi_is_mixture
        ), (
            "Distributions of the latitude and colatitude "
            "cannot be both mixtures."
        )

    def generate_samples(
        self,
        rng: jax.random.PRNGKeyArray,
        num_samples: int,
        return_indices: bool = False,
    ):
        """Sample generator from the distribution.
        Can be used to sample once the distribution, without
        having to define an iterator.
        
        When the distribution defined on the sphere is a mixture,
        can be used to samples points by returning 
        also the component of the mixture to which each generated sample 
        is asscoiated. 
        This is useful for plotting purposes.
        Args:
          rng: PRNG key.
          num_samples: number of samples to generate.
          return_indices: whether to return the indices corresponding to
            the component of the mixture to which each generated sample is asscoiated.
        Returns:
          Samples from the probability distribution, and if ``return_indices`` 
            is ``True``, the associated components.
        """
        
        # sample from colatitude and latitude distributions
        rng_theta, rng_phi = random.split(rng)
        samples_theta, indices_theta = self.theta_dist.sample_with_intermediates(
            rng_theta, sample_shape=(num_samples,)
        )
        samples_phi, indices_phi = self.phi_dist.sample_with_intermediates(
            rng_phi, sample_shape=(num_samples,)
        )
        samples_phi = jnp.squeeze(samples_phi)
        samples_theta = jnp.squeeze(samples_theta)

        # get samples from radius, colatitude and latitude
        samples = jnp.concatenate(
            (
                (jnp.sin(samples_phi) * jnp.cos(samples_theta))[:, None],
                (jnp.sin(samples_phi) * jnp.sin(samples_theta))[:, None],
                (jnp.cos(samples_phi))[:, None]
            ),
            axis=1
        )
        indices = (
            indices_theta if self.theta_is_mixture
            else indices_phi if self.phi_is_mixture
            else None
        )
        to_return = (
            samples if not return_indices
            else (samples, jnp.squeeze(jnp.array(indices)))
        )
        return to_return

Name_t = Literal[
    "moon_upper", "moon_lower",
    "circle", "s_curve", "swiss"
]
class SklearnDistribution(Distribution):
    """Probability distribution defined with sklearn datasets.

    One can rotate the distribution by setting the ``theta_rotation``.
    Args:
      name: name of the correspondinf sklearn dataset 
      translation: vector to translate the distribution,
        since not all dataset has zero mean, it might not 
        correspond to the mean of the distribution.
      batch_size: batch size of the samples
      init_rng: initial PRNG key
    """

    def __init__(
        self,
        name: Name_t,
        dim_data: int = 2,
        translation: Optional[jnp.ndarray] = None,
        theta_rotation: Optional[float] = None,
        scale: float = 1.,
        factor: float = .5,
        noise: float = .01,
        batch_size: int = 1_024,
        init_rng: Optional[jax.random.PRNGKeyArray] = None,
    ):
        self.name = name
        super(SklearnDistribution, self).__init__(
            batch_size=batch_size, init_rng=init_rng
        )
        self.setup(
            name,
            theta_rotation,
            scale,
            factor,
            noise,
            translation,
            dim_data
        )

    def setup(
        self, 
        name: Name_t,
        theta_rotation: Optional[float],
        scale: float,
        factor: float,
        noise: float,
        translation: Optional[jnp.ndarray] = None,
        dim_data: int = 2,
    ):
        """Setup each feature of the distribution."""
        assert name in [
            "moon_upper", "moon_lower",
            "circle", "s_curve", "swiss"
        ], (
            f"SklearnDistribution `{name}` not implemented."
        )
        assert (
            scale > 0
            and 
            noise > 0
            and
            factor > 0
        ), (
            "Noise and scale parameters must be positive."
        )
        self.noise = noise
        self.scale = scale
        self.factor = factor
        self.dim_data = dim_data

        # set translation vector 
        self.translation = (
            jnp.zeros(self.dim_data) if translation is None else translation
        )
        
        # define rotation matrix to rotate distribution
        if self.dim_data != 2:
            assert dim_data == 3, (
                "Data dimension must be 2 or 3."
            )
            assert theta_rotation is None, (
                "Rotation not provided for 3d data."
            )
        if theta_rotation is not None:
            self.rotation = jnp.array(
                [
                    [jnp.cos(theta_rotation), -jnp.sin(theta_rotation)],
                    [jnp.sin(theta_rotation), jnp.cos(theta_rotation)]
                ]
            )
        else:
            self.rotation = None
            

    def generate_samples(
        self, rng: PRNGKeyArray, num_samples: int, return_colors: bool = False
    ):
        """Samples generator."""
        seed = jax.random.randint(rng, [], minval=0, maxval=1e5).item()

        if self.name == "moon_upper":
            samples, colors = sklearn.datasets.make_moons(
                n_samples=[num_samples, 0], 
                random_state=seed,
                noise=self.noise,
            )

        elif self.name == "moon_lower":
            samples, colors = sklearn.datasets.make_moons(
                n_samples=[0, num_samples], 
                random_state=seed,
                noise=self.noise,
            )

        elif self.name == "circle":
            samples, colors = sklearn.datasets.make_circles(
                n_samples=[num_samples, 0],
                factor=self.factor, 
                noise=self.noise, 
                random_state=seed
            )

        elif self.name == "s_curve":
            X, colors = sklearn.datasets.make_s_curve(
                num_samples, 
                noise=self.noise, 
                random_state=seed
            )
            samples = X if self.dim_data == 3 else X[:, [2, 0]]

        elif self.name == "swiss":
            X, colors = sklearn.datasets.make_swiss_roll(
                num_samples, 
                noise=self.noise, 
                random_state=seed
            )
            samples = X if self.dim_data == 3 else X[:, [2, 0]]
        
        samples = self.scale * samples + self.translation
        if self.rotation is not None:
            samples = jnp.squeeze(
                jnp.matmul(self.rotation[None, :], samples.T).T
            )
        to_return = (
            (samples, colors) if return_colors
            else samples
        )

        return to_return
    



