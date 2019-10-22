"""
**For most use cases, this can just be considered an internal class and
ignored.**

This module contains the abstract class AttackerStep as well as a few subclasses. 

AttackerStep is a generic way to implement optimizers specifically for use with
:class:`robustness.attacker.AttackerModel`. In general, except for when you want
to :ref:`create a custom optimization method <adding-custom-steps>`, you probably do not need to
import or edit this module and can just think of it as internal.
"""

import torch as ch

class AttackerStep:
    '''
    Generic class for attacker steps, under perturbation constraints
    specified by an "origin input" and a perturbation magnitude.
    Must implement project, step, and random_perturb
    '''
    def __init__(self, orig_input, eps, step_size, use_grad=True, **kwargs):
        '''
        Initialize the attacker step with a given perturbation magnitude.

        Args:
            eps (float): the perturbation magnitude
            orig_input (ch.tensor): the original input
        '''
        self.orig_input = orig_input
        self.eps = eps
        self.step_size = step_size
        self.use_grad = use_grad
        self.kwargs = kwargs

    def project(self, x):
        '''
        Given an input x, project it back into the feasible set

        Args:
            ch.tensor x : the input to project back into the feasible set.

        Returns:
            A `ch.tensor` that is the input projected back into
            the feasible set, that is,
        .. math:: \min_{x' \in S} \|x' - x\|_2
        '''
        raise NotImplementedError

    def step(self, x, g):
        '''
        Given a gradient, make the appropriate step according to the
        perturbation constraint (e.g. dual norm maximization for :math:`\ell_p`
        norms).

        Parameters:
            g (ch.tensor): the raw gradient

        Returns:
            The new input, a ch.tensor for the next step.
        '''
        raise NotImplementedError

    def random_perturb(self, x):
        '''
        Given a starting input, take a random step within the feasible set
        '''
        raise NotImplementedError

    def to_image(self, x):
        '''
        Given an input (which may be in an alternative parameterization),
        convert it to a valid image (this is implemented as the identity
        function by default as most of the time we use the pixel
        parameterization, but for alternative parameterizations this functino
        must be overriden).
        '''
        return x

### Instantiations of the AttackerStep class

# L-infinity threat model
class LinfStep(AttackerStep):
    """
    Attack step for :math:`\ell_\infty` threat model. Given :math:`x_0`
    and :math:`\epsilon`, the constraint set is given by:

    .. math:: S = \{x | \|x - x_0\|_\infty \leq \epsilon\}
    """
    def project(self, x):
        """
        """
        diff = x - self.orig_input
        diff = ch.clamp(diff, -self.eps, self.eps)
        return ch.clamp(diff + self.orig_input, 0, 1)

    def step(self, x, g):
        """
        """
        step = ch.sign(g) * self.step_size
        return x + step

    def random_perturb(self, x):
        """
        """
        new_x = x + 2 * (ch.rand_like(x) - 0.5) * self.eps
        return ch.clamp(new_x, 0, 1)

# L2 threat model
class L2Step(AttackerStep):
    """
    Attack step for :math:`\ell_2` threat model. Given :math:`x_0`
    and :math:`\epsilon`, the constraint set is given by:

    .. math:: S = \{x | \|x - x_0\|_2 \leq \epsilon\}
    """
    def project(self, x):
        """
        """
        diff = x - self.orig_input
        diff = diff.renorm(p=2, dim=0, maxnorm=self.eps)
        return ch.clamp(self.orig_input + diff, 0, 1)

    def step(self, x, g):
        """
        """
        # Scale g so that each element of the batch is at least norm 1
        l = len(x.shape) - 1
        g_norm = ch.norm(g.view(g.shape[0], -1), dim=1).view(-1, *([1]*l))
        scaled_g = g / (g_norm + 1e-10)
        return x + scaled_g * self.step_size

    def random_perturb(self, x):
        """
        """
        new_x = x + (ch.rand_like(x) - 0.5).renorm(p=2, dim=1, maxnorm=self.eps)
        return ch.clamp(new_x, 0, 1)

# Unconstrained threat model
class UnconstrainedStep(AttackerStep):
    """
    Unconstrained threat model, :math:`S = [0, 1]^n`.
    """
    def project(self, x):
        """
        """
        return ch.clamp(x, 0, 1)

    def step(self, x, g):
        """
        """
        return x + g * self.step_size

    def random_perturb(self, x):
        """
        """
        new_x = x + (ch.rand_like(x) - 0.5).renorm(p=2, dim=1, maxnorm=step_size)
        return ch.clamp(new_x, 0, 1)

class FourierStep(AttackerStep):
    """
    Step under the Fourier (decorrelated) parameterization of an image.

    See https://distill.pub/2017/feature-visualization/#preconditioning for more information.
    """
    def project(self, x):
        """
        """
        return x

    def step(self, x, g):
        """
        """
        return x + g * self.step_size

    def random_perturb(self, x):
        """
        """
        return x

    def to_image(self, x):
        """
        """
        return ch.sigmoid(ch.irfft(x, 2, normalized=True, onesided=False))

# L1 threat model
class L1Step(AttackerStep):
    """
    Attack step for :math:`\ell_1` threat model. Given :math:`x_0`
    and :math:`\epsilon`, the constraint set is given by:

    .. math:: S = \{x | \|x - x_0\|_1 \leq \epsilon\}

    Uses SLIDE instead of plain PGD for :math:`\ell_1` norm
    Paper link: (Tramer and Boneh 2019): https://arxiv.org/pdf/1904.13000.pdf
    """
    def project(self, x):
        """
        """
        diff = x - self.orig_input
        diff = diff.renorm(p=1, dim=0, maxnorm=self.eps)
        return ch.clamp(self.orig_input + diff, 0, 1)

    def step(self, x, g):
        """
        """
        grad_view = g.view(g.shape[0], -1)
        abs_grad  = ch.abs(grad_view)
        sign      = ch.sign(grad_view)

        q_range = self.kwargs['percentile_range']
        q = q_range[0] + ch.rand(1)[0] * (q_range[1] - q_range[0])
        k = int(q * abs_grad.shape[1])

        percentile_value, _ = ch.kthvalue(abs_grad, k, keepdim=True)
        percentile_value = percentile_value.repeat(1, grad_view.shape[1])
        tied_for_max = (abs_grad >= percentile_value)
        num_ties = ch.sum(tied_for_max, dim=1, keepdim=True)

        e  = (sign * tied_for_max) / num_ties
        e  =  e.view(g.shape)

        return x + e * self.step_size

    def random_perturb(self, x):
        """
        """
        m = ch.distributions.laplace.Laplace(ch.tensor([0.0]), torch.tensor([1.0]))
        new_x = x + m.sample(x.shape).renorm(p=1, dim=1, maxnorm=self.eps)
        return ch.clamp(new_x, 0, 1)
