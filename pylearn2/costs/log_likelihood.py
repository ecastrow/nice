"""
Non-linear independent components estimation cost
"""
__authors__ = "Laurent Dinh"
__copyright__ = "Copyright 2014, Universite de Montreal"
__credits__ = ["Laurent Dinh", "Devon Hjelm"]
__license__ = "3-clause BSD"
__maintainer__ = "Laurent Dinh"
__email__ = "dinhlaur@iro"

from nice.pylearn2.models.mlp import Homothety
from nice.pylearn2.models.mlp import SigmaScaling
import operator
from pylearn2.costs.cost import Cost, DefaultDataSpecsMixin, NullDataSpecsMixin
from pylearn2.utils import wraps
import theano
import theano.tensor as T


class NegativeLogLikelihood(DefaultDataSpecsMixin, Cost):
    supervised = False

    @wraps(Cost.expr)
    def expr(self, model, data, **kwargs):
        space, sources = self.get_data_specs(model)
        space.validate(data)
        return -T.cast(model.get_log_likelihood(data).mean(),
                       theano.config.floatX)

class SigmaPenalty(NullDataSpecsMixin, Cost):
    """L1 regularization cost for Sigma.
    Sigma is 1/S = exp(-layer.D)

    coeff * sum(abs(weights)) for each set of weights.

    Parameters
    ----------
    coeff : float
        Decay hyperparameter
    """

    def __init__(self, coeff):
        self.__dict__.update(locals())
        del self.self

    def expr(self, model, data, ** kwargs):
        assert T.scalar() != 0.
        self.get_data_specs(model)[0].validate(data)
        layer = model.encoder.layers[-1]
        assert isinstance(layer, (Homothety, SigmaScaling))
        cost = layer.get_sigma_l1_decay(self.coeff)
        cost.name = "sigma_l1_penalty"
        return cost

    @wraps(Cost.is_stochastic)
    def is_stochastic(self):
        return False