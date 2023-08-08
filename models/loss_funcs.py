import pyro
from pyro.infer import SVI, Trace_ELBO
import pyro.poutine as poutine
from pyro.infer.util import torch_item

""
class CustomTrace_ELBO(Trace_ELBO):
    # Overwrite Trace_ELBO so that retain_graph = True
    def loss_and_grads(self, model, guide, *args, **kwargs):
        loss = 0.0
        # grab a trace from the generator
        for model_trace, guide_trace in self._get_traces(model, guide, args, kwargs):
            loss_particle, surrogate_loss_particle = self._differentiable_loss_particle(
                model_trace, guide_trace
            )
            loss += loss_particle / self.num_particles

            trainable_params = any(

                # collect parameters to train from model and guide
               site["type"] == "param"
                for trace in (model_trace, guide_trace)
                for site in trace.nodes.values()
            )

            if trainable_params and getattr(
                surrogate_loss_particle, "requires_grad", False
            ):
                surrogate_loss_particle = surrogate_loss_particle / self.num_particles
                surrogate_loss_particle.backward(retain_graph=True) # We simply rewrite this line
#         warn_if_nan(loss, "loss")
        return loss


class CustomSVI(SVI):

    def step(self, weight, *args, **kwargs):

        # get loss and compute gradients
        with poutine.trace(param_only=True) as param_capture:
            loss = weight * self.loss_and_grads(self.model, self.guide, *args, **kwargs)

        params = set(
            site["value"].unconstrained() for site in param_capture.trace.nodes.values()
        )

        # actually perform gradient steps
        # torch.optim objects gets instantiated for any params that haven't been seen yet
        self.optim(params)

        # zero gradients
        pyro.infer.util.zero_grads(params)

        if isinstance(loss, tuple):
            # Support losses that return a tuple, e.g. ReweightedWakeSleep.
            return type(loss)(map(torch_item, loss))
        else:
            return torch_item(loss)