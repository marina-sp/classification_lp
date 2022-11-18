from torch.autograd import Function
import torch
from torch.distributions import kl_divergence
from torch.distributions.multivariate_normal import MultivariateNormal


# Inherit from Function
class KLDivergence(Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input):
        ctx.save_for_backward(input, weight, bias)
        h_mean, h_vari, r_mean, r_vari, t_mean, t_vari = input
        batch_size = len(h_mean)
        embedding_dim = h_mean.shape[-1]
        device = h_mean.device

        # entity distribution
        mean = h_mean - t_mean
        vari = h_vari + t_vari
        cov = (torch.eye(embedding_dim, device=device) *
               vari.view(
                   batch_size, -1, 1, embedding_dim
               )
               ).squeeze(1)
        entity_dist = MultivariateNormal(mean, cov)

        # relation distribution
        r_cov = (torch.eye(embedding_dim, device=device) *
                 r_vari.view(
                     batch_size, -1, 1, embedding_dim
                 )
                 ).squeeze(1)

        # todo: find batch based distribution and divergence calculation
        relation_dist = MultivariateNormal(r_mean, r_cov)
        try:
            distances = kl_divergence(entity_dist, relation_dist)
        except:
            print(mean, vari, r_mean, r_vari)
            distances = torch.randn((batch_size,))
        return distances

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_bias