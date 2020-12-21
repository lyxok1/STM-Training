from torch.nn.modules.module import Module
from torch.autograd import Function, Variable
import warp_cuda
import torch

class WarpFunction(Function):

    @staticmethod
    def forward(ctx, input1, input2):
        assert input1.is_contiguous()
        assert input2.is_contiguous()
        assert input1.is_cuda and input2.is_cuda

        ctx.save_for_backward(input1, input2)

        _, c, _, _ = input1.size()
        b, _, h, w = input2.size()
        output = input1.new(b, c, h, w).zero_()

        warp_cuda.forward(input1, input2, output)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.contiguous()
        assert grad_output.is_contiguous()

        input1, input2 = ctx.saved_tensors

        grad_input1 = Variable(input1.new(input1.size()).zero_())
        grad_input2 = Variable(input1.new(input2.size()).zero_())

        warp_cuda.backward(input1, input2, grad_output.data,
                            grad_input1.data, grad_input2.data)

        return grad_input1, grad_input2

class Warp(Module):

    def forward(self, input1, input2):
        input1_c = input1.contiguous()
        input2_c = input2.contiguous()
        return WarpFunction.apply(input1_c, input2_c)

if __name__ == '__main__':

    m = Warp()

    # test warping function
    input1 = torch.zeros(1, 1, 5, 5).cuda()
    input1[0, 0, 2, 2] = 1.0
    input1.requires_grad = True

    print('input map: ')
    print(input1)

    flow = torch.zeros(1, 2, 5, 5).cuda()
    flow[0, 0, 2, 2] = 1.0
    flow[0, 1, 2, 2] = 0.3
    flow.requires_grad = True

    out = m(input1, flow)
    qua = out**2

    print('output map')
    print(qua)

    loss = torch.sum(qua)
    loss.backward()

    print(out.grad)
    print(input1.grad)
    print(flow.grad)

