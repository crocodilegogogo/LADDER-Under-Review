import torch
from torch.autograd import Function

class DWTFunction_1D(Function):
    @staticmethod
    def forward(ctx, input, matrix_Low, matrix_High):
        ctx.save_for_backward(matrix_Low, matrix_High)
        for i in range(1):
            L0 = torch.matmul(input[:, :, i, :], matrix_Low.t())
            H0 = torch.matmul(input[:, :, i, :], matrix_High.t())
            L0 = L0.unsqueeze(2)
            H0 = H0.unsqueeze(2)
            if i == 0:
                L = L0
                H = H0
            else:
                L = torch.cat((L, L0), 2)
                H = torch.cat((H, H0), 2)
        return L, H
    @staticmethod
    def backward(ctx, grad_L, grad_H):
        matrix_L, matrix_H = ctx.saved_variables
        grad_input = torch.add(torch.matmul(grad_L, matrix_L), torch.matmul(grad_H, matrix_H))
        return grad_input, None, None


class DWTFunction_1D_L(Function):
    @staticmethod
    def forward(ctx, input, matrix_Low, matrix_High):
        ctx.save_for_backward(matrix_Low, matrix_High)
        for i in range(1):
            L0 = torch.matmul(input[:, :, i, :], matrix_Low.t())
            L0 = L0.unsqueeze(2)
            if i == 0:
                L = L0
            else:
                L = torch.cat((L, L0), 2)
        return L
    @staticmethod
    def backward(ctx, grad_L):
        matrix_L, matrix_H = ctx.saved_variables
        grad_input = torch.matmul(grad_L, matrix_L)
        return grad_input, None, None

class DWTFunction_1D_H(Function):
    @staticmethod
    def forward(ctx, input, matrix_Low, matrix_High):
        ctx.save_for_backward(matrix_Low, matrix_High)
        for i in range(1):
            H0 = torch.matmul(input[:, :, i, :], matrix_High.t())
            H0 = H0.unsqueeze(2)
            if i == 0:
                H = H0
            else:
                H = torch.cat((H, H0), 2)
        return H
    @staticmethod
    def backward(ctx, grad_H):
        matrix_L, matrix_H = ctx.saved_variables
        grad_input = torch.matmul(grad_H, matrix_H)
        return grad_input, None, None


class IDWTFunction_1D(Function):
    @staticmethod
    def forward(ctx, input_L, input_H, matrix_L, matrix_H):
        ctx.save_for_backward(matrix_L, matrix_H)
        output = torch.add(torch.matmul(input_L, matrix_L), torch.matmul(input_H, matrix_H))
        return output
    @staticmethod
    def backward(ctx, grad_output):
        matrix_L, matrix_H = ctx.saved_variables
        grad_L = torch.matmul(grad_output, matrix_L.t())
        grad_H = torch.matmul(grad_output, matrix_H.t())
        return grad_L, grad_H, None, None