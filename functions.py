import numpy as np
from scipy import ndimage
from scipy.signal import convolve


    # Args:
    #     input: shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
    #     W: weight, shape = c_out (#output channel) x c_in (#input channel) x k (#kernel_size) x k (#kernel_size)
    #     b: bias, shape = c_out
    #     kernel_size: size of the convolving kernel (or filter)
    #     pad: number of zero added to both sides of input
    # Returns:
    #     output: shape = n (#sample) x c_out (#output channel) x h_out x w_out,
    #         where h_out, w_out is the height and width of output, after convolution
    
def conv2d_forward(input, W, b, kernel_size, pad):
    N, c, h, w = input.shape
    K, c, H, Wk = W.shape

    #stride 1 for now...default?
    stride = 1 
    h_out = int((h + 2 * pad - H)/stride + 1)
    w_out = int((w + 2 * pad - Wk)/stride + 1)

    input_pad = zero_padding(input, pad)
    conv_out = np.zeros([N,K,h_out,w_out])
    
    for n in range(N):
        for channel in range(K):
            conv_out[n,channel,:,:] = np.sum(convolve(input_pad[n,:,:,:], np.flip(W[channel,:,:,:],axis=(0,1,2)), mode='valid'),axis=0) + b[channel] 
    return conv_out


#     Args:
#         input: shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
#         grad_output: shape = n (#sample) x c_out (#output channel) x h_out x w_out
#         W: weight, shape = c_out (#output channel) x c_in (#input channel) x k (#kernel_size) x k (#kernel_size)
#         b: bias, shape = c_out
#         kernel_size: size of the convolving kernel (or filter)
#         pad: number of zero added to both sides of input
#     Returns:
#         grad_input: gradient of input, shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
#         grad_W: gradient of W, shape = c_out (#output channel) x c_in (#input channel) x k (#kernel_size) x k (#kernel_size)
#         grad_b: gradient of b, shape = c_out

def conv2d_backward(input, grad_output, W, b, kernel_size, pad):

    N, c, h, w = input.shape
    K, c, Hk, Wk = W.shape
    N, K, h_out, w_out = grad_output.shape
    
    grad_pad = zero_padding(grad_output,pad)

    dx = np.zeros(input.shape)
    dw = np.zeros(W.shape)
    db = np.zeros(b.shape)

    for n in range(N):
        for channel in range(c):
            dx[n, channel, :, :] = np.sum(convolve(grad_pad[n, :, :, :], np.flip(W[:, channel, :, :],axis=0), mode='valid'),axis=0)

    flip90g = np.flip(grad_output,0)
    flip180g = np.flip(flip90g,1)

    for x in range(K):
        for y in range(c):
            dw[x, y, :, :] = convolve(input[:, y, :, :], flip180g[:, x, :, :], mode='valid')
    dw /= N

    for channel in range(K):
        db[channel] = (1/N) * np.sum(grad_output[:,channel,:,:])

    # db = (1/N)*np.sum(grad_output, axis=(0,1,2), keepdims=True)

    return dx,dw,db


def zero_padding(x, size):
    return np.pad(x, ((0,0), (0,0), (size, size), (size, size)), 'constant', constant_values=0)

    # Args:
    #     input: shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
    #     kernel_size: size of the window to take average over
    #     pad: number of zero added to both sides of input

    # Returns:
    #     output: shape = n (#sample) x c_in (#input channel) x h_out x w_out,
    #         where h_out, w_out is the height and width of output, after average pooling over input

def avgpool2d_forward(input, kernel_size, pad):

    N, c, h, w = input.shape

    h_out = (1 + (h - kernel_size) // kernel_size)
    w_out = (1 + (w - kernel_size) // kernel_size)

    input_pad = zero_padding(input, pad)
    conv_out = np.zeros((N,c,h_out,w_out))

    for n in range(N):
        for channel in range(c):
            conv_out[n, channel, :, :] = input_pad[n,channel,:h_out * kernel_size, :w_out * kernel_size].reshape(h_out, kernel_size, w_out, kernel_size).mean(axis=(1, 3))
    return conv_out


    # #print('avgpool forward')
    # # todo add zero padding, before or after avg_pooling?
    # if(pad > 0):
    #     solution = zero_padding(input, pad)

    # n, c, h, w = input.shape
    # solution = input.reshape(n, c, h//kernel_size, kernel_size, w//kernel_size, kernel_size).mean(axis=(3,5))   
    # h_out = solution.shape[2]
    # w_out = solution.shape[3]
    # #print(solution.shape)
    # # #print('done')
    # return solution

# Args:
#         input: shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
#         grad_output: shape = n (#sample) x c_in (#input channel) x h_out x w_out
#         kernel_size: size of the window to take average over
#         pad: number of zero added to both sides of input

#     Returns:
#         grad_input: gradient of input, shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
def avgpool2d_backward(input, grad_output, kernel_size, pad):
    # #print('avgpool backward')
    # fmap = np.repeat(np.repeat(input, kernel_size, axis=2), kernel_size, axis=3)
    # dmap = np.repeat(np.repeat(grad_output, kernel_size, axis=2), kernel_size, axis=3)
    # dx = np.zeros(input.shape)
    # dx = (fmap == input) * dmap
    # #print(dx.shape)
    # #print('done')
    # return dx
    N, c, h, w = input.shape
    N, c, hG, wG = grad_output.shape
    dx = np.zeros(input.shape)
    grad_input = np.kron(grad_output, np.ones((kernel_size, kernel_size)))
    return grad_input
